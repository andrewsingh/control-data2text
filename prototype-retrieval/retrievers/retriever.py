import datasets
import os
import editdistance
import random
import json

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer


class Retriever():

    MASK_TOKEN = "[MASK]"
    
    def __init__(self, dataset_path, data_json=None):
        self.dataset_path = dataset_path
        if os.path.exists(dataset_path):
            self.dataset = datasets.load_from_disk(self.dataset_path)
        elif os.path.exists(data_json):
            print(f"Creating dataset from JSON file: {os.path.basename(data_json)}")
            self.dataset = datasets.load_dataset("json", data_files=data_json)["train"]
            self.dataset.save_to_disk(self.dataset_path)
        else:
            raise FileNotFoundError("Could not find path to Hugging Face dataset or JSON file")


    @classmethod
    def mask_target(cls, source, target):
        pass


    def add_masked_targets(self, num_proc=1):
        def add_masked_target(example):
            masked_target = self.__class__.mask_target(example["source"], example["target"])
            split_masked_target = masked_target.split()
            return {"masked_target": masked_target, "split_masked_target": split_masked_target}

        self.dataset = self.dataset.map(add_masked_target, num_proc=num_proc)
        self.dataset.save_to_disk(self.dataset_path)


    def add_edit_dist_maps(self, retrieval_path=None, map_name="edit_dist_map", edit_dist_thresh=50, edit_dist_index_size=25, num_proc=1):        
        if retrieval_path:
            retrieval_dataset = datasets.load_from_disk(retrieval_path)
        else:
            retrieval_dataset = self.dataset
        
        def add_edit_dist_map(example):
            dist_map = defaultdict(lambda: [])
            a = example["split_masked_target"]
            for i, b in enumerate(retrieval_dataset["split_masked_target"]):
                edit_dist = editdistance.eval(a, b)
                dist_map[edit_dist].append(i)
            dist_map.pop(0, None)
            dist_list = sorted(dist_map.items())
            trunc_idx = 0
            total_examples = 0
            for (i, (_, indices)) in enumerate(dist_list):
                total_examples += len(indices)
                if total_examples >= edit_dist_thresh:
                    trunc_idx = i + 1
                    break
            dist_list = dist_list[:trunc_idx]
            edit_dist_map = [None] * (edit_dist_index_size + 1)
            for dist, indices in dist_list:
                if dist <= edit_dist_index_size:
                    edit_dist_map[dist] = indices
            
            return {map_name: edit_dist_map}

        self.dataset = self.dataset.map(add_edit_dist_map, num_proc=num_proc)
        self.dataset.save_to_disk(self.dataset_path)


    def add_embeds(self, feature, sentence_encoder_name="stsb-distilbert-base", gpu=None):
        sentence_encoder = SentenceTransformer(sentence_encoder_name)
        if gpu:
            sentence_encoder.to("cuda:{}".format(gpu))
        
        def add_embed(example):
            feature_embed = sentence_encoder.encode(example[feature])
            return {"{}_embed".format(feature): feature_embed}

        self.dataset = self.dataset.map(add_embed)
        self.dataset.save_to_disk(self.dataset_path)


    def write_train_set(self, out_json, retrieval_path=None, map_name="edit_dist_map", retrieval_k=5, max_edit_dist=None, weighted=False, seed=42):
        if retrieval_path:
            retrieval_dataset = datasets.load_from_disk(retrieval_path)
        else:
            retrieval_dataset = self.dataset
        
        random.seed(seed)
        lines = []
        new_target_lines = []
        examples = []

        if map_name not in self.dataset[0]:
            if not retrieval_path:
                print("Edit distance maps not in index, adding them now.")
                self.add_edit_dist_maps()
            else:
                print("Error: edit distance maps not in index, pleased add them and try again.")
                return
            

        if not max_edit_dist:
            max_edit_dist = len(self.dataset[0][map_name])
        
        for i in tqdm(range(len(self.dataset))):
            chosen_indices = []
            if weighted:
                chosen_edit_dists = []
            for (edit_dist, indices) in list(enumerate(self.dataset[i][map_name]))[:max_edit_dist + 1]:
                if indices:
                    remaining_indices = retrieval_k - len(chosen_indices)
                    if len(indices) <= remaining_indices:
                        chosen_indices += indices
                        if weighted:
                            chosen_edit_dists += [edit_dist] * len(indices)
                    else:
                        chosen_indices += random.sample(indices, remaining_indices)
                        if weighted:
                            chosen_edit_dists += [edit_dist] * remaining_indices
                        break

            if weighted:
                assert(len(chosen_indices) == len(chosen_edit_dists))

            if len(chosen_indices) > 0:
                if weighted:
                    example = {}
                    example["source"] = self.dataset[i]["source"].strip()
                    example["target"] = self.dataset[i]["target"].strip()
                    example["prototypes"] = " [SEP] ".join([retrieval_dataset[proto_idx]["target"] for proto_idx in chosen_indices])
                    example["edit_dists"] = chosen_edit_dists
                    examples.append(example)
                else:
                    for proto_idx in chosen_indices:
                        example = {}
                        example["source"] = (retrieval_dataset[proto_idx]["target"] + " [SEP] " + self.dataset[i]["source"]).strip()
                        example["target"] = self.dataset[i]["target"].strip()
                        examples.append(example)

        os.makedirs(os.path.dirname(out_json), exist_ok=True)

        with open(out_json, "w+") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
            print(f"Wrote training data to file: {out_json}")


    def write_eval_set(self, out_dir, eval_k):
        pass



