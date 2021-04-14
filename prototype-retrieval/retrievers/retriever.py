import datasets
import os
import editdistance
import random

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class Retriever():

    mask_str = "[MASK]"
    root = os.getenv("ROOT")
    
    def __init__(self, split, index_path, data_dir, proto_data_dir):
        self.split = split
        self.src_file = Path(data_dir).joinpath(split + ".source")
        self.tgt_file = Path(data_dir).joinpath(split + ".target")
        self.index_path = index_path
        self.proto_data_dir = proto_data_dir
        

    def mask_target(self, source, target):
        pass


    def mask_dataset(self, num_proc=1):
        data_sep = datasets.load_dataset("text", data_files={"train": self.src_file, "test": self.tgt_file})
        data_dict = {"source": data_sep["train"]["text"], "target": data_sep["test"]["text"]}
        data = datasets.Dataset.from_dict(data_dict)

        def mask_and_split(example):
            masked_target = self.mask_target(example["source"], example["target"])
            split_masked_target = masked_target.split()
            return {"masked_target": masked_target, "split_masked_target": split_masked_target}

        masked_data = data.map(mask_and_split, num_proc=num_proc)
        masked_data.save_to_disk(self.index_path)


    def add_edit_dist_maps(self, edit_dist_thresh=50, edit_dist_index_size=50):
        data = datasets.load_from_disk(self.index_path)
        
        def add_edit_dist_map(example, num_proc=1):
            dist_map = defaultdict(lambda: [])
            a = example["split_masked_target"]
            for i, b in enumerate(data["split_masked_target"]):
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
            return {"{}_edit_dist_map".format(compare_split): edit_dist_map}

        data = data.map(add_edit_dist_maps, num_proc=num_proc)
        data.save_to_disk(self.index_path)


    def add_embed_index(self, feature, sentence_encoder_name="stsb-distilbert-base", gpu=None):
        data = datasets.load_from_disk(self.index_path)
        sentence_encoder = SentenceTransformer(sentence_encoder_name)
        if gpu:
            sentence_encoder.to("cuda:{}".format(gpu))
        
        def add_embed(example):
            feature_embed = sentence_encoder.encode(example[feature])
            return {"{}_embed".format(feature): feature_embed}

        data = data.map(add_embed)
        data.save_to_disk(self.index_path)


    def create_train_set(self, retrieval_k=5, max_edit_dist=None):
        lines = []
        new_target_lines = []
        data = datasets.load_from_disk(self.index_path)

        if "edit_dist_map" not in data[0]:
            print("Edit distance maps not in index, adding them now...")
            self.add_edit_dist_maps()
            data = datasets.load_from_disk(self.index_path)

        if not max_edit_dist:
            max_edit_dist = len(data[0]["edit_dist_map"])
        
        for i in tqdm(range(len(data))):
            chosen_indices = []
            for (edit_dist, indices) in list(enumerate(data[i]["edit_dist_map"]))[:max_edit_dist + 1]:
                if indices:
                    remaining_indices = retrieval_k - len(chosen_indices)
                    if len(indices) <= remaining_indices:
                        chosen_indices += indices
                    else:
                        chosen_indices += random.sample(indices, remaining_indices)
                        break
            
            for proto_idx in chosen_indices:
                lines.append(data[proto_idx]["target"] + " [SEP] " + data[i]["source"] + "\n")

            entry_target = data[i]["target"]
            num_targets = len(chosen_indices)
            if "\n" in entry_target:
                new_target_lines += ([entry_target] * num_targets)
            else:
                new_target_lines += ([entry_target + "\n"] * num_targets)

        if not os.path.exists(self.proto_data_dir):
            os.mkdir(self.proto_data_dir)

        with open("{}/{}.source".format(self.proto_data_dir, self.split), "w+") as f:
            f.writelines(lines)

        with open("{}/{}.target".format(self.proto_data_dir, self.split), "w+") as f:
            f.writelines(new_target_lines)


    def create_eval_set(self, eval_k):
        pass



