import datasets
import spacy
import os
import json
import numpy as np
import random

from spacy.symbols import ORTH
from retriever import Retriever
from pathlib import Path
from tqdm import tqdm


class TottoRetriever(Retriever):
    nlp = spacy.load("en_core_web_lg")
    special_tokens = ["<page_title>", "</page_title>", "<section_title>", "</section_title>", "<table>", "</table>", "<cell>", "</cell>", "<row_header>", "</row_header>", "<col_header>", "</col_header>"]


    def __init__(self, dataset_path, data_json=None):
        super().__init__(dataset_path, data_json)


    @classmethod
    def mask_target(cls, source, target, mask_token=Retriever.MASK_TOKEN):
        def hasNumbers(s):
            return any(char.isdigit() for char in s)

        keep_pos = ["AUX", "ADP", "DET", "SCONJ", "CCONJ", "PRON", "PART", "PUNCT", "VERB", "ADV"]
        target_doc = TottoRetriever.nlp(target)
        final_toks = []
        upper_flag = False
        for i, tok in enumerate(target_doc):
            word = tok.text
            if word[0].isupper() and (i > 0 or tok.pos_ == "PROPN"):
                if not upper_flag:
                    upper_flag = True
                    final_toks.append(mask_token)
            else:
                if upper_flag:
                    upper_flag = False
                if tok.pos_ not in keep_pos and tok.text.lower() in source.lower():
                    final_toks.append(mask_token)
                else:
                    final_toks.append(word)
                
        final_toks = [tok if not hasNumbers(tok) else mask_token for tok in final_toks]
        return " ".join(final_toks)


    def write_eval_set(self, out_json, query_embed="source_embed", retrieval_embed=None, retrieval_path=None, eval_k=40, gpu=None):
        if retrieval_path:
            retrieval_dataset = datasets.load_from_disk(retrieval_path)
        else:
            retrieval_dataset = self.dataset
            
        if query_embed == "random":
            examples = []
            for i in tqdm(range(len(self.dataset))):
                proto_idx = random.randint(0, len(retrieval_dataset) - 1)
                example = {}
                example["source"] = (retrieval_dataset[proto_idx]["target"] + " [SEP] " + self.dataset[i]["source"]).strip()
                example["target"] = self.dataset[i]["target"].strip()
                examples.append(example)
        else:
            if not retrieval_embed:
                retrieval_embed = query_embed
            if query_embed not in self.dataset[0]:
                print("Query embeddings not in own dataset, adding them now")
                self.add_embeds(retrieval_embed.replace("_embed", ""), gpu=gpu)
                print("Query embeddings added to own dataset")
            if retrieval_embed not in retrieval_dataset[0]:
                print("Error: retrieval embeddings not in retrieval dataset. Please add them and try again.")
                return

            retrieval_dataset.add_faiss_index(column=retrieval_embed)
            examples = []
            
            for i in tqdm(range(len(self.dataset))):
                results = retrieval_dataset.get_nearest_examples(retrieval_embed, np.array(self.dataset[i][query_embed], dtype=np.float32), k=eval_k)

                proto_start = 1
                if retrieval_embed == "source_embed":
                    while results[0][proto_start] == 0:
                        proto_start += 1
                else:
                    while results[1]["target"][proto_start] == self.dataset[i]["target"]:
                        proto_start += 1

                example = {}
                example["source"] = (results[1]["target"][proto_start] + " [SEP] " + self.dataset[i]["source"]).strip()
                example["target"] = self.dataset[i]["target"].strip()
                examples.append(example)

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w+") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
            print(f"Wrote eval data to file: {out_json}")


    def write_eval_set_from_protos(self, proto_txt, val_json, out_json):
        proto_f = open(proto_txt, "r")
        proto_lines = [line for line in proto_f]
        val_f = open(val_json, "r")
        val_lines = [line for line in val_f]
        assert(len(val_lines) == len(proto_lines))
        new_examples = []

        for i in tqdm(range(len(val_lines))):
            val_example = json.loads(val_lines[i])
            new_example = {}
            new_example["source"] = proto_lines[i].strip() + " [SEP] " + val_example["source"]
            new_example["target"] = val_example["target"]
            new_examples.append(new_example)

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w+") as f:
            for example in new_examples:
                f.write(json.dumps(example) + "\n")
            print(f"Wrote eval data to file: {out_json}")


