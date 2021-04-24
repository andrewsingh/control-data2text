import datasets
import spacy
import os
import json
import numpy as np

from spacy.symbols import ORTH
from retriever import Retriever
from pathlib import Path
from tqdm import tqdm


class TottoRetriever(Retriever):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.tokenizer.add_special_case(Retriever.MASK_STR, [{ORTH: Retriever.MASK_STR}])


    def mask_target(self, source, target):
        def hasNumbers(s):
            return any(char.isdigit() for char in s)

        keep_pos = ["AUX", "ADP", "DET", "SCONJ", "CCONJ", "PRON", "PART", "PUNCT", "VERB", "ADV"]
        target_doc = nlp(target)
        final_toks = []
        upper_flag = False
        for i, tok in enumerate(target_doc):
            word = tok.text
            if word[0].isupper() and (i > 0 or tok.pos_ == "PROPN"):
                if not upper_flag:
                    upper_flag = True
                    final_toks.append(Retriever.MASK_STR)
            else:
                if upper_flag:
                    upper_flag = False
                if tok.pos_ not in keep_pos and tok.text.lower() in source.lower():
                    final_toks.append(Retriever.MASK_STR)
                else:
                    final_toks.append(word)
                
        final_toks = [tok if not hasNumbers(tok) else Retriever.MASK_STR for tok in final_toks]
        return " ".join(final_toks)


    def create_eval_set(self, out_json, eval_k=40, gpu=None):
        dataset = datasets.load_from_disk(self.dataset_path)
        if "source_embed" not in dataset[0]:
            print("Source embeddings not in index, adding them now")
            self.add_embeds("source", gpu=gpu)
            dataset = datasets.load_from_disk(self.dataset_path)
            print("Source embeddings added, creating eval dataset")

        dataset.add_faiss_index(column="source_embed")
        examples = []
        
        for i in tqdm(range(len(dataset))):
            results = dataset.get_nearest_examples("source_embed", np.array(dataset[i]["source_embed"], dtype=np.float32), k=eval_k)

            proto_start = 1
            while results[0][proto_start] == 0:
                print("\n[{}] low score: {}".format(i, results[0][proto_start]))
                proto_start += 1

            example = {}
            example["source"] = (results[1]["target"][proto_start] + " [SEP] " + dataset[i]["source"]).strip()
            example["target"] = dataset[i]["target"].strip()
            examples.append(example)

        os.makedirs(os.path.dirname(out_json), exist_ok=True)

        with open(out_json, "w+") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
            print(f"Wrote eval data to file: {out_json}")

