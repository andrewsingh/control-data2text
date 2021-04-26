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
    nlp = spacy.load("en_core_web_lg")

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


    def write_eval_set(self, out_json, retrieval_embed="source_embed", eval_k=40, gpu=None):
        self.dataset = datasets.load_from_disk(self.dataset_path)
        if retrieval_embed not in self.dataset[0]:
            print("Retrieval embeddings not in index, adding them now")
            self.add_embeds(retrieval_embed.replace("_embed", ""), gpu=gpu)
            print("Retrieval embeddings added, creating eval dataset")

        self.dataset.add_faiss_index(column=retrieval_embed)
        examples = []
        
        for i in tqdm(range(len(self.dataset))):
            results = self.dataset.get_nearest_examples(retrieval_embed, np.array(self.dataset[i][retrieval_embed], dtype=np.float32), k=eval_k)

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

