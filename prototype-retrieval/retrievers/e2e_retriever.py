import datasets
import spacy
import os

from spacy.symbols import ORTH
from retriever import Retriever
from pathlib import Path


class E2ERetriever(Retriever):

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.tokenizer.add_special_case(Retriever.MASK_STR, [{ORTH: Retriever.MASK_STR}])


    def mask_target(self, values, target):
        def hasNumbers(s):
            return any(char.isdigit() for char in s)

        target = target.lower()
        for value in values.split():
            target = target.replace(value.lower(), Retriever.MASK_STR)
            target = target.replace(value.replace("_", " "), Retriever.MASK_STR)

        final_toks = []
        for word in target.split(" "):
            if hasNumbers(word) or Retriever.MASK_STR in word:
                final_toks.append(Retriever.MASK_STR)
            # elif word not in ["$", "£"] and len(word) >= 1:
            else:
                final_toks.append(word)
        return " ".join(final_toks)


    def create_eval_set(self, out_dir, eval_k):
        pass


