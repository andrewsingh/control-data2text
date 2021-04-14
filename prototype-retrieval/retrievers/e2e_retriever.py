import datasets
import spacy
import os

from spacy.symbols import ORTH
from retriever import Retriever
from pathlib import Path


class E2ERetriever(Retriever):

    def __init__(self, split, index_path, data_dir=f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/e2e", proto_data_dir=None):
        if not proto_data_dir:
            proto_data_dir = str(Path(Path(data_dir).parent, "prototypes"))
        super().__init__(split, index_path, data_dir, proto_data_dir)
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp.tokenizer.add_special_case(Retriever.mask_str, [{ORTH: Retriever.mask_str}])


    def mask_target(self, values, target):
        def hasNumbers(s):
            return any(char.isdigit() for char in s)

        target = target.lower()
        for value in values.split():
            target = target.replace(value.lower(), mask_str)
            target = target.replace(value.replace("_", " "), mask_str)

        final_toks = []
        for word in target.split(" "):
            if hasNumbers(word) or mask_str in word:
                final_toks.append(mask_str)
            # elif word not in ["$", "£"] and len(word) >= 1:
            else:
                final_toks.append(word)
        return " ".join(final_toks)


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


    def create_eval_set(self, eval_k):
        pass


