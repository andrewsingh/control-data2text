# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import transformers
import math
import json
import pandas as pd
from tqdm import tqdm
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    T5Tokenizer
)
from sacremoses import MosesDetokenizer


# %%
root_dir = "/projects/ogma2/users/andrewsi/control-data2text"
model_path = f"{root_dir}/transformers/examples/language-modeling/exp/e2e_targets/gpt2-01/checkpoint-7458"
gpu = "3"

def compute_perplexity(preds):
    e2e_lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    e2e_lm = GPT2LMHeadModel.from_pretrained(model_path)
    e2e_lm.to(f"cuda:{gpu}")
    ppls = []
    for pred in tqdm(preds):
        inputs = e2e_lm_tokenizer(pred, return_tensors='pt').to("cuda:3")
        outputs = e2e_lm(**inputs, labels=inputs['input_ids'])
        ppls.append(math.exp(outputs.loss))
    return round((sum(ppls) / len(ppls)), 4)


# %%
md = MosesDetokenizer(lang='en')

def process_e2e_preds(inpath, outpath=None):
    processed_lines = []
    with open(inpath, "r") as f:
        original_lines = [line for line in f]
    for line in tqdm(original_lines):
        processed_lines.append(md.detokenize(line.strip().replace("_", " ").split()))
    if outpath:
        with open(outpath, "w+") as f:
            f.writelines([line + "\n" for line in processed_lines])
    return processed_lines


# %%
train_file = "/projects/ogma2/users/andrewsi/control-data2text/DTG-SI/e2e_data/train/y_aux.train.txt"
val_file = "/projects/ogma2/users/andrewsi/control-data2text/DTG-SI/e2e_data/val/y_aux.valid.txt"
test_file = "/projects/ogma2/users/andrewsi/control-data2text/DTG-SI/e2e_data/test/y_aux.test.txt"
outdir = "/projects/ogma2/users/andrewsi/control-data2text/transformers/examples/language-modeling/test_data/e2e_targets"

# %%
process_e2e_preds(test_file, f"{outdir}/test.txt")


# %%
def get_new_ppl(inpath):
    return compute_perplexity(process_e2e_preds(inpath))


# %%
def get_prop_longer(col, thresh):
    return len(col[col > thresh]) / len(col)

def get_len_df(data_file):
    with open(data_file, "r") as f:
        data_lines = [line for line in f]
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=4096)
    special_tokens = ["[SEP]"]
        
    if len(special_tokens) > 0:
        special_tokens_dict = {"additional_special_tokens": (special_tokens)}
        tokenizer.add_special_tokens(special_tokens_dict)
    print("\nTokenizer length: {}".format(len(tokenizer)))
    
    src_lens = []
    tgt_lens = []
    print(f"Num lines: {len(data_lines)}\nFirst line: {data_lines[0]}")
    for line in tqdm(data_lines):
        json_example = json.loads(line)
        src_lens.append(len(tokenizer(json_example["source"], max_length=4096, truncation=True)['input_ids'])) 
        tgt_lens.append(len(tokenizer(json_example["target"], max_length=4096, truncation=True)['input_ids']))

    return pd.DataFrame([src_lens, tgt_lens], index=["src_len", "tgt_len"]).transpose()
# %%



