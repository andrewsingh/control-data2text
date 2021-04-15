"""
Utilities.
"""

import collections
import os
from data2text.data_utils import extract_entities


dataset = 'e2e'

# load all entities
all_ents = set()
with open(f"{os.getenv('CTRL_D2T_ROOT')}/DTG-SI/e2e_data/x_value.vocab.txt", "r") as f:
    all_vocb = f.readlines()
    for vocab in all_vocb:
        all_ents.add(vocab.strip('\n'))
    

x_fields = ['value', 'type', 'associated']
x_strs = ['x', 'x_ref']
y_strs = ['y_aux', 'y_ref']
ref_strs = ['', '_ref']


def replace_data_in_sent(sent, token="<UNK>"):
    datas = extract_entities(sent, all_ents)
    datas.sort(key=lambda data: data.start, reverse=True)
    for data in datas:
        sent[data.start] = token
    return sent
        
def mask_refs_preds(list_of_references, hypotheses):
    hypotheses = list(map(str.split, hypotheses))
    list_of_references = [
        list(map(replace_data_in_sent, refs))
        for refs in list_of_references]
    hypotheses = list(map(replace_data_in_sent, hypotheses))
    out_refs = [" ".join(ref[0]) for ref in list_of_references]
    out_hypos = [" ".join(hypo) for hypo in hypotheses]
    return out_refs, out_hypos

def read_sents_from_file(file_name):
    with open(file_name, 'r') as f:
        return list(map(str.split, f))



