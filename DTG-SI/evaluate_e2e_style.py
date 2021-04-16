import os

import sys
import nltk
from nltk import word_tokenize

from utils_e2e_eval import *
from absl import app
from absl import flags

flags.DEFINE_string(
    "refs_path", f"{os.getenv('CTRL_D2T_ROOT')}/DTG-SI/e2e_data/val/y_ref.valid.txt",
    "References path")
flags.DEFINE_string(
    "preds_path", None,
    "Model predictions path")

FLAGS = flags.FLAGS


def compute_m_bleu(ref_file, pred_file):
    refs = read_refs_from_file(ref_file)
    refs = [[ref] for ref in refs]
    preds = [" ".join(word_tokenize(pred)) for pred in read_preds_from_file(pred_file)]
    bleu = corpus_bleu(refs, list(map(str.split, preds))) * 1e-2
    print("BLEU: {:.2%}".format(bleu))


def main(_):
    """ Starts the data preparation
    """
    compute_m_bleu(FLAGS.refs_path, FLAGS.preds_path)
    

if __name__ == "__main__":
    app.run(main)