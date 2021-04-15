import os

import sys
import nltk

from utils_e2e_eval import *
from absl import app
from absl import flags

flags.DEFINE_string(
    "preds_path", None,
    "Model predictions path")
flags.DEFINE_string(
    "refs_path", f"{os.getenv('CTRL_D2T_ROOT')}/DTG-SI/e2e_data/val/y_ref.valid.txt",
    "References path")

FLAGS = flags.FLAGS


def prepare_data(refs_path, preds_path):
    refs = read_sents_from_file(refs_path)
    refs = [[ref] for ref in refs]
    with open(preds_path, "r") as f:
        preds = list(f)
    masked_refs, masked_preds = mask_refs_preds(refs, preds))
    with open()

    


def main(_):
    """ Starts the data preparation
    """
    prepare_data(FLAGS.refs_path, FLAGS.preds_path)
    


if __name__ == "__main__":
  app.run(main)