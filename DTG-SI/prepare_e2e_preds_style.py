import os

import sys
import nltk

from utils_e2e_eval import *
from absl import app
from absl import flags

flags.DEFINE_string(
    "refs_path", f"{os.getenv('CTRL_D2T_ROOT')}/DTG-SI/e2e_data/val/y_ref.valid.txt",
    "References path")
flags.DEFINE_string(
    "preds_path", None,
    "Model predictions path")
flags.DEFINE_string(
    "output_path", None,
    "Output directory path")

FLAGS = flags.FLAGS


def prepare_data(refs_path, preds_path, output_path):
    print("Masking references and predicitons")
    if not output_path:
        output_path = os.path.dirname(preds_path)
    refs = read_sents_from_file(refs_path)
    refs = [[ref] for ref in refs]
    with open(preds_path, "r") as f:
        preds = list(f)
    masked_refs, masked_preds = mask_refs_preds(refs, preds)
    refs_name = os.path.basename(refs_path)
    preds_name = os.path.basename(preds_path)
    with open(f"{output_path}/mask_{refs_name}", "w+") as f:
        f.writelines([ref + "\n" for ref in masked_refs])
    with open(f"{output_path}/mask_{preds_name}", "w+") as f:
        f.writelines([ref + "\n" for ref in masked_preds])
    print(f"Wrote masked references and predictions to directory: {output_path}")
    


def main(_):
    """ Starts the data preparation
    """
    prepare_data(FLAGS.refs_path, FLAGS.preds_path, FLAGS.output_path)
    


if __name__ == "__main__":
    app.run(main)