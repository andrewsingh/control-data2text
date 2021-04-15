# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Produces TFRecords files and modifies data configuration file
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import nltk
# pylint: disable=no-name-in-module
sys.path.append('./bert/utils')
from absl import app
from absl import flags
# pylint: disable=intest-name, too-many-locals, too-many-statements
nltk.download('punkt')
flags.DEFINE_string(
    "task", "SST",
    "The task to run experiment on. ")
flags.DEFINE_string(
    "vocab_file", 'bert/bert_config/all.vocab.txt',
    "The one-wordpiece-per-line vocabary file directory.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maxium length of sequence, longer sequence will be trimmed.")
flags.DEFINE_string(
    "tfrecords_output_dir", 'bert/e2e_preparation',
    "The output directory where the TFRecords files will be generated.")
flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_string(
    "save_path", "e2e_output",
    "The saved directory during training, such as `e2e_ours` or `e2e_s2s`.")
flags.DEFINE_string(
    "step", "0",
    "The training step  you'd like to evaluate")
flags.DEFINE_string(
    "output_path", f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/text-classification/test_data/e2e_generations",
    "The directory to output csv files")

FLAGS = flags.FLAGS

e2e_data_dir = "e2e_data/val"
refs = ['', '_ref']


def tsv_to_csv(input_file, output_file):
    output = []
    with open(input_file, 'r') as f:
        output = ['"sentence","label"\n'] + ['"{}","0"\n'.format(line.strip()) for line in f]
    with open(output_file, "w+") as f:
        f.writelines(output)
    # os.remove(input_file)


def prepare_data(save_path, step, output_path):
    """
    Builds the model and runs.
    """
    # Loads data
    print("Loading data")

    # task_datasets_rename = {
    #     "SST": "E2E",
    # }
    
    data_dir = 'bert/{}'.format('e2e_preparation')
    # if FLAGS.task.upper() in task_datasets_rename:
    #     data_dir = 'data/{}'.format(
    #         task_datasets_rename[FLAGS.task])

    #TO DO:Prepare data for the transformer classifier
    #i.e. Concat x' with y and see whether x' was compressed in y
    ref = refs[1]
    with open(os.path.join(e2e_data_dir, "x{}_type.valid.txt".format(ref)),'r') as f_type, \
        open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(ref)),'r') as f_entry,\
        open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(refs[0])), 'r') as f_entry_x,\
        open("{}/ckpt/hypos.step{}.val.txt".format(save_path, step), 'r') as f_sent:

        lines_type = f_type.readlines()
        lines_entry = f_entry.readlines()
        lines_entry_x = f_entry_x.readlines()
        lines_sent = f_sent.readlines()

        for (idx_line, line_type) in enumerate(lines_type):
            line_type = line_type.strip('\n').split(' ')
            for (idx_val, attr) in enumerate(line_type):
                entry_list = lines_entry[idx_line].strip('\n').split(' ')
                if (lines_entry_x[idx_line].find(entry_list[idx_val]) == -1):
                    neg_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
                    with open("bert/e2e_preparation/{}.step{}.2.tsv".format(save_path, step), 'a') as f_w:
                        f_w.write(neg_samp)

    # Concat x with y and see whether x was compressed in y
    ref = refs[0]
    with open(os.path.join(e2e_data_dir, "x{}_type.valid.txt".format(ref)),'r') as f_type,\
        open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(ref)),'r') as f_entry, \
        open("{}/ckpt/hypos.step{}.val.txt".format(save_path, step), 'r') as f_sent:

        lines_type = f_type.readlines()
        lines_entry = f_entry.readlines()
        lines_sent = f_sent.readlines()
        for (idx_line, line_type) in enumerate(lines_type):
            line_type = line_type.strip('\n').split(' ')
            for (idx_val, attr) in enumerate(line_type):
                entry_list = lines_entry[idx_line].strip('\n').split(' ')
                pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
                with open("bert/e2e_preparation/{}.step{}.1.tsv".format(save_path, step), 'a') as f_w:
                    f_w.write(pos_samp)

    tsv_to_csv(f"bert/e2e_preparation/{save_path}.step{step}.1.tsv", f"{output_path}/{save_path}.step{step}.1.csv")
    tsv_to_csv(f"bert/e2e_preparation/{save_path}.step{step}.2.tsv", f"{output_path}/{save_path}.step{step}.2.csv")
    print("Data preparation complete")



    
def main(_):
    """ Starts the data preparation
    """
    prepare_data(FLAGS.save_path, FLAGS.step, FLAGS.output_path)
    


if __name__ == "__main__":
  app.run(main)