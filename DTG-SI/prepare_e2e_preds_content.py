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
    "input_path", None,
    "The path to the predictions file")
flags.DEFINE_string(
    "output_path", None,
    "The directory to output csv files")
flags.DEFINE_string(
    "e2e_data_dir", f"{os.getenv('CTRL_D2T_ROOT')}/DTG-SI/e2e_data/val",
    "The data directory")

FLAGS = flags.FLAGS

refs = ['', '_ref']


def tsv_to_csv(input_file):
    output_file = f"{os.path.splitext(input_file)[0]}.csv"
    output = []
    with open(input_file, 'r') as f:
        output = ['"sentence","label"\n'] + ['"{}","0"\n'.format(line.strip()) for line in f]
    with open(output_file, "w+") as f:
        f.writelines(output)
    os.remove(input_file)


def prepare_data(input_path, output_path):
    """
    Builds the model and runs.
    """
    # Loads data
    print("Preparing predictions for content evaluation")

    e2e_data_dir = FLAGS.e2e_data_dir
    data_dir = 'bert/{}'.format('e2e_preparation')
    output_path_1 = f"{output_path}/validation_preds_content.1.tsv"
    output_path_2 = f"{output_path}/validation_preds_content.2.tsv"


    #TO DO:Prepare data for the transformer classifier
    #i.e. Concat x' with y and see whether x' was compressed in y
    ref = refs[1]
    with open(os.path.join(e2e_data_dir, "x{}_type.valid.txt".format(ref)),'r') as f_type, \
        open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(ref)),'r') as f_entry,\
        open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(refs[0])), 'r') as f_entry_x,\
        open(input_path, 'r') as f_sent:

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
                    with open(output_path_2, 'a') as f_w:
                        f_w.write(neg_samp)

    # Concat x with y and see whether x was compressed in y
    ref = refs[0]
    with open(os.path.join(e2e_data_dir, "x{}_type.valid.txt".format(ref)),'r') as f_type,\
        open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(ref)),'r') as f_entry, \
        open(input_path, 'r') as f_sent:

        lines_type = f_type.readlines()
        lines_entry = f_entry.readlines()
        lines_sent = f_sent.readlines()
        for (idx_line, line_type) in enumerate(lines_type):
            line_type = line_type.strip('\n').split(' ')
            for (idx_val, attr) in enumerate(line_type):
                entry_list = lines_entry[idx_line].strip('\n').split(' ')
                pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
                with open(output_path_1, 'a') as f_w:
                    f_w.write(pos_samp)

    print("Converting from TSV to CSV")
    tsv_to_csv(output_path_1)
    tsv_to_csv(output_path_2)
    print("Data preparation complete")


def main(_):
    """ Starts the data preparation
    """
    prepare_data(FLAGS.input_path, FLAGS.output_path)
    

if __name__ == "__main__":
  app.run(main)