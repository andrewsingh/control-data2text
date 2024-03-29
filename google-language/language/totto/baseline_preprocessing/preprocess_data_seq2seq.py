# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Augments json files with table linearization used by baselines.

Note that this code is merely meant to be starting point for research and
there may be much better table representations for this task.
"""
import copy
import json

from absl import app
from absl import flags

from language.totto.baseline_preprocessing import preprocess_utils

import six

flags.DEFINE_string("input_path", None, "Input json file.")
flags.DEFINE_string("output_path", None, "Output directory.")
flags.DEFINE_bool("with_headers_only", False, "Include additional column in data with only headers and section title of table.")

FLAGS = flags.FLAGS


def _generate_processed_examples(input_path, with_headers_only=False):
    """Generate TF examples."""
    linearized_examples = []
    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if len(linearized_examples) % 1000 == 0:
                print("Num examples processed: %d" % len(linearized_examples))

            line = six.ensure_text(line, "utf-8")
            json_example = json.loads(line)
            table = json_example["table"]
            table_page_title = json_example["table_page_title"]
            table_section_title = json_example["table_section_title"]
            cell_indices = json_example["highlighted_cells"]

            subtable = (
                preprocess_utils.get_highlighted_subtable(
                    table=table,
                    cell_indices=cell_indices,
                    with_heuristic_headers=True))

            subtable_metadata_str = (
                preprocess_utils.linearize_subtable(
                    subtable=subtable,
                    table_page_title=table_page_title,
                    table_section_title=table_section_title))

            linearized_example = {}
            linearized_example["source"] = subtable_metadata_str
            linearized_example["target"] = json_example["sentence_annotations"][0]["final_sentence"]
            if with_headers_only:
                subtable_headers_only_metadata_str = (
                    preprocess_utils.linearize_subtable(
                        subtable=subtable,
                        table_page_title=None,
                        table_section_title=table_section_title,
                        headers_only=True))
                linearized_example["source_headers_only"] = subtable_headers_only_metadata_str

            linearized_examples.append(linearized_example)

    print("Num examples processed: %d" % len(linearized_examples))
    return linearized_examples


def main(_):
    processed_json_examples = _generate_processed_examples(FLAGS.input_path, with_headers_only=FLAGS.with_headers_only)
    with open(FLAGS.output_path, "w+", encoding="utf-8") as f:
        for json_example in processed_json_examples:
            f.write(json.dumps(json_example) + "\n")
    

if __name__ == "__main__":
    app.run(main)
