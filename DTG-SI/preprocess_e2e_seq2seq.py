import json
import os

from pathlib import Path
from absl import app
from absl import flags

flags.DEFINE_string("input_dir", f"{os.getenv('CTRL_D2T_ROOT')}/DTG-SI/e2e_data/", "Input directory.")
flags.DEFINE_string("output_dir", f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/e2e", "Output directory.")
flags.DEFINE_string("split", None, "Dataset split: 'train', 'validation', or 'test'")
flags.DEFINE_bool("include_refs", False, "Whether to include style references in source (for controlled generation)")


FLAGS = flags.FLAGS


def preprocess_split(input_dir, split, include_refs=False):
    file_split = "valid" if split == "validation" else split
    dir_split = "val" if split == "validation" else split
    data_dir = input_dir + dir_split

    x_type_file = open(f"{data_dir}/x_type.{file_split}.txt", "r")
    x_value_file = open(f"{data_dir}/x_value.{file_split}.txt", "r")
    y_aux_file = open(f"{data_dir}/y_aux.{file_split}.txt", "r")

    x_type_lines = [line for line in x_type_file]
    x_value_lines = [line for line in x_value_file]
    y_aux_lines = [line for line in y_aux_file]

    if include_refs:
      y_ref_file = open("{}/y_ref.{}.txt".format(data_dir, file_split), "r")
      y_ref_lines = [line for line in y_ref_file]

    preprocessed_examples = []

    for i in range(len(x_type_lines)):
        types = x_type_lines[i].split()
        values = x_value_lines[i].split()
        src_list = []
        assert(len(types) == len(values))
        for j in range(len(types)):
            src_list.append(types[j])
            src_list.append(values[j])

        if include_refs:
          src_str = y_ref_lines[i].strip() + " [SEP] " + " ".join(src_list)
        else:
          src_str = " ".join(src_list)
        
        preprocessed_example = {}
        preprocessed_example["source"] = src_str.strip()
        preprocessed_example["target"] = y_aux_lines[i].strip()
        preprocessed_examples.append(preprocessed_example)

    return preprocessed_examples


def main(_):
  output_dir = FLAGS.output_dir
  split = FLAGS.split
  preprocessed_examples = preprocess_split(FLAGS.input_dir, split, include_refs=FLAGS.include_refs)
  print(f"Writing data to: {Path(output_dir).joinpath(split + '.json')}")
  with open(Path(output_dir).joinpath(split + ".json"), "w+") as f:
    for example in preprocessed_examples:
      f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
  app.run(main)