import json
import os

from pathlib import Path
from absl import app
from absl import flags

flags.DEFINE_string("input_dir", f"{os.getenv('CTRL_D2T_ROOT')}/DTG-SI/e2e_data", "Input directory.")
flags.DEFINE_string("output_dir", f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/e2e", "Output directory.")
flags.DEFINE_string("split", None, "Dataset split: 'train', 'validation', or 'test'")
flags.DEFINE_bool("include_refs", False, "Whether to include style references in source (for controlled generation)")
flags.DEFINE_bool("preprocess_split", False, "Run ordinary preprocessing method")
flags.DEFINE_bool("preprocess_dtg_si_train", False, "Run DTG-SI pre-trained comparison preprocessing method")


FLAGS = flags.FLAGS


def read_from_file(fname, strip=True):
    with open(fname, "r") as f:
        if strip:
            return [line.strip() for line in f]
        else:
            return [line for line in f]


def get_src_str(type_line, value_line):
    types = type_line.split()
    values = value_line.split()
    src_list = []
    assert(len(types) == len(values))
    for j in range(len(types)):
        src_list.append(types[j])
        src_list.append(values[j])
    src_str = " ".join(src_list)
    return src_str


def preprocess_split(input_dir, split, include_refs=False):
    split_file = "valid" if split == "validation" else split
    split_dir = "val" if split == "validation" else split
    data_dir = os.path.join(input_dir, split_dir)

    x_type_lines = read_from_file(f"{data_dir}/x_type.{split_file}.txt")
    x_value_lines = read_from_file(f"{data_dir}/x_value.{split_file}.txt")
    y_aux_lines = read_from_file(f"{data_dir}/y_aux.{split_file}.txt")
    if include_refs:
        y_ref_lines = read_from_file(f"{data_dir}/y_ref.{split_file}.txt")

    preprocessed_examples = []
    for i in range(len(x_type_lines)):
        src_str = get_src_str(x_type_lines[i], x_value_lines[i])

        if include_refs:
            src_str = y_ref_lines[i] + " [SEP] " + src_str
       
        preprocessed_example = {}
        preprocessed_example["source"] = src_str
        preprocessed_example["target"] = y_aux_lines[i]
        preprocessed_examples.append(preprocessed_example)

    return preprocessed_examples


def preprocess_dtg_si_train(input_dir):
    train_dir = os.path.join(input_dir, "train")
    x_type_lines = read_from_file(f"{train_dir}/x_type.train.txt")
    x_value_lines = read_from_file(f"{train_dir}/x_value.train.txt")
    x_ref_type_lines = read_from_file(f"{train_dir}/x_ref_type.train.txt")
    x_ref_value_lines = read_from_file(f"{train_dir}/x_ref_value.train.txt")
    y_aux_lines = read_from_file(f"{train_dir}/y_aux.train.txt")
    y_ref_lines = read_from_file(f"{train_dir}/y_ref.train.txt")
    
    preprocessed_examples = []
    for i in range(len(x_type_lines)):
        x_src_str = get_src_str(x_type_lines[i], x_value_lines[i])
        x_ref_src_str = get_src_str(x_ref_type_lines[i], x_ref_value_lines[i])
        src_str = x_src_str + " [SEP] " + x_ref_src_str
        tgt_str = y_aux_lines[i] + " [SEP] " + y_ref_lines[i]

        preprocessed_example = {}
        preprocessed_example["source"] = src_str
        preprocessed_example["target"] = tgt_str
        preprocessed_examples.append(preprocessed_example)

    return preprocessed_examples



def main(_):
    output_dir = FLAGS.output_dir
    if FLAGS.preprocess_split:
        split = FLAGS.split
        preprocessed_examples = preprocess_split(FLAGS.input_dir, split, include_refs=FLAGS.include_refs)
    elif FLAGS.preprocess_dtg_si_train:
        split = "train"
        preprocessed_examples = preprocess_dtg_si_train(FLAGS.input_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir).joinpath(split + ".json")
    print(f"Writing data to: {output_file}")
    with open(output_file, "w+") as f:
        for example in preprocessed_examples:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
  app.run(main)