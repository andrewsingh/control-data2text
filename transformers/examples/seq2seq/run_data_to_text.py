#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import subprocess
import re
import csv
import math
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import nltk  # Here to have a nice missing dependency error message early on
from nltk import word_tokenize
import numpy as np
import random
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process, PREFIX_CHECKPOINT_DIR
from transformers.utils import check_min_version
from sacremoses import MosesDetokenizer
from dtg_si_trainer import DtgSiTrainer


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"setting model.config to task specific params for {task}:\n {pars}")
        logger.info("note: command line args may override some of these")
        model.config.update(pars)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default="source",
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default="target",
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    task: Optional[str] = field(
        default=None, metadata={"help": "Task name for configuration purposes."}
    )
    loss_lambda: Optional[float] = field(
        default=None, metadata={"help": "Balancing weight for DTG-SI joint training objective."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    root_dir = os.getenv('CTRL_D2T_ROOT')

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    default_params = {
        "prefix": "", 
        "max_length": data_args.val_max_target_length,
        "num_beams": data_args.num_beams, 
    }
    config.task_specific_params["e2e"] = default_params
    config.task_specific_params["e2e_dtg_si"] = default_params
    config.task_specific_params["totto"] = default_params

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    use_task_specific_params(model, data_args.task)

    print(f"Model config max_length: {model.config.max_length}")
    print(f"Model config num_beams: {model.config.num_beams}")

    # Add dataset-specific special tokens to tokenizer
    special_tokens = ["[SEP]"]
    if data_args.task == "totto":
        special_tokens += ["<page_title>", "</page_title>", "<section_title>", "</section_title>", "<table>", "</table>", "<cell>", "</cell>", "<row_header>", "</row_header>", "<col_header>", "</col_header>"]

    if len(special_tokens) > 0:
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        print("Adding {} special tokens: {}".format(len(special_tokens_dict["additional_special_tokens"]), special_tokens_dict["additional_special_tokens"]))
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        print("New vocab size: {}".format(model.config.vocab_size))

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = None
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def postprocess_totto_preds(preds):
        return [pred.strip() for pred in preds]

    def postprocess_e2e_preds(preds):
        print("Post-processing E2E predictions")
        return [" ".join(word_tokenize(pred.strip())) for pred in preds]


    def compute_default_metrics(eval_preds):
        metric = load_metric("bleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    def compute_totto_metrics(eval_preds):
        preds, _ = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds = postprocess_totto_preds(decoded_preds)

        totto_dir = f"{root_dir}/google-language/language/totto"
        temp_dir = totto_dir + "/temp"
        os.makedirs(temp_dir, exist_ok=True)
        preds_file_path = os.path.join(temp_dir, f"totto_preds_{random.randint(0, 1e9)}.txt") 
        while os.path.exists(preds_file_path):
            preds_file_path = os.path.join(temp_dir, f"totto_preds_{random.randint(0, 1e9)}.txt") 

        preds_file = open(preds_file_path, "w+")
        preds_file.write('\n'.join(decoded_preds))
        preds_file.close()
        results = subprocess.run(["bash", totto_dir + "/totto_eval.sh", "--prediction_path", preds_file_path, "--target_path", totto_dir + "/totto_data/totto_dev_data.jsonl", "--output_dir", temp_dir], stdout=subprocess.PIPE)

        def get_bleu():
            return float(re.search("BLEU\+case\.mixed\+numrefs\.3\+smooth\.exp\+tok\.13a\+version\.1\.5\.1 = ([0-9]+.[0-9]+)", str(results.stdout)).group(1))
            
        def get_parent_metric(metric):
            return float(re.search("{} = ([0-9]+.[0-9]+)".format(metric), str(results.stdout)).group(1))

        parent_metrics = ["Precision", "Recall", "F-score"]
        metric_dict = {"BLEU": get_bleu()}
        for metric in parent_metrics:
            metric_dict[metric] = get_parent_metric(metric)
        os.remove(preds_file_path)
        return metric_dict

    
    def compute_e2e_metrics(eval_preds):
        print("Computing E2E evaluation metrics")
        gpu = "0"
        device = f"cuda:{gpu}"
        if trainer.state.global_step == 0:
            output_dir = training_args.output_dir
        else:
            output_dir = os.path.join(training_args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}")
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to directory: {output_dir}")
        
        # last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # if last_checkpoint:
        #     output_dir = last_checkpoint
        # else:
        #     output_dir = training_args.output_dir

        md = MosesDetokenizer(lang='en')
        e2e_lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        e2e_lm = GPT2LMHeadModel.from_pretrained(f"{root_dir}/transformers/examples/language-modeling/exp/e2e_targets/gpt2-02/checkpoint-9464")
        e2e_lm.to(device)

        def process_preds_for_lm(preds):
            processed_preds = []
            for pred in tqdm(preds):
                processed_preds.append(md.detokenize(pred.replace("_", " ").split()))
            return processed_preds
        
        def compute_accuracy(path, correct_label):
            num_correct = 0
            with open(path, 'r') as f:
                data = csv.reader(f, delimiter='\t')
                data = list(data)
                for (_, pred) in data[1:]:
                    if int(pred) == correct_label:
                        num_correct += 1
                return num_correct / len(data)

        def compute_perplexity(preds):
            ppls = []
            for pred in tqdm(preds):
                inputs = e2e_lm_tokenizer(pred, return_tensors='pt').to(device)
                outputs = e2e_lm(**inputs, labels=inputs['input_ids'])
                ppls.append(math.exp(outputs.loss))
            return round((sum(ppls) / len(ppls)), 4)

        preds, _ = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = postprocess_e2e_preds(decoded_preds)

        preds_file = f"{output_dir}/validation_preds.txt"
        with open(preds_file, "w+") as f:
            f.writelines([pred + "\n" for pred in decoded_preds])

        prepare_preds_content_path = f"{root_dir}/DTG-SI/prepare_e2e_preds_content.py"
        predict_e2e_content_path = f"{root_dir}/transformers/examples/text-classification/predict_e2e_content.sh"

        print("Preparing predictions for content classifier")
        subprocess.run(["python", prepare_preds_content_path, "--input_path", preds_file, "--output_path", output_dir])

        print("Running content classifier")
        subprocess.run(["bash", predict_e2e_content_path, gpu, output_dir, f"{output_dir}/validation_preds_content.1.csv"])
        subprocess.run(["bash", predict_e2e_content_path, gpu, output_dir, f"{output_dir}/validation_preds_content.2.csv"])

        content_results_1 = f"{output_dir}/test_results_validation_preds_content.1.csv.txt"
        content_results_2 = f"{output_dir}/test_results_validation_preds_content.2.csv.txt"

        inc_new = compute_accuracy(content_results_1, 1)
        exc_old = compute_accuracy(content_results_2, 0)

        os.remove(content_results_1)
        os.remove(content_results_2)
        
        print("Computing m-BLEU")
        eval_style_path = f"{root_dir}/DTG-SI/evaluate_e2e_style.py"
        eval_style_results = subprocess.run(["python", eval_style_path, "--preds_path", preds_file], stdout=subprocess.PIPE)
        m_bleu = float(re.search("BLEU: ([0-9]+.[0-9]+)", str(eval_style_results.stdout)).group(1))

        print("Running language model")
        try:
            ppl = compute_perplexity(process_preds_for_lm(decoded_preds))
        except:
            ppl = -1

        metric_dict = {
            "Inc-New": inc_new,
            "Exc-Old": exc_old,
            "m-BLEU": m_bleu,
            "Perplexity": ppl
        }
        return metric_dict
        

    if data_args.task == "totto":
        metrics_fn = compute_totto_metrics
    elif data_args.task in ["e2e", "e2e_dtg_si"]:
        metrics_fn = compute_e2e_metrics
    else:
        metrics_fn = compute_default_metrics
    

    # Initialize our Trainer
    if data_args.task == "e2e_dtg_si":
        training_args.loss_lambda = data_args.loss_lambda
        training_args.max_source_length = data_args.max_source_length
        training_args.max_target_length = data_args.max_target_length
        training_args.per_device_train_batch_size = 1
        trainer = DtgSiTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=metrics_fn if training_args.predict_with_generate else None,
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=metrics_fn if training_args.predict_with_generate else None,
        )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = test_results.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
