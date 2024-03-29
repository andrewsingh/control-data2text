"""
This is the Data Loading Pipeline for Sentence Classifier Task from
https://github.com/google-research/bert/blob/master/run_classifier.py
"""
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

import os
import csv
import collections
import sys

sys.path.append(os.path.dirname(__file__))
import tokenization
import tensorflow as tf

from texar.data import make_vocab
sys.path.append('..')
sys.path.append('..')
#import prepare_data
num_tokens = 0


class InputExample():
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.
		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence.
				For single sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second
				sequence. Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
				specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class InputFeatures():
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_test_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for prediction."""
		raise NotImplementedError()

	def get_accu_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for prediction."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with tf.gfile.Open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
		return lines



class SSTProcessor(DataProcessor):
	"""Processor for the MRPC data set (GLUE version)."""

	def get_test_examples(self, data_dir, expr_name, step, which_score):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "{}.step{}.{}.tsv".format(expr_name, step, which_score))), "test")

	def get_accu_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
			self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")


	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		if set_type == 'test':
			for (i, line) in enumerate(lines):
				guid = "%s-%s" % (set_type, i)
				text_a = tokenization.convert_to_unicode(line[0])
				# Single sentence classification, text_b doesn't exist
				text_b = None
				label = '0'  # arbitrary set as 0
				examples.append(InputExample(guid=guid, text_a=text_a,
											 text_b=text_b, label=label))

		if set_type == 'train' or set_type == 'dev': #or 'test':
			for (i, line) in enumerate(lines):
				if i == 0:
					continue
				guid = "%s-%s" % (set_type, i)
				text_a = tokenization.convert_to_unicode(line[0])
				# Single sentence classification, text_b doesn't exist
				text_b = None
				label = tokenization.convert_to_unicode(line[1])
				examples.append(InputExample(guid=guid, text_a=text_a,
											 text_b=text_b, label=label))
		return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
						   tokenizer):
	"""Converts a single `InputExample` into a single `InputFeatures`."""
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	tokens_a = tokenizer.tokenize(example.text_a)
	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)

	if tokens_b:
		# Modifies `tokens_a` and `tokens_b` in place so that the total
		# length is less than the specified length.
		# Account for [CLS], [SEP], [SEP] with "- 3"
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	else:
		# Account for [CLS] and [SEP] with "- 2"
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	global all_vocab
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)

	tokens.append("[SEP]")
	segment_ids.append(0)

	if tokens_b:
		for token in tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		tokens.append("[SEP]")
		segment_ids.append(1)
	global num_tokens
	num_tokens += 1
	# if num_tokens % 3000 == 0:
	# 	print('===========No.{} tokens are : {}'.format(num_tokens, tokens))
	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length

	label_id = label_map[example.label]

	# here we disable the verbose printing of the data
	if ex_index < 0:
		tf.logging.info("*** Example ***")
		tf.logging.info("guid: %s" % (example.guid))
		tf.logging.info("tokens: %s" % " ".join(
			[tokenization.printable_text(x) for x in tokens]))
		tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
		tf.logging.info("input_ids length: %d" % len(input_ids))
		tf.logging.info("input_mask: %s" % \
						" ".join([str(x) for x in input_mask]))
		tf.logging.info("segment_ids: %s" % \
						" ".join([str(x) for x in segment_ids]))
		tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

	feature = InputFeatures(input_ids=input_ids,
							input_mask=input_mask,
							segment_ids=segment_ids,
							label_id=label_id)
	return feature


def file_based_convert_examples_to_features(
		examples, label_list, max_seq_length, tokenizer, output_file):
	"""Convert a set of `InputExample`s to a TFRecord file."""
	global all_vocab
	writer = tf.python_io.TFRecordWriter(output_file)

	for (ex_index, example) in enumerate(examples):

		feature = convert_single_example(ex_index, example, label_list,
										 max_seq_length, tokenizer)

		def create_int_feature(values):
			return tf.train.Feature(
				int64_list=tf.train.Int64List(value=list(values)))

		features = collections.OrderedDict()
		features["input_ids"] = create_int_feature(feature.input_ids)
		features["input_mask"] = create_int_feature(feature.input_mask)
		features["segment_ids"] = create_int_feature(feature.segment_ids)
		features["label_ids"] = create_int_feature([feature.label_id])

		tf_example = tf.train.Example(
			features=tf.train.Features(feature=features))
		writer.write(tf_example.SerializeToString())

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal
	# percent of tokens from each, since if one sequence is very short then
	# each token that's truncated likely contains more information than a
	# longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def prepare_TFRecord_data(processor, tokenizer,
						  data_dir, max_seq_length, output_dir, expr_name, step):
	"""
	Args:
		processor: Data Preprocessor, which must have get_lables,
			get_train/dev/test/examples methods defined.
		tokenizer: The Sentence Tokenizer. Generally should be
			SentencePiece Model.
		data_dir: The input data directory.
		max_seq_length: Max sequence length.
		batch_size: mini-batch size.
		model: `train`, `eval` or `test`.
		output_dir: The directory to save the TFRecords in.
	"""
	label_list = processor.get_labels()
	# accu_examples = processor.get_accu_examples(data_dir)
	# accu_file = os.path.join(output_dir, "record/test.tf_record")
	# file_based_convert_examples_to_features(
	# 	accu_examples, label_list,
	# 	max_seq_length, tokenizer, accu_file)

	test_examples = processor.get_test_examples(data_dir, expr_name, step, 1)
	test_file = os.path.join(output_dir, "record/{}.1.tf_record".format(expr_name))
	file_based_convert_examples_to_features(
		test_examples, label_list,
		max_seq_length, tokenizer, test_file)

	test_examples = processor.get_test_examples(data_dir, expr_name, step, 2)
	test_file = os.path.join(output_dir, "record/{}.2.tf_record".format(expr_name))
	file_based_convert_examples_to_features(
		test_examples, label_list,
		max_seq_length, tokenizer, test_file)

	# train_examples = processor.get_train_examples(data_dir)
	# train_file = os.path.join(output_dir, "record/train.tf_record")
	# file_based_convert_examples_to_features(
	# 	train_examples, label_list, max_seq_length,
	# 	tokenizer, train_file)
	#
	# eval_examples = processor.get_dev_examples(data_dir)
	# eval_file = os.path.join(output_dir, "record/eval.tf_record")
	# file_based_convert_examples_to_features(
	# 	eval_examples, label_list,
	# 	max_seq_length, tokenizer, eval_file)

