from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import PaddingStrategy
from transformers import DataCollatorForSeq2Seq
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import pdb


@dataclass
class DataCollatorForWeightedLoss(DataCollatorForSeq2Seq):

    def __call__(self, features):
        if "edit_dists" in features[0].keys():
            features = features[0]
            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # prepare decoder_input_ids
            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
                features["decoder_input_ids"] = decoder_input_ids

            return features
        else:
            return super().__call__(features)