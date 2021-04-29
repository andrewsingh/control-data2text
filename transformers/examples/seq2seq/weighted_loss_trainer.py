import torch
import datasets
import re
import numpy as np
from transformers.trainer_seq2seq import Seq2SeqTrainer
from torch.nn import CrossEntropyLoss
import pdb


class WeightedLossTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        if "edit_dists" in inputs.keys():            
            edit_dists = inputs.pop("edit_dists")
            outputs = model(**inputs)
            logits = outputs["logits"]
            logits = torch.transpose(logits, 1, 2)
            labels = inputs.pop("labels")
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss_vec = loss_fct(logits, labels)
            loss_vec = torch.mean(loss_vec, 1)
            raw_weights = 1 / edit_dists
            weights = raw_weights / raw_weights.sum()
            loss = torch.dot(weights, loss_vec)
            # print(f"\nweighted loss: {loss}\nunweighted loss: {outputs['loss']}\n")
            return (loss, outputs) if return_outputs else loss
        else:
            super().compute_loss(model, inputs, return_outputs=return_outputs)

