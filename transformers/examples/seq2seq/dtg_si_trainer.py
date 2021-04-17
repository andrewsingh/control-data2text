import torch
import datasets
import re
import numpy as np
from transformers.trainer_seq2seq import Seq2SeqTrainer


class DtgSiTrainer(Seq2SeqTrainer):

    # def __init__(self, config=None, data_args=None, *args, **kwargs):
    #     super().__init__(config, data_args, *args, **kwargs)

    #     self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="none")
    #     print(f"loss_lambda: {self.args.loss_lambda}\n")
        
    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.tokenizer.pad_token_id
        return shifted_input_ids

    def compute_individual_loss(self, model, source, target):
        inputs = self.tokenizer.prepare_seq2seq_batch(
            [source], 
            tgt_texts=[target],
            max_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
            return_tensors="pt",
        ).to("cuda:0")

        inputs["decoder_input_ids"] = self._shift_right_t5(inputs["labels"])
        # print("NEW INPUTS: {}\n".format(inputs))
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(loss)
        return loss

    def compute_loss(self, model, inputs):
        # print(f"OLD INPUTS: {inputs}\n")
        # print(f"Decoded inputs: {self.tokenizer.decode(inputs['input_ids'][0][:-1])}\n")
        [x, x_e] = self.tokenizer.decode(inputs["input_ids"][0][:-1]).split("[SEP]")
        [y_x, y_e] = self.tokenizer.decode(inputs["labels"][0][:-1]).split("[SEP]")
        content_src = y_e.strip() + " [SEP] " + x.strip()
        content_tgt = y_x.strip()
        style_src = y_e.strip() + " [SEP] " + x_e.strip()
        style_tgt = y_e.strip()

        # print(f"content source: {content_src}\ncontent target: {content_tgt}\nstyle source: {style_src}\nstyle target: {style_tgt}\n")
        content_loss = self.compute_individual_loss(model, content_src, content_tgt)
        style_loss = self.compute_individual_loss(model, style_src, style_tgt)
        # print(f"loss_lambda: {self.args.loss_lambda}")
        joint_loss = self.args.loss_lambda * content_loss + (1 - self.args.loss_lambda) * style_loss
        # print(joint_loss)
        return joint_loss




    
