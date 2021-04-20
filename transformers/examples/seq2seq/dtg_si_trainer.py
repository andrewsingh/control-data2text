import torch
import datasets
import re
import numpy as np
from transformers.trainer_seq2seq import Seq2SeqTrainer


class DtgSiTrainer(Seq2SeqTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(f"Training mode: {model.training}")
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

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.tokenizer.pad_token_id
        return shifted_input_ids

    



    # def split_array(self, arr, x):
    #     split_idx = np.where(arr == x)[0][0]
    #     return arr[:split_idx], arr[(split_idx + 1):]

    # def compute_loss(self, model, inputs):
    #     # print(f"OLD INPUTS: {inputs}\n")
    #     # print(f"Decoded inputs: {self.tokenizer.decode(inputs['input_ids'][0][:-1])}\n")
        
    #     sep_tok = model.config.vocab_size - 1
    #     batch_size = inputs["input_ids"].shape[0]
    #     content_toks = np.zeros((batch_size, ))
    #     for i in range(batch_size):
    #         src_toks = inputs["input_ids"][i].numpy()
    #         x_arr, x_e_arr = self.split_array(src_toks[:-1], sep_tok)
    #         tgt_toks = inputs["labels"][i].numpy()
    #         y_x_arr, y_e_arr = self.split_array(tgt_toks[:-1], sep_tok)
    #         content_src_toks = np.concatenate(y_e_arr, x_arr, [1])
    #         content_tgt_toks = np.concatenate(y_x_arr, [1])
    #         style_src_toks = np.concatenate(y_e_arr, x_e_arr, [1])
    #         style_tgt_toks = np.concatenate(y_e_arr, [1])

            


    #     [x, x_e] = self.tokenizer.decode(inputs["input_ids"][0][:-1]).split("[SEP]")
    #     [y_x, y_e] = self.tokenizer.decode(inputs["labels"][0][:-1]).split("[SEP]")
    #     content_src = y_e.strip() + " [SEP] " + x.strip()
    #     content_tgt = y_x.strip()
    #     style_src = y_e.strip() + " [SEP] " + x_e.strip()
    #     style_tgt = y_e.strip()

    #     # print(f"content source: {content_src}\ncontent target: {content_tgt}\nstyle source: {style_src}\nstyle target: {style_tgt}\n")
    #     content_loss = self.compute_individual_loss(model, content_src, content_tgt)
    #     style_loss = self.compute_individual_loss(model, style_src, style_tgt)
    #     # print(f"loss_lambda: {self.args.loss_lambda}")
    #     joint_loss = self.args.loss_lambda * content_loss + (1 - self.args.loss_lambda) * style_loss
    #     # print(joint_loss)
    #     return joint_loss




    
