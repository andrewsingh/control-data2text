import torch
from seq2seq_trainer import Seq2SeqTrainer
import datasets
import re
import numpy as np


class PrototypeRetrievalTrainer(Seq2SeqTrainer):
    def __init__(self, config=None, data_args=None, *args, **kwargs):
        super().__init__(config, data_args, *args, **kwargs)

        self.index = datasets.load_from_disk("indexes/{}".format("e2e_v1")) # TODO: make dataset name dynamic
        self.args.per_device_train_batch_size = 1
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="none")
        # TODO: make these two hyperparams script arguments
        self.k = 1
        self.loss_lambda = 0.1
        print("Prototype Retrieval Hyperparams:\nk={}\nlambda={}".format(self.k, self.loss_lambda))
        
    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.config.pad_token_id
        return shifted_input_ids

    def compute_loss(self, model, inputs):
        # print("OLD INPUTS: {}\n".format(inputs))
        og_src_text = self.tokenizer.decode(inputs["input_ids"][0][:-1])
        og_tgt_text = self.tokenizer.decode(inputs["labels"][0][:-1])
        # src_texts = [og_tgt_text + " [SEP] " + og_src_text]
        # print("OG source: {}".format(og_src_text))
        indices = np.random.choice(len(self.index), self.k, replace=False)
        proto_src_text = self.index[indices]["source"][0]
        proto_tgt_text = self.index[indices]["target"][0]
        src_texts = [proto_tgt_text + " [SEP] " + og_src_text, proto_tgt_text + " [SEP] " + proto_src_text]
        tgt_texts = [og_tgt_text, proto_tgt_text]
        # for proto_text in self.index[indices]["target"]:
        #     src_texts.append(proto_text + " [SEP] " + og_src_text)

        new_inputs = self.tokenizer.prepare_seq2seq_batch(
            src_texts, 
            tgt_texts=tgt_texts,
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            return_tensors="pt",
        ).to("cuda:0")
        # new_inputs["labels"] = inputs["labels"].repeat(self.k + 1, 1)
        new_inputs["decoder_input_ids"] = self._shift_right_t5(new_inputs["labels"])
        # print("NEW INPUTS: {}\n".format(new_inputs))
        
        new_labels = new_inputs.pop("labels")
        logits = model(**new_inputs, use_cache=False)[0]
        # print("logits shape: {}".format(logits.shape))
        logits = torch.transpose(logits, 1, 2)
        loss = self.loss_fn(logits, new_labels)
        # print("loss shape: {}".format(loss.shape))
        loss = torch.mean(loss, 1)
        content_loss = loss[0]
        style_loss = loss[1]
        # content_loss = torch.mean(loss[1:])
        joint_loss = self.loss_lambda * content_loss + (1 - self.loss_lambda) * style_loss
        return joint_loss


    
