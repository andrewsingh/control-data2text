#!/usr/bin/env python

import datasets
import argparse
import os
import sys
sys.path.append("retrievers")

from totto_retriever import TottoRetriever
from e2e_retriever import E2ERetriever


datasets.logging.set_verbosity_info()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--retrieval_k", type=int, default=5)
    parser.add_argument("--max_edit_dist", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--index_path", type=str, default=None)
    parser.add_argument("--create_train", action="store_true")
    parser.add_argument("--create_eval", action="store_true")
    parser.add_argument("--run_full", action="store_true")
    
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    
    if args.dataset == "e2e":
        data_dir = args.data_dir or f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/e2e"
        retriever = E2ERetriever(data_dir, args.split, args.index_path)
    elif args.dataset == "totto":
        data_dir = args.data_dir or f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/totto"
        retriever = TottoRetriever(data_dir, args.split, args.index_path)

    if args.create_train:
        out_dir = args.out_dir or f"{args.retriever}_k{args.retrieval_k}"
        full_out_dir = f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/{out_dir}"
        retriever.create_train_set(full_out_dir, retrieval_k=args.retrieval_k, max_edit_dist=args.max_edit_dist)
    elif args.create_eval:
        out_dir = args.out_dir or f"{args.retriever}_{args.split}"
        full_out_dir = f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/{out_dir}"
        retriever.create_eval_set(full_out_dir)


if __name__ == '__main__':
    main()