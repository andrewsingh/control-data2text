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
    parser.add_argument("--retriever", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--proto_name", type=str, default=None)
    parser.add_argument("--retrieval_k", type=int, default=5)
    parser.add_argument("--max_edit_dist", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--create_train", action="store_true")
    parser.add_argument("--run_full", action="store_true")
    
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    if args.proto_name:
        proto_data_dir = f"{os.getenv('CTRL_D2T_ROOT')}/transformers/examples/seq2seq/test_data/{args.proto_name}"
    else:
        proto_data_dir = None
    
    if args.retriever == "e2e":
        retriever = E2ERetriever(args.split, args.index, proto_data_dir=proto_data_dir)
    elif args.retriever == "totto":
        retriever = TottoRetriever(args.split, args.index, proto_data_dir=proto_data_dir)

    if args.create_train:
        retriever.create_train_set(retrieval_k=args.retrieval_k, max_edit_dist=args.max_edit_dist)




if __name__ == '__main__':
    main()