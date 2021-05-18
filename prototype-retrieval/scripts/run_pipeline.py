#!/usr/bin/env python

import datasets
import argparse
import os
import sys
sys.path.append("../retrievers")
sys.path.append("retrievers")

from totto_retriever import TottoRetriever
from e2e_retriever import E2ERetriever


datasets.logging.set_verbosity_info()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--map_name", type=str, default="edit_dist_map")
    parser.add_argument("--col_name", type=str, default="split_masked_target")
    parser.add_argument("--retrieval_path", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--create_train", action="store_true")
    parser.add_argument("--create_eval", action="store_true")
    parser.add_argument("--add_edit_dist", action="store_true")
    parser.add_argument("--run_full", action="store_true")
    
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    
    if args.dataset == "e2e":
        retriever = E2ERetriever(args.dataset_path)
    elif args.dataset == "totto":
        retriever = TottoRetriever(args.dataset_path)

    if args.add_edit_dist:
        retriever.add_edit_dist_maps(retrieval_path=args.retrieval_path, map_name=args.map_name, col_name=args.col_name, num_proc=args.num_proc)

if __name__ == '__main__':
    main()