# -*- coding: utf-8 -*-
"""
@author: Sebastian Riedel <sriedel@suse.com>
"""
import argparse
from datasets import load_dataset
import os
import shutil

def get_args():
    parser = argparse.ArgumentParser(
        "Revert dataset back to LegalDB training data format"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="legaldb-ml-data.jsonl",
        help="path to input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="legaldb-ml-data-reverted",
        help="path to output folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    input = args.input
    output = args.output

    dataset = load_dataset("json", data_files=args.input)

    good = os.path.join(output, "good")
    if not os.path.isdir(good):
        os.makedirs(good)
    bad = os.path.join(output, "bad")
    if not os.path.isdir(bad):
        os.makedirs(bad)

    for data_point in dataset["train"]:
        is_legal_text = data_point["is_legal_text"]
        snippet = data_point["snippet"]
        unique_id = data_point["unique_id"]
        if is_legal_text:
            with open(os.path.join(good, f"{unique_id}.txt"), "w") as f:
                f.write(snippet)
        else:
            with open(os.path.join(bad, f"{unique_id}.txt"), "w") as f:
                f.write(snippet)
