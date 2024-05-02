# -*- coding: utf-8 -*-
"""
@author: Sebastian Riedel <sriedel@suse.com>
"""
import argparse
import glob
import json
import os
import random
import shutil

instruction = """
Analyze the code or documentation snippet enclosed in "[CODE]" and "[/CODE]" tokens to determine if it contains legal
text that was written with the intention of describing how the code should be used. Answer only with "yes" or "no".
""".replace(
    "\n", " "
)


def append_snippet(f, snippet, is_legal_text):
    f.write(json.dumps({"snippet": snippet, "is_legal_text": is_legal_text}) + "\n")


def get_args():
    parser = argparse.ArgumentParser(
        "Convert LegalDB training data into various formats"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["alpaca", "cavil", "datasets"],
        default="datasets",
        help="output format (default: datasets)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="legaldb-ml-data",
        help="path to input folder",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="number of samples to use",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="legaldb-ml-data.jsonl",
        help="path to output file or folder",
    )
    return parser.parse_args()


def get_files(input, type, limit):
    files = glob.glob(os.path.join(input, type, "*.txt"))
    random.shuffle(files)
    if limit != None:
        files = files[:limit]
    return sorted(files)


def load_dump(fn):
    f = open(fn)
    try:
        return f.read()
    except UnicodeDecodeError:
        pass
    f.close()
    f = open(fn, encoding="iso-8859-15")
    return f.read()


def main():
    args = get_args()

    input = args.input
    output = args.output
    format = args.format
    limit = args.limit

    if format == "alpaca":
        output_alpaca(input, output, limit)
    elif format == "cavil":
        output_cavil(input, output, limit)
    elif format == "datasets":
        output_datasets(input, output, limit)


def output_alpaca(input, output, limit):
    records = []
    for fn in get_files(input, "bad", limit):
        snippet = load_dump(fn)[:2048]
        records.append(
            {
                "instruction": instruction,
                "input": f"[CODE]{snippet}[/CODE]",
                "output": "no",
            }
        )

    for fn in get_files(input, "good", limit):
        snippet = load_dump(fn)[:2048]
        records.append(
            {
                "instruction": instruction,
                "input": f"[CODE]{snippet}[/CODE]",
                "output": "yes",
            }
        )
    random.shuffle(records)
    with open(output, "a") as f:
        f.write(json.dumps(records))


def output_cavil(input, output, limit):
    good = os.path.join(output, "good")
    if not os.path.isdir(good):
        os.makedirs(good)
    bad = os.path.join(output, "bad")
    if not os.path.isdir(bad):
        os.makedirs(bad)

    for fn in get_files(input, "bad", limit):
        shutil.copy(fn, bad)

    for fn in get_files(input, "good", limit):
        shutil.copy(fn, good)


def output_datasets(input, output, limit):
    with open(output, "a") as f:
        for fn in get_files(input, "bad", limit):
            append_snippet(f, load_dump(fn), False)

        for fn in get_files(input, "good", limit):
            append_snippet(f, load_dump(fn), True)


main()
