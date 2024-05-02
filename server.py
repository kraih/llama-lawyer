# -*- coding: utf-8 -*-
"""
@author: Sebastian Riedel <sriedel@suse.com>
"""
import json
import argparse
from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# We were having trouble with Apple M1, better to use CPU for now
device = "cpu"
torch_dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.bfloat16

system_prompt = """
You are a helpful lawyer. Analyze the code or documentation snippet enclosed
in "[CODE]" and "[/CODE]" tokens to determine if it contains legal text that
was written with the intention of describing how the code should be used.
Answer only with "yes" or "no".

User:
[CODE]// SPDX-License-Identifier: MIT[/CODE]
Assistant:
yes

User:
[CODE]// Released under BSD-2-clause license[/CODE]
Assistant:
yes

User:
[CODE]# Released under BSD-3-clause license[/CODE]
Assistant:
yes

User:
[CODE]Hello World[/CODE]
Assistant:
no

User:
[CODE]Foo Bar Baz[/CODE]
Assistant:
no

User:
[CODE]GPL License Version 2.0[/CODE]
Assistant:
yes

User:
[CODE]// Copyright 2024
//Licensed as BSD-3-clause
[/CODE]
Assistant:
yes

User:
[CODE]my $foo = 23;[/CODE]
Assistant:
no

User:
[CODE]
# SPDX-License-Identifier: MIT
my $foo = 23;
[/CODE]
Assistant:
yes

User:
[CODE]if (license === true) {[/CODE]
Assistant:
no

Analyze the following code or documentation snippet. Answer only with "yes" or "no".
"""


def get_args():
    parser = argparse.ArgumentParser("LegalDB text classification server")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="../Meta-Llama-8B-Instruct-Cavil-hf",
        help="path to model",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="TCP port to listen on",
    )
    return parser.parse_args()


app = Flask(__name__)
args = get_args()
model = AutoModelForCausalLM.from_pretrained(
    args.model, device_map=device, torch_dtype=torch_dtype
)
tokenizer = AutoTokenizer.from_pretrained(args.model)
eos_token_id = tokenizer.encode("\n")


def get_prompt(snippet):
    return f"{system_prompt}\nUser:\n[CODE]{snippet}[/CODE]\nAssistant:\n"


def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(
        inputs=inputs,
        num_return_sequences=1,
        max_new_tokens=1,
        top_p=None,
        temperature=None,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = inputs.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    token = generated_tokens[0][0]
    score = transition_scores[0][0].cpu()

    return {
        "token": f"{token}",
        "text": tokenizer.decode(token),
        "score": f"{score.numpy():.4f}",
        "confidence": f"{np.exp(score.numpy()):.2%}",
    }


@app.route("/", methods=["POST"])
def classify():
    snippet = request.get_data().decode("utf-8")[:2048]
    response = get_response(get_prompt(snippet))
    print(str(response))

    # We don't expect anything else than "yes" or "no", but you never know
    result = not response["text"].lower() == "no"
    confidence = response["confidence"].strip("%")
    data = {"license": result, "confidence": confidence}

    return jsonify(data)


app.run(host="0.0.0.0", port=args.port)
