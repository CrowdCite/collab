import argparse
from collections import *
from functools import *
from itertools import *
from glob import glob
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import random
from typing import *

import torch
import torch.multiprocessing as torch_mp
import torch.nn.functional as F
from transformers import *

import boto3
from tqdm import tqdm

chaini = chain.from_iterable


def reader(args, fns: List[str], tokenizer, queue: mp.Queue):
    keys = []
    text_pairs = []
    for fn in fns:
        lns = open(fn).readlines()
        for lineno, ln in enumerate(lns):
            js = json.loads(ln.strip())
            keys.extend(
                repeat([fn, len(lns), lineno, js["page"]],
                       len(js["candidates"])))
            text_pairs.extend([js["query"], cand] for cand in js["candidates"])
        print(f"read {fn}")
        while len(text_pairs) >= args.batch_size:
            _, cands = zip(*text_pairs[-args.batch_size:])
            queue.put([
                keys[-args.batch_size:], js["query"], cands,
                tokenizer.batch_encode_plus(text_pairs[-args.batch_size:],
                                            padding=True,
                                            return_tensors="pt")
            ])
            del keys[-args.batch_size]
            del text_pairs[-args.batch_size]


def estimator(pid, args, input_queue: mp.Queue, output_queue: mp.Queue):
    device = torch.device(f"cuda:{args.gpus[pid]}")
    lm = AutoModelForNextSentencePrediction.from_pretrained(
        args.lm_card).to(device)
    while True:
        try:
            keys, query, cands, inputs = input_queue.get()
        except TypeError:
            break
        with torch.no_grad():
            outputs = lm(**inputs.to(device))  # (B * N, 2)
        probs = outputs["logits"].softmax(1)[:, 0]  # (B * N,)
        for key, cand, prob in zip(keys, cands, probs):
            output_queue.put([key, query, cand, prob])


def upload(fn, jsl):
    jsonl = []
    for lineno in sorted(jsl):
        [page, query, *_], *_ = jsl[lineno]
        _, _, cands, probs = zip(*jsl[lineno])
        jsonl.append({
            "page": page,
            "query": query,
            "candidates": list(zip(cands, probs))
        })

    open(fn, 'w').write(json.dumps('\n'.join(jsonl)))


def writer(queue: mp.Queue):
    jsls = defaultdict(lambda: defaultdict(list))
    cand_cnts = defaultdict(int)
    while True:
        try:
            [fn, num_cands, lineno, page], query, cand, prob = queue.get()
        except TypeError:
            break
        jsls[fn][lineno].append([page, query, cand, prob])
        cand_cnts[fn] += 1
        if cand_cnts[fn] == num_cands:
            mp.Process(target=upload, args=[jsls[fn]]).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpus", type=int, nargs='+')
    parser.add_argument("--lm-card",
                        type=str,
                        default="allenai/scibert_scivocab_cased")
    parser.add_argument("--num-readers", type=int)
    args = parser.parse_args()

    fns = glob("inputs/queries.*.json")

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    num_files_per_worker = math.ceil(len(fns) / args.num_readers)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_card)
    readers = [
        mp.Process(target=reader,
                   args=[
                       args, fns[idx * num_files_per_worker:(idx + 1) *
                                 num_files_per_worker], tokenizer, input_queue
                   ]) for idx in range(args.num_readers)
    ]
    for proc in readers:
        proc.start()

    # torch_mp.spawn(estimator,
    #                args=[args, input_queue, output_queue],
    #                nprocs=len(args.gpus))
    estimator(0, args, input_queue, output_queue)

    writer_proc = mp.Process(target=writer, args=[output_queue])
    writer_proc.start()

    for proc in readers:
        proc.join()
    for _ in range(len(args.gpus)):
        input_queue.put(None)
    output_queue.put(None)
    writer_proc.join()