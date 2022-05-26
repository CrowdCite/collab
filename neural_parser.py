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
import torch.nn.functional as F
from transformers import *

import boto3
from tqdm import tqdm

chaini = chain.from_iterable


def reader(args, queue: mp.Queue, fns: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(args.lm_card)
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


def estimator(args, device, input_queue: mp.Queue, output_queue: mp.Queue):
    lm = AutoModelForNextSentencePrediction.from_pretrained(
        args.lm_card).to(device)
    while True:
        try:
            keys, query, cands, inputs = input_queue.get()
        except TypeError:
            break
        outputs = lm(**inputs)  # (B * N, 2)
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
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()

    seqs: List[List[str]] = pickle.load(open(args.dat_dir, "rb"))
    random.shuffle(seqs)
    train_seqs, dev_seqs = seqs[:args.num_train_items], seqs[args.
                                                             num_train_items:]

    fns = glob("queries.*.json")

    input_queue = mp.Queue
    output_queue = mp.Queue
    num_files_per_worker = math.ceil(len(fns) / args.num_workers)
    readers = [
        mp.Process(target=reader,
                   args=[
                       input_queue, fns[idx * num_files_per_worker:(idx + 1) *
                                        num_files_per_worker]
                   ]) for idx in range(args.num_readers)
    ]
    for reader in readers:
        reader.start()
    estimators = [
        mp.Process(target=estimator,
                   args=[
                       args,
                       torch.device(f"cuda:{gpu}"), input_queue, output_queue
                   ]) for gpu in args.gpus
    ]
    for estimator in estimators:
        estimator.start()
    writer_proc = mp.Process(target=writer, args=[])

    for reader in readers:
        reader.join()
    for _ in range(len(estimators)):
        input_queue.put(None)
    for estimator in estimators:
        estimator.join()
    output_queue.put(None)
    writer.join()