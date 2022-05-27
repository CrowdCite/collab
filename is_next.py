import argparse
from collections import *
from functools import *
from glob import glob
from itertools import *
import json
import math
# import multiprocessing as mp
from typing import *

import torch
import torch.multiprocessing as mp
from transformers import *


def reader(args, fns: List[str], tokenizer, queue: mp.Queue):
    keys = []
    text_pairs = []
    for fn in fns:
        lns = open(fn).readlines()
        jss = list(map(json.loads, lns))
        num_cands = sum(len(js["candidates"]) for js in jss)
        for lineno, js in enumerate(jss):
            keys.extend(
                repeat([fn, num_cands, lineno, js["page"]],
                       len(js["candidates"])))
            text_pairs.extend([js["query"], cand] for cand in js["candidates"])
        while len(text_pairs) >= args.batch_size:
            _, cands = zip(*text_pairs[-args.batch_size:])
            queue.put([
                keys[-args.batch_size:], text_pairs[-1][0], cands,
                tokenizer.batch_encode_plus(text_pairs[-args.batch_size:],
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="pt")
            ])
            del keys[-args.batch_size:]
            del text_pairs[-args.batch_size:]

    _, cands = zip(*text_pairs[-args.batch_size:])
    queue.put([
        keys[-args.batch_size:], text_pairs[-1][0], cands,
        tokenizer.batch_encode_plus(text_pairs[-args.batch_size:],
                                    padding=True,
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt")
    ])
    del keys[-args.batch_size:]
    del text_pairs[-args.batch_size:]


def writer(queue):
    while True:
        try:
            print(queue.get())
            fn, jsl = queue.get()
            print(fn)
        except TypeError:
            break
        jsonl = []
        for lineno in sorted(jsl):
            [page, query, *_], *_ = jsl[lineno]
            print(lineno, page, query[:25])
            _, _, cands, probs = zip(*jsl[lineno])
            jsonl.append({
                "page": page,
                "query": query,
                "candidates": list(zip(cands, probs))
            })

        open(fn.replace("inputs", "outputs"),
             'w').write('\n'.join(map(json.dumps, jsonl)))


def run(pid, args, input_queue, output_queue):
    device = torch.device(f"cuda:{args.gpus[pid]}")
    torch.cuda.set_device(device)

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                         world_size=len(args.gpus),
                                         rank=pid)

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
            # print(key, query[:10], cand[:10])
            output_queue.put([key, query, cand, prob.item()])


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpus", type=int, nargs='+')
    parser.add_argument("--lm-card",
                        type=str,
                        default="allenai/scibert_scivocab_cased")
    parser.add_argument("--num-cands", type=int)
    parser.add_argument("--num-readers", type=int)
    parser.add_argument("--num-writers", type=int)
    args = parser.parse_args()

    filenames = glob("inputs/*.json")

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    upload_queue = mp.Queue()

    num_files_per_reader = math.ceil(len(filenames) / args.num_readers)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_card)
    readers = [
        mp.Process(target=reader,
                   args=[
                       args, filenames[idx * num_files_per_reader:(idx + 1) *
                                       num_files_per_reader], tokenizer,
                       input_queue
                   ]) for idx in range(args.num_readers)
    ]
    for proc in readers:
        proc.start()

    writers = [
        mp.Process(target=writer, args=[upload_queue])
        for _ in range(args.num_writers)
    ]
    for proc in writers:
        proc.start()

    x = mp.spawn(run, [args, input_queue, output_queue],
                 len(args.gpus),
                 join=False)

    num_readers = args.num_readers
    jsls = defaultdict(lambda: defaultdict(list))
    cand_cnts = defaultdict(int)
    while num_readers > 0:
        print(sum(cand_cnts.values()))
        try:
            [fn, num_cands, lineno,
             page], query, cand, prob = output_queue.get()
        except TypeError:
            num_readers -= 1
            continue
        # print(lineno, page, query[:25])
        jsls[fn][lineno].append([page, query, cand, prob])
        cand_cnts[fn] += 1
        if cand_cnts[fn] == num_cands:
            # print(fn, cand_cnts[fn], num_cands)
            print(fn)
            upload_queue.put([fn, jsls[fn]])
            del jsls[fn]

    for proc in readers:
        proc.join()

    for _ in range(len(args.gpus)):
        input_queue.put(None)

    for _ in writers:
        output_queue.put(None)
    for proc in writers:
        proc.join()
