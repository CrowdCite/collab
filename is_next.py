import argparse
from collections import *
from functools import *
from itertools import *
import json
import math
from pathlib import Path
from typing import *

import torch
import torch.multiprocessing as mp
from transformers import *


def reader(args, filenames: List[str], queue: mp.Queue):
    kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(args.lm_card)
    if "gpt2" in args.lm_card:
        tokenizer.pad_token = tokenizer.eos_token
        kwargs["pad_token_id"] = tokenizer.eos_token_id

    def put(text_pairs, records, queue):
        _, cands = zip(*text_pairs[-args.batch_size:])
        queue.put([
            records[-args.batch_size:], cands,
            tokenizer.batch_encode_plus(text_pairs[-args.batch_size:],
                                        padding=True,
                                        truncation=True,
                                        max_length=512,
                                        return_tensors="pt"), kwargs
        ])
        del records[-args.batch_size:]
        del text_pairs[-args.batch_size:]

    records = []
    text_pairs = []
    for filename in filenames:
        lines = open(filename).readlines()
        jsons = list(map(json.loads, lines))
        num_cands = sum(len(js["candidates"]) for js in jsons)
        for lineno, js in enumerate(jsons):
            records.extend(
                repeat([filename, num_cands, lineno, js["page"], js["query"]],
                       len(js["candidates"])))
            text_pairs.extend([js["query"], cand] for cand in js["candidates"])
        while len(text_pairs) >= args.batch_size:
            put(text_pairs, records, queue)

    if text_pairs:
        put(text_pairs, records, queue)

    while True:
        pass


def estimator(args, device, input_queue, output_queue):
    if "bert" in args.lm_card:
        lm = AutoModelForNextSentencePrediction.from_pretrained(args.lm_card)
    elif "gpt" in args.lm_card:
        lm = lm = GPT2LMHeadModel.from_pretrained(args.lm_card)
    else:
        raise ValueError()
    lm = lm.to(device)

    while True:
        records, cands, inputs, kwargs = input_queue.get()
        with torch.no_grad():
            outputs = lm(**inputs.to(device))  # (B * N, 2)

        if "bert" in args.lm_card:
            probs = outputs["logits"].softmax(1)[:, 0]  # (B * N,)
        elif "gpt" in args.lm_card:
            input_ids = inputs["input_ids"]
            logits = outputs["logits"].gather(2, input_ids[:, :,
                                                           None]).squeeze(2)
            mask = (input_ids == kwargs["pad_token_id"])
            logits = logits.masked_fill(mask, 0)
            probs = logits.sum(1) / (logits.size(1) - mask.sum(1))

        for record, cand, prob in zip(records, cands, probs):
            output_queue.put([record, cand, prob.item()])
        del records, cands, inputs


def writer(args, queue):
    while True:
        try:
            filename, jsonl = queue.get()
        except TypeError:
            break
        open(args.out_dir / filename.name, 'w').write('\n'.join(
            map(partial(json.dumps, ensure_ascii=False),
                (jsonl[lineno] for lineno in sorted(jsonl)))))


if __name__ == "__main__":
    mp.set_start_method("spawn")
    logging.set_verbosity_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpus", type=int, nargs='+')
    parser.add_argument("--in-dir", type=Path)
    parser.add_argument("--lm-card",
                        type=str,
                        default="allenai/scibert_scivocab_cased")
    parser.add_argument("--num-readers", type=int)
    parser.add_argument("--num-writers", type=int)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    filenames = list(args.in_dir.glob("*.jsonl"))

    download_queue = mp.Queue()
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    upload_queue = mp.Queue()

    num_files_per_reader = math.ceil(len(filenames) / args.num_readers)
    readers = [
        mp.Process(target=reader,
                   args=[
                       args, filenames[idx * num_files_per_reader:(idx + 1) *
                                       num_files_per_reader], input_queue
                   ]) for idx in range(args.num_readers)
    ]

    estimators = [
        mp.Process(target=estimator,
                   args=[
                       args,
                       torch.device(f"cuda:{gpu}"), input_queue, output_queue
                   ]) for gpu in args.gpus
    ]

    writers = [
        mp.Process(target=writer, args=[args, upload_queue])
        for _ in range(args.num_writers)
    ]

    for proc in chain(readers, estimators, writers):
        proc.start()

    num_files = len(filenames)
    num_cands_by_file = defaultdict(int)
    jsons = defaultdict(lambda: defaultdict(lambda: {
        "page": None,
        "query": None,
        "candidates": []
    }))
    while num_files > 0:
        [filename, num_cands, lineno, page,
         query], cand, prob = output_queue.get()

        js = jsons[filename][lineno]
        js.update({"page": page, "query": query})
        js["candidates"].append([cand, prob])

        num_cands_by_file[filename] += 1
        if num_cands_by_file[filename] == num_cands:
            jsons[filename].default_factory = None
            upload_queue.put([filename, jsons[filename]])
            del jsons[filename]
            num_files -= 1

    for proc in chain(readers, estimators):
        proc.kill()

    for _ in writers:
        upload_queue.put(None)
    for proc in writers:
        proc.join()