import argparse
from collections import *
from functools import *
from itertools import *
import json
import math
from pathlib import Path
from typing import *

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import *


def reader(args, filenames: List[str], queue: mp.Queue):
    tokenizer = AutoTokenizer.from_pretrained(args.lm_card)

    def put(text_pairs, records, queue):
        _, cands = zip(*text_pairs[-args.batch_size:])
        if 'bert' in args.lm_card:
            inputs = tokenizer.batch_encode_plus(text_pairs[-args.batch_size:],
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=512,
                                                 return_tensors='pt')
        elif 'gpt' in args.lm_card:
            tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer.batch_encode_plus(
                text_pairs[-args.batch_size:],
                padding=True,
                truncation=True,  # TODO
                max_length=512,
                return_tensors='pt',
                return_attention_mask=False,
                return_token_type_ids=True)

        queue.put([records[-args.batch_size:], cands, inputs])
        del records[-args.batch_size:]
        del text_pairs[-args.batch_size:]

    records = []
    text_pairs = []
    for filename in filenames:
        lines = open(filename).readlines()
        jsons = list(map(json.loads, lines))
        num_cands = sum(len(js['candidates']) for js in jsons)
        for lineno, js in enumerate(jsons):
            records.extend(
                repeat([filename, num_cands, lineno, js['page'], js['query']],
                       len(js['candidates'])))
            text_pairs.extend(
                [f'{tokenizer.bos_token} {js["query"]}', f' {cand}']
                for cand in js['candidates'])
        while len(text_pairs) >= args.batch_size:
            put(text_pairs, records, queue)

    if text_pairs:
        put(text_pairs, records, queue)

    while True:
        pass


def estimator(args, device, input_queue, output_queue):
    torch.set_printoptions(precision=3, sci_mode=False)

    if 'bert' in args.lm_card:
        lm = AutoModelForNextSentencePrediction.from_pretrained(args.lm_card)
    elif 'gpt' in args.lm_card:
        lm = lm = GPT2LMHeadModel.from_pretrained(args.lm_card)
    else:
        raise ValueError()
    lm = lm.eval().to(device)

    while True:
        records, cands, inputs = input_queue.get()
        if 'bert' in args.lm_card:
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = lm(**inputs)
                scores = outputs['logits'].softmax(1)[:, 0]
        elif 'gpt' in args.lm_card:
            inputs = inputs.to(device)
            with torch.no_grad():
                token_type_ids = inputs.pop('token_type_ids')
                outputs = lm(inputs['input_ids'])
                cumsum = token_type_ids.cumsum(1)
                mask = F.pad((token_type_ids == 1) & (cumsum <= args.win_size),
                             [-1, 1, 0, 0])
                # scores = outputs['logits'].log_softmax(2).gather(
                #     2,
                #     F.pad(inputs['input_ids'],
                #           [-1, 1, 0, 0])[:, :, None]).squeeze(2).masked_fill(
                #               ~mask, 0).sum(1) / mask.sum(1)
                input_ids = F.pad(inputs['input_ids'], [-1, 1, 0, 0])[:, :,
                                                                      None]
                scores = outputs['logits'].log_softmax(2).gather(
                    2, input_ids).squeeze(2).masked_fill(
                        ~mask, 0).sum(1) / mask.sum(1)

        for record, cand, score in zip(records, cands, scores):
            output_queue.put([record, cand, score.item()])
        del records, cands, inputs


def writer(args, queue):
    while True:
        try:
            filename, jsonl = queue.get()
        except TypeError:
            break
        open(str(filename).replace('queries', 'results'), 'w').write('\n'.join(
            map(partial(json.dumps, ensure_ascii=False),
                (jsonl[lineno] for lineno in sorted(jsonl)))))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    logging.set_verbosity_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--dir', type=Path)
    parser.add_argument('--lm-card',
                        type=str,
                        default='allenai/scibert_scivocab_cased')
    parser.add_argument('--num-readers', type=int)
    parser.add_argument('--num-writers', type=int)
    parser.add_argument('--win-size', type=int)
    args = parser.parse_args()

    filenames = list(args.dir.glob('*.*.pdf/scale-1x/base/queries.jsonl'))

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
                       torch.device(f'cuda:{gpu}'), input_queue, output_queue
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
        'page': None,
        'query': None,
        'candidates': []
    }))
    while num_files > 0:
        [filename, num_cands, lineno, page,
         query], cand, score = output_queue.get()

        js = jsons[filename][lineno]
        js.update({'page': page, 'query': query})
        js['candidates'].append([cand, score])

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