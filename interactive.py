#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import namedtuple
import numpy as np
import sys
import scipy.misc

import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator


Batch = namedtuple('Batch', 'srcs tokens lengths')
Translation = namedtuple('Translation', 'src_str hypos alignments')


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, src_dict, max_positions):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = data.EpochBatchIterator(
        dataset=data.LanguagePairDataset(tokens, lengths, src_dict),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, model_args = utils.load_ensemble_for_inference(model_paths, task, model_arg_overrides=eval(args.model_overrides))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(beamable_mm_beam_size=None if args.no_beamable_mm else args.beam)
        if args.fp16:
            model.half()

    # Initialize generator
    translator = SequenceGenerator(
        models, tgt_dict, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen, sampling=args.sampling, sampling_topk=args.sampling_topk,
        minlen=args.min_len,
    )

    if use_cuda:
        translator.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    def make_result(src_str, hypos):
        result = Translation(
            src_str='O\t{}'.format(src_str),
            hypos=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu(),
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
            result.alignments.append('A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment))))
        return result

    def process_batch(batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        translations = translator.generate(
            tokens,
            lengths,
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )

        print("translations",translations)
        print(batch.srcs[0])

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]




    batch, batch_indices = next(make_batches(["My name is John . "], args, src_dict, models[0].max_positions()))
    # indices.extend(batch_indices)
    # results += process_batch(batch)

    translations = translator.generate(
            Variable(batch.tokens),
            Variable(batch.lengths),
            maxlen=int(args.max_len_a * batch.tokens.size(1) + args.max_len_b),
            prefix_tokens=["Je"],
        )

    batch_2, batch_indices_2 = next(make_batches(["John is my name . "], args, src_dict, models[0].max_positions()))
    # indices.extend(batch_indices)
    # results += process_batch(batch)

    print([tgt_dict.symbols[x] for x in np.argsort(-translations)[:10]],"translations")


    translations_2 = translator.generate(
            Variable(batch_2.tokens),
            Variable(batch_2.lengths),
            maxlen=int(args.max_len_a * batch_2.tokens.size(1) + args.max_len_b),
            prefix_tokens=["Je"],
        )

    print([tgt_dict.symbols[x] for x in np.argsort(-translations_2)[:10]],"translations_2")

    # l0 = translations - (scipy.misc.logsumexp([translations, translations_2],axis=0))

    s1_0 = np.exp(translations) / (np.exp(translations)+np.exp(translations_2))
    s1_1 = np.exp(translations_2) / (np.exp(translations)+np.exp(translations_2))

    # print(l0)

    # print()
    

    print([tgt_dict.symbols[x] for x in np.argsort(-s1_0)[:10]],"s1_0")
    print([tgt_dict.symbols[x] for x in np.argsort(-s1_1)[:10]],"s1_1")
    
    raise Exception



    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    for inputs in buffered_read(args.buffer_size):
    #     print("inputs, interactive",inputs)

    # for inputs in ["my name is John."]:

        indices = []
        results = []



        for batch, batch_indices in make_batches(inputs, args, src_dict, models[0].max_positions()):
            print(batch.tokens,batch.lengths)
            raise Exception
            indices.extend(batch_indices)
            results += process_batch(batch)

            # raise Exception

        for i in np.argsort(indices):
            result = results[i]
            print(result.src_str)
            for hypo, align in zip(result.hypos, result.alignments):
                print(hypo)
                print(align)



if __name__ == '__main__':


    #     translations = translator.generate(
    #     Variable(tokens),
    #     Variable(lengths),
    #     maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
    # )

    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    # args.target='fr'
    # args.source='en'
    # args.beam=5
    # args.path='wmt14.en-fr.fconv-py/model.pt wmt14.en-fr.fconv-py'
    # --path wmt14.en-fr.fconv-py/model.pt wmt14.en-fr.fconv-py
    main(args)
