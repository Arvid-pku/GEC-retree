# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    LanguagePairWithGraphDataset,
)

from . import FairseqTask, register_task


@register_task('translation_with_graph_attention_with_copy')
class TranslationWithGraphAttentionWithCopyTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('-e', '--edge-lang', default='edge', metavar='EDGE',
                            help='edge language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--max-edge-positions', default=1024, type=int, metavar='N',
                            help='max number of edges in the source graph')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, edge_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.edge_dict = edge_dict


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        edge_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.edge_lang)))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        print('| [{}] dictionary: {} types'.format(args.edge_lang, len(edge_dict)))

        return cls(args, src_dict, tgt_dict, edge_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        def split_exists(split, src, tgt, lang, data_path, dataset_impl):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            return indexed_dataset.dataset_exists(filename, dataset_impl)

        src_datasets = []
        tgt_datasets = []
        edge_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')

            # infer langcode
            if 'test' not in split_k:
                src, tgt, edge = self.args.source_lang, self.args.target_lang, self.args.edge_lang
                if split_exists(split_k, src, tgt, src, data_path, self.args.dataset_impl):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path, self.args.dataset_impl):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
            else:
                src, tgt, edge = self.args.source_lang, self.args.target_lang, self.args.edge_lang
                if split_exists(split_k, src, tgt, src, data_path, 'raw'):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path, 'raw'):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
            if 'test' not in prefix:
                edge_datasets.append(indexed_dataset.make_dataset(prefix + edge, impl=self.args.dataset_impl,
                                                                  fix_lua_indexing=True, dictionary=self.edge_dict))
                src_datasets.append(indexed_dataset.make_dataset(prefix + src, impl=self.args.dataset_impl,
                                                                 fix_lua_indexing=True, dictionary=self.src_dict))
                tgt_datasets.append(indexed_dataset.make_dataset(prefix + tgt, impl=self.args.dataset_impl,
                                                                 fix_lua_indexing=True, dictionary=self.tgt_dict))
            else:
                edge_datasets.append(indexed_dataset.make_dataset(prefix + edge, impl='raw',
                                                                  fix_lua_indexing=True, dictionary=self.edge_dict))
                src_datasets.append(indexed_dataset.make_dataset(prefix + src, impl='raw',
                                                                 fix_lua_indexing=True, dictionary=self.src_dict))
                tgt_datasets.append(indexed_dataset.make_dataset(prefix + tgt, impl='raw',
                                                                 fix_lua_indexing=True, dictionary=self.tgt_dict))

            print('| {} {}'.format(data_path, split_k))
            print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets) == len(edge_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset, edge_dataset = src_datasets[0], tgt_datasets[0], edge_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            edge_dataset = ConcatDataset(edge_datasets, sample_ratios)

        self.datasets[split] = LanguagePairWithGraphDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            edge_dataset, edge_dataset.sizes, self.edge_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            max_edge_positions=self.args.max_edge_positions,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @property
    def edge_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.edge_dict

    def build_generator(self, args):
        if args.score_reference:
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator_with_copy import SequenceGenerator_with_copy
            return SequenceGenerator_with_copy(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                temperature=args.temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
