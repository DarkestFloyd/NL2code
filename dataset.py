from __future__ import division
import copy

import nltk
from collections import OrderedDict, defaultdict
from query_grammar import QueryParser
import logging
import collections
import numpy as np
import string
import re
import astor
import math
import re

# from canon_utils import fetch_queries_codes, parsable_code, reproducable_code
# from canonicalization import canonicalize_bunch
# from type_simulation  import TYPES

# import multiprocessing
# import multiprocessing as mp

from itertools import chain

from nn.utils.io_utils import serialize_to_file, deserialize_from_file
from file_utils import file_contents

import config
from lang.py.parse import get_grammar
from lang.py.unaryclosure import get_top_unary_closures, apply_unary_closures

# define dataset files
ALLANNO   = "./en-django/all.anno"
ALLCODE   = "./en-django/all.code"
TRAINANNO = "./en-django/train.anno"
DEVANNO   = "./en-django/dev.anno"
TESTANNO  = "./en-django/test.anno"
#GENALLANNO = "./gendata/pycin_all.anno"
#GENALLCODE = "./gendata/pycin_all.code"
GENALLANNO = "./gendata/all.anno"
GENALLCODE = "./gendata/all.code"

# define actions
APPLY_RULE = 0
GEN_TOKEN = 1
COPY_TOKEN = 2
GEN_COPY_TOKEN = 3

ACTION_NAMES = {APPLY_RULE: 'APPLY_RULE',
                GEN_TOKEN: 'GEN_TOKEN',
                COPY_TOKEN: 'COPY_TOKEN',
                GEN_COPY_TOKEN: 'GEN_COPY_TOKEN'}

class Action(object):
    def __init__(self, act_type, data):
        self.act_type = act_type
        self.data = data

    def __repr__(self):
        data_str = self.data if not isinstance(self.data, dict) else \
            ', '.join(['%s: %s' % (k, v) for k, v in self.data.iteritems()])
        repr_str = 'Action{%s}[%s]' % (ACTION_NAMES[self.act_type], data_str)

        return repr_str


class Vocab(object):
    def __init__(self):
        self.token_id_map = OrderedDict()
        self.insert_token('<pad>')
        self.insert_token('<unk>')
        self.insert_token('<eos>')

    @property
    def unk(self):
        return self.token_id_map['<unk>']

    @property
    def eos(self):
        return self.token_id_map['<eos>']

    def __getitem__(self, item):
        if item in self.token_id_map:
            return self.token_id_map[item]

        logging.debug('encounter one unknown word [%s]' % item)
        return self.token_id_map['<unk>']

    def __contains__(self, item):
        return item in self.token_id_map

    @property
    def size(self):
        return len(self.token_id_map)

    def __setitem__(self, key, value):
        self.token_id_map[key] = value

    def __len__(self):
        return len(self.token_id_map)

    def __iter__(self):
        return self.token_id_map.iterkeys()

    def iteritems(self):
        return self.token_id_map.iteritems()

    def complete(self):
        self.id_token_map = dict((v, k) for (k, v) in self.token_id_map.iteritems())

    def get_token(self, token_id):
        return self.id_token_map[token_id]

    def insert_token(self, token):
        if token in self.token_id_map:
            return self[token]
        else:
            idx = len(self)
            self[token] = idx

            return idx


replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))


def tokenize(str):
    str = str.translate(replace_punctuation)
    return nltk.word_tokenize(str)


def gen_vocab(tokens, vocab_size=3000, freq_cutoff=5):

    word_freq = defaultdict(int)

    for token in tokens:
        word_freq[token] += 1

    print 'total num. of tokens: %d' % len(word_freq)

    words_freq_cutoff = [w for w in word_freq if word_freq[w] >= freq_cutoff]
    print 'num. of words appear at least %d: %d' % (freq_cutoff, len(words_freq_cutoff))

    ranked_words = sorted(words_freq_cutoff, key=word_freq.get, reverse=True)[:vocab_size-2]
    ranked_words = set(ranked_words)

    vocab = Vocab()
    for token in tokens:
        if token in ranked_words:
            vocab.insert_token(token)

    vocab.complete()

    return vocab


class DataEntry:
    def __init__(self, raw_id, query, parse_tree, code, actions, meta_data=None):
        self.raw_id = raw_id
        self.eid = -1
        # FIXME: rename to query_token
        self.query = query
        self.parse_tree = parse_tree
        self.actions = actions
        self.code = code
        self.meta_data = meta_data

    @property
    def data(self):
        if not hasattr(self, '_data'):
            assert self.dataset is not None, 'No associated dataset for the example'

            self._data = self.dataset.get_prob_func_inputs([self.eid])

        return self._data

    def copy(self):
        e = DataEntry(self.raw_id, self.query, self.parse_tree, self.code, self.actions, self.meta_data)

        return e

    def __repr__(self):

        s = "-" * 60

        s = s + "\n" * 2
        s = s + str(self.raw_id)
        s = s + "\n" * 2
        s = s + str(self.eid)
        s = s + "\n" * 2
        s = s + str(self.query)
        s = s + "\n" * 2
        s = s + str(self.code)
        s = s + "\n" * 2
        s = s + str(self.meta_data)
        s = s + "\n" * 2

        return s


class DataSet:
    def __init__(self, annot_vocab, terminal_vocab, grammar, name='train_data', phrase_vocab = None, pos_vocab = None):
        self.annot_vocab = annot_vocab
        self.terminal_vocab = terminal_vocab
        self.name = name
        self.examples = []
        self.data_matrix = dict()
        self.grammar = grammar
        self.phrase_vocab = phrase_vocab
        self.pos_vocab    = pos_vocab

    def __repr__(self):

        s = "#" * 60
        s = s + "\n" * 2
        s = s + str(self.annot_vocab)
        s = s + "\n" * 2
        s = s + str(self.terminal_vocab)
        s = s + "\n" * 2
        s = s + str(self.phrase_vocab)
        s = s + "\n" * 2
        s = s + str(self.pos_vocab)
        s = s + "\n" * 2

        return s

    def add(self, example):
        example.eid = len(self.examples)
        example.dataset = self
        self.examples.append(example)

    def get_dataset_by_ids(self, ids, name):
        dataset = DataSet(self.annot_vocab, self.terminal_vocab,
                          self.grammar, name)
        for eid in ids:
            example_copy = self.examples[eid].copy()
            dataset.add(example_copy)

        for k, v in self.data_matrix.iteritems():
            dataset.data_matrix[k] = v[ids]

        return dataset

    @property
    def count(self):
        if self.examples:
            return len(self.examples)

        return 0

    def get_examples(self, ids):
        if isinstance(ids, collections.Iterable):
            return [self.examples[i] for i in ids]
        else:
            return self.examples[ids]

    def get_prob_func_inputs(self, ids):
        order = ['query_tokens', 'tgt_action_seq', 'tgt_action_seq_type',
                 'tgt_node_seq', 'tgt_par_rule_seq', 'tgt_par_t_seq',
                 'query_tokens_phrase', 'query_tokens_pos']#, 'query_tokens_cid']

        max_src_seq_len = max(len(self.examples[i].query) for i in ids)
        max_tgt_seq_len = max(len(self.examples[i].actions) for i in ids)

        logging.debug('max. src sequence length: %d', max_src_seq_len)
        logging.debug('max. tgt sequence length: %d', max_tgt_seq_len)

        data = []
        for entry in order:
            if entry == 'query_tokens':
                data.append(self.data_matrix[entry][ids, :max_src_seq_len])
            elif entry == 'query_tokens_phrase':
                data.append(self.data_matrix[entry][ids, :max_src_seq_len])
            elif entry == 'query_tokens_pos':
                data.append(self.data_matrix[entry][ids, :max_src_seq_len])
            # elif entry == 'query_tokens_cid':
                # data.append(self.data_matrix[entry][ids, :max_src_seq_len])
            else:
                data.append(self.data_matrix[entry][ids, :max_tgt_seq_len])

        return data


    def init_data_matrices(self, max_query_length=70, max_example_action_num=100):

        logging.info('init data matrices for [%s] dataset', self.name)
        annot_vocab = self.annot_vocab
        terminal_vocab = self.terminal_vocab

        phrase_vocab = self.phrase_vocab
        pos_vocab    = self.pos_vocab

        # figure out unique ids for phrase and pos vocab
        phrase_vocab_uid = {}
        pos_vocab_uid    = {}

        # fill phrase_vocab unique id
        for idx, pv in enumerate(phrase_vocab):
            assert (phrase_vocab_uid.get(pv) is None)
            phrase_vocab_uid[pv] = idx
        # fill pos vocab unique id
        for idx, posv in enumerate(pos_vocab):
            assert (pos_vocab_uid.get(posv) is None)
            pos_vocab_uid[posv] = idx

        # np.max([len(e.query) for e in self.examples])
        # np.max([len(e.rules) for e in self.examples])

        query_tokens        = self.data_matrix['query_tokens'] = np.zeros((self.count, max_query_length), dtype='int32')
        query_tokens_phrase = self.data_matrix['query_tokens_phrase'] = np.zeros((self.count, max_query_length), dtype='int32')
        query_tokens_pos    = self.data_matrix['query_tokens_pos'] = np.zeros((self.count, max_query_length), dtype='int32')
        # query_tokens_cid = self.data_matrix['query_tokens_cid'] = np.zeros((self.count, max_query_length), dtype='int32')

        tgt_node_seq = self.data_matrix['tgt_node_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_par_rule_seq = self.data_matrix['tgt_par_rule_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_par_t_seq = self.data_matrix['tgt_par_t_seq'] = np.zeros((self.count, max_example_action_num), dtype='int32')
        tgt_action_seq = self.data_matrix['tgt_action_seq'] = np.zeros((self.count, max_example_action_num, 3), dtype='int32')
        tgt_action_seq_type = self.data_matrix['tgt_action_seq_type'] = np.zeros((self.count, max_example_action_num, 3), dtype='int32')
        for eid, example in enumerate(self.examples):

            exg_query_tokens = example.query[:max_query_length]
            exg_query_tokens_phrase = example.meta_data['phrase'][:max_query_length]
            exg_query_tokens_pos    = example.meta_data['pos'][:max_query_length]
            exg_action_seq = example.actions[:max_example_action_num]

            for tid, token in enumerate(normalize_query_tokens(exg_query_tokens)):
                token_id = annot_vocab[token]
                query_tokens[eid, tid] = token_id

            for tid, token in enumerate(exg_query_tokens):
                # if the token contains some _[0-9], thats the canon id
                res = re.findall(r"_([0-9]+)", token)
                # id
                cid = 0
                if len(res) == 1:
                    cid = int(res[0]) + 1
                #print token, " 's cid ", cid
                # we got the cid
                query_tokens_cid[eid, tid] = cid

            for tid, p in enumerate(exg_query_tokens_phrase):
                assert (phrase_vocab_uid.get(p) is not None)
                phrase_id = phrase_vocab_uid[p]
                query_tokens_phrase[eid, tid] = phrase_id

            for tid, pos in enumerate(exg_query_tokens_pos):
                assert (pos_vocab_uid.get(pos) is not None)
                pos_id = pos_vocab_uid[pos]
                query_tokens_pos[eid, tid] = pos_id

            assert len(exg_action_seq) > 0

            for t, action in enumerate(exg_action_seq):

                if action.act_type == APPLY_RULE:
                    rule = action.data['rule']
                    tgt_action_seq[eid, t, 0] = self.grammar.rule_to_id[rule]
                    tgt_action_seq_type[eid, t, 0] = 1

                elif action.act_type == GEN_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1

                elif action.act_type == COPY_TOKEN:
                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1

                elif action.act_type == GEN_COPY_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1

                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1
                else:
                    raise RuntimeError('wrong action type!')

                # parent information
                rule = action.data['rule']
                parent_rule = action.data['parent_rule']
                tgt_node_seq[eid, t] = self.grammar.get_node_type_id(rule.parent)
                if parent_rule:
                    tgt_par_rule_seq[eid, t] = self.grammar.rule_to_id[parent_rule]
                else:
                    assert t == 0
                    tgt_par_rule_seq[eid, t] = -1

                # parent hidden states
                parent_t = action.data['parent_t']
                tgt_par_t_seq[eid, t] = parent_t

            example.dataset = self


class DataHelper(object):
    @staticmethod
    def canonicalize_query(query):
        return query


def parse_django_dataset_nt_only():
    from parse import parse_django

    #annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
    annot_file = ALLANNO

    vocab = gen_vocab(annot_file, vocab_size=4500)

    #code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'
    code_file = ALLCODE

    grammar, all_parse_trees = parse_django(code_file)

    train_data = DataSet(vocab, grammar, name='train')
    dev_data = DataSet(vocab, grammar, name='dev')
    test_data = DataSet(vocab, grammar, name='test')

    # train_data

    #train_annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/train.anno'
    train_annot_file = TRAINANNO
    train_parse_trees = all_parse_trees[0:16000]
    for line, parse_tree in zip(open(train_annot_file), train_parse_trees):
        if parse_tree.is_leaf:
            continue

        line = line.strip()
        tokens = tokenize(line)
        entry = DataEntry(tokens, parse_tree)

        train_data.add(entry)

    train_data.init_data_matrices()

    # dev_data

    #dev_annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/dev.anno'
    dev_annot_file = DEVANNO
    dev_parse_trees = all_parse_trees[16000:17000]
    for line, parse_tree in zip(open(dev_annot_file), dev_parse_trees):
        if parse_tree.is_leaf:
            continue

        line = line.strip()
        tokens = tokenize(line)
        entry = DataEntry(tokens, parse_tree)

        dev_data.add(entry)

    dev_data.init_data_matrices()

    # test_data

    #test_annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/test.anno'
    test_annot_file = TESTANNO
    test_parse_trees = all_parse_trees[17000:18805]
    for line, parse_tree in zip(open(test_annot_file), test_parse_trees):
        if parse_tree.is_leaf:
            continue

        line = line.strip()
        tokens = tokenize(line)
        entry = DataEntry(tokens, parse_tree)

        test_data.add(entry)

    test_data.init_data_matrices()

    serialize_to_file((train_data, dev_data, test_data), 'django.typed_rule.bin')


def gen_phrase_vocab(data):

    all_phrase = []

    for entry in data:

        phrase = entry['phrase']

        all_phrase = all_phrase + phrase

    return list(set(all_phrase))

def gen_pos_vocab(data):

    all_pos = []

    for entry in data:

        pos = entry['pos']

        all_pos = all_pos + pos

    return list(set(all_pos))

def normalize_query_tokens(tokens):
    # consider all TYPES tokens in base form. example STR_0 -> STR
    normalized_tokens = []
    for t in tokens:
        for k, v in TYPES.iteritems():
            t = re.sub(v + "_[0-9]+", v, t)

        normalized_tokens.append(t)
    assert (len(tokens) == len(normalized_tokens))

    return normalized_tokens

def parse_django_dataset():
    from lang.py.parse import parse_raw
    from lang.util import escape
    MAX_QUERY_LENGTH = 70
    UNARY_CUTOFF_FREQ = 30

    ##annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
    ##code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'
    #annot_file = ALLANNO
    #code_file  = ALLCODE
    annot_file = GENALLANNO
    code_file  = GENALLCODE

    #data = preprocess_dataset(annot_file, code_file)
    #data = preprocess_gendataset(annot_file, code_file)
    data = preprocess_syndataset(annot_file, code_file)

    # print data
    #for e in data:

    #    print "-" * 60
    #    print "\n\n"
    #    print "idx - ", e['id']
    #    print "\n\n"
    #    print "query tokens - ", e['query_tokens']
    #    print "\n\n"
    #    print "code - ", e['code']
    #    print "\n\n"
    #    print "str_map - ", e['str_map']
    #    print "\n\n"
    #    print "raw_code - ", e['raw_code']
    #    print "\n\n"
    #    print "bannot - ", e['bannot']
    #    print "\n\n"
    #    print "bcode  - ", e['bcode']
    #    print "\n\n"
    #    print "ref_type - ", e['ref_type']
    #    print "\n\n"

    for e in data:
        e['parse_tree'] = parse_raw(e['code'])

    parse_trees = [e['parse_tree'] for e in data]

    # apply unary closures
    # unary_closures = get_top_unary_closures(parse_trees, k=0, freq=UNARY_CUTOFF_FREQ)
    # for i, parse_tree in enumerate(parse_trees):
    #     apply_unary_closures(parse_tree, unary_closures)

    # build the grammar
    grammar = get_grammar(parse_trees)

    # write grammar
    with open('django.grammar.unary_closure.txt', 'w') as f:
        for rule in grammar:
            f.write(rule.__repr__() + '\n')

    # # build grammar ...
    # from lang.py.py_dataset import extract_grammar
    # grammar, all_parse_trees = extract_grammar(code_file)

    annot_tokens = list(chain(*[e['query_tokens'] for e in data]))
    annot_tokens = normalize_query_tokens(annot_tokens)

    annot_vocab = gen_vocab(annot_tokens, vocab_size=5000, freq_cutoff=3) # gen_vocab(annot_tokens, vocab_size=5980)
    #annot_vocab = gen_vocab(annot_tokens, vocab_size=5000, freq_cutoff=0) # gen_vocab(annot_tokens, vocab_size=5980)

    terminal_token_seq = []
    empty_actions_count = 0

    # helper function begins
    def get_terminal_tokens(_terminal_str):
        # _terminal_tokens = filter(None, re.split('([, .?!])', _terminal_str)) # _terminal_str.split('-SP-')
        # _terminal_tokens = filter(None, re.split('( )', _terminal_str))  # _terminal_str.split('-SP-')
        tmp_terminal_tokens = _terminal_str.split(' ')
        _terminal_tokens = []
        for token in tmp_terminal_tokens:
            if token:
                _terminal_tokens.append(token)
            _terminal_tokens.append(' ')

        return _terminal_tokens[:-1]
        # return _terminal_tokens
    # helper function ends

    # first pass
    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        parse_tree = entry['parse_tree']

        for node in parse_tree.get_leaves():
            if grammar.is_value_node(node):
                terminal_val = node.value
                terminal_str = str(terminal_val)

                terminal_tokens = get_terminal_tokens(terminal_str)

                for terminal_token in terminal_tokens:
                    assert len(terminal_token) > 0
                    terminal_token_seq.append(terminal_token)

    terminal_vocab = gen_vocab(terminal_token_seq, vocab_size=5000, freq_cutoff=3)
    #terminal_vocab = gen_vocab(terminal_token_seq, vocab_size=5000, freq_cutoff=0)
    phrase_vocab = gen_phrase_vocab(data)
    pos_vocab    = gen_pos_vocab(data)
    #assert '_STR:0_' in terminal_vocab

    train_data = DataSet(annot_vocab, terminal_vocab, grammar, 'train_data', phrase_vocab, pos_vocab)
    dev_data = DataSet(annot_vocab, terminal_vocab, grammar, 'dev_data', phrase_vocab, pos_vocab)
    test_data = DataSet(annot_vocab, terminal_vocab, grammar, 'test_data', phrase_vocab, pos_vocab)

    all_examples = []

    can_fully_gen_num = 0

    # second pass
    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        str_map = entry['str_map']
        parse_tree = entry['parse_tree']

        rule_list, rule_parents = parse_tree.get_productions(include_value_node=True)

        #print "Rule List - "
        #for r in rule_list:
        #    print "Rule -", r

        #for k, v in rule_parents.iteritems():
        #    print "Rule parents - ", k, " - ", v
        #print "Rule parents - ", rule_parents

        actions = []
        can_fully_gen = True
        rule_pos_map = dict()

        for rule_count, rule in enumerate(rule_list):
            if not grammar.is_value_node(rule.parent):
                assert rule.value is None
                parent_rule = rule_parents[(rule_count, rule)][0]
                if parent_rule:
                    parent_t = rule_pos_map[parent_rule]
                else:
                    parent_t = 0

                rule_pos_map[rule] = len(actions)

                d = {'rule': rule, 'parent_t': parent_t, 'parent_rule': parent_rule}
                action = Action(APPLY_RULE, d)

                actions.append(action)
            else:
                assert rule.is_leaf

                parent_rule = rule_parents[(rule_count, rule)][0]
                parent_t = rule_pos_map[parent_rule]

                terminal_val = rule.value
                terminal_str = str(terminal_val)
                terminal_tokens = get_terminal_tokens(terminal_str)

                # assert len(terminal_tokens) > 0

                for terminal_token in terminal_tokens:
                    term_tok_id = terminal_vocab[terminal_token]
                    tok_src_idx = -1
                    try:
                        tok_src_idx = query_tokens.index(terminal_token)
                    except ValueError:
                        pass

                    d = {'literal': terminal_token, 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}

                    # cannot copy, only generation
                    # could be unk!
                    if tok_src_idx < 0 or tok_src_idx >= MAX_QUERY_LENGTH:
                        action = Action(GEN_TOKEN, d)
                        if terminal_token not in terminal_vocab:
                            if terminal_token not in query_tokens:
                                # print terminal_token
                                can_fully_gen = False
                    else:  # copy
                        if term_tok_id != terminal_vocab.unk:
                            d['source_idx'] = tok_src_idx
                            action = Action(GEN_COPY_TOKEN, d)
                        else:
                            d['source_idx'] = tok_src_idx
                            action = Action(COPY_TOKEN, d)

                    actions.append(action)

                d = {'literal': '<eos>', 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}
                actions.append(Action(GEN_TOKEN, d))

        if len(actions) == 0:
            empty_actions_count += 1
            continue

        example = DataEntry(idx, query_tokens, parse_tree, code, actions,
                {'raw_code': entry['raw_code'], 'str_map': entry['str_map'],
                 'phrase' : entry['phrase'], 'pos' : entry['pos'],
                 'bannot' : entry['bannot'], 'bcode' : entry['bcode'],
                 'ref_type' : entry['ref_type']})

        if can_fully_gen:
            can_fully_gen_num += 1

        # train, valid, test
        if 0 <= idx < 13000:
            train_data.add(example)
        elif 13000 <= idx < 14000:
            dev_data.add(example)
        else:
            test_data.add(example)

        # modified train valid test counts
        #if 0 <= idx < 10000:
        #    train_data.add(example)
        #elif 10000 <= idx < 11000:
        #    dev_data.add(example)
        #else:
        #    test_data.add(example)

        all_examples.append(example)

    # print statistics
    max_query_len = max(len(e.query) for e in all_examples)
    max_actions_len = max(len(e.actions) for e in all_examples)

    serialize_to_file([len(e.query) for e in all_examples], 'query.len')
    serialize_to_file([len(e.actions) for e in all_examples], 'actions.len')

    logging.info('examples that can be fully reconstructed: %d/%d=%f',
                 can_fully_gen_num, len(all_examples),
                 can_fully_gen_num / len(all_examples))
    logging.info('empty_actions_count: %d', empty_actions_count)
    logging.info('max_query_len: %d', max_query_len)
    logging.info('max_actions_len: %d', max_actions_len)

    train_data.init_data_matrices()
    dev_data.init_data_matrices()
    test_data.init_data_matrices()

    #print train_data

    ## print train_data matrix
    #print "Data matrix: query_tokens "
    #print train_data.data_matrix['query_tokens']
    #print "\n" * 2
    #print "Data matrix : query_tokens_phrase"
    #print "\n" * 2
    #print train_data.data_matrix['query_tokens_phrase']
    #print "\n" * 2
    #print "Data matrix : query_tokens_pos"
    #print "\n" * 2
    #print train_data.data_matrix['query_tokens_pos']
    #print "\n" * 2
    #print "Data matrix : query_tokens_cid"
    #print "\n" * 2
    #print train_data.data_matrix['query_tokens_cid']
    #print "\n" * 2

    ## print few data entries
    #for d in train_data.examples[:5]:
    #    print "\n" * 2
    #    print d

    ## lets print dataset for good measure

    serialize_to_file((train_data, dev_data, test_data),
                      # 'data/django.pnet.qparse.dataset.freq3.par_info.refact.space_only.bin')
                      'data/django.pnet.fullcanon.dataset.freq3.par_info.refact.space_only.bin')
                      # 'data/django.pnet.dataset.freq3.par_info.refact.space_only.bin')
                      #'data/django.cleaned.dataset.freq3.par_info.refact.space_only.order_by_ulink_len.bin')
                      # 'data/django.cleaned.dataset.freq5.par_info.refact.space_only.unary_closure.freq{UNARY_CUTOFF_FREQ}.order_by_ulink_len.bin'.format(UNARY_CUTOFF_FREQ=UNARY_CUTOFF_FREQ))

    return train_data, dev_data, test_data


def check_terminals():
    from parse import parse_django, unescape
    #grammar, parse_trees = parse_django('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code')
    #annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'

    grammar, parse_trees = parse_django(ALLCODE)
    annot_file = ALLANNO

    unique_terminals = set()
    invalid_terminals = set()

    for i, line in enumerate(open(annot_file)):
        parse_tree = parse_trees[i]
        utterance = line.strip()

        leaves = parse_tree.get_leaves()
        # tokens = set(nltk.word_tokenize(utterance))
        leave_tokens = [l.label for l in leaves if l.label]

        not_included = []
        for leaf_token in leave_tokens:
            leaf_token = str(leaf_token)
            leaf_token = unescape(leaf_token)
            if leaf_token not in utterance:
                not_included.append(leaf_token)

                if len(leaf_token) <= 15:
                    unique_terminals.add(leaf_token)
                else:
                    invalid_terminals.add(leaf_token)
            else:
                if isinstance(leaf_token, str):
                    print leaf_token

        # if not_included:
        #     print str(i) + '---' + ', '.join(not_included)

    # print 'num of unique leaves: %d' % len(unique_terminals)
    # print unique_terminals
    #
    # print 'num of invalid leaves: %d' % len(invalid_terminals)
    # print invalid_terminals


def query_to_data(query, annot_vocab):
    query_tokens = query.split(' ')
    token_num = min(config.max_query_length, len(query_tokens))
    data = np.zeros((1, token_num), dtype='int32')

    for tid, token in enumerate(query_tokens[:token_num]):
        token_id = annot_vocab[token]

        data[0, tid] = token_id

    return data


QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")


def canonicalize_query(query):
    """
    canonicalize the query, replace strings to a special place holder
    """
    str_count = 0
    str_map = dict()

    matches = QUOTED_STRING_RE.findall(query)
    # de-duplicate
    cur_replaced_strs = set()
    for match in matches:
        # If one or more groups are present in the pattern,
        # it returns a list of groups
        quote = match[0]
        str_literal = quote + match[1] + quote

        if str_literal in cur_replaced_strs:
            continue

        # FIXME: substitute the ' % s ' with
        if str_literal in ['\'%s\'', '\"%s\"']:
            continue

        str_repr = '_STR:%d_' % str_count
        str_map[str_literal] = str_repr

        query = query.replace(str_literal, str_repr)

        str_count += 1
        cur_replaced_strs.add(str_literal)

    # tokenize
    query_tokens = nltk.word_tokenize(query)

    new_query_tokens = []
    # break up function calls like foo.bar.func
    for token in query_tokens:
        new_query_tokens.append(token)
        i = token.find('.')
        if 0 < i < len(token) - 1:
            new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
            new_query_tokens.extend(new_tokens)

    query = ' '.join(new_query_tokens)

    return query, str_map


def canonicalize_example(query, code):
    from lang.py.parse import parse_raw, parse_tree_to_python_ast, canonicalize_code as make_it_compilable
    import astor, ast

    canonical_query, str_map = canonicalize_query(query)
    canonical_code = code

    for str_literal, str_repr in str_map.iteritems():
        canonical_code = canonical_code.replace(str_literal, '\'' + str_repr + '\'')

    canonical_code = make_it_compilable(canonical_code)

    # sanity check
    parse_tree = parse_raw(canonical_code)
    gold_ast_tree = ast.parse(canonical_code).body[0]
    gold_source = astor.to_source(gold_ast_tree)
    ast_tree = parse_tree_to_python_ast(parse_tree)
    source = astor.to_source(ast_tree)

    assert gold_source == source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, source)

    query_tokens = canonical_query.split(' ')

    return query_tokens, canonical_code, str_map


def process_query(query, code):
    from parse import code_to_ast, ast_to_tree, tree_to_ast, parse
    import astor
    str_count = 0
    str_map = dict()

    match_count = 1
    match = QUOTED_STRING_RE.search(query)
    while match:
        str_repr = '_STR:%d_' % str_count
        str_literal = match.group(0)
        str_string = match.group(2)

        match_count += 1

        # if match_count > 50:
        #     return
        #

        query = QUOTED_STRING_RE.sub(str_repr, query, 1)
        str_map[str_literal] = str_repr

        str_count += 1
        match = QUOTED_STRING_RE.search(query)

        code = code.replace(str_literal, '\'' + str_repr + '\'')

    # clean the annotation
    # query = query.replace('.', ' . ')

    for k, v in str_map.iteritems():
        if k == '\'%s\'' or k == '\"%s\"':
            query = query.replace(v, k)
            code = code.replace('\'' + v + '\'', k)

    # tokenize
    query_tokens = nltk.word_tokenize(query)

    new_query_tokens = []
    # break up function calls
    for token in query_tokens:
        new_query_tokens.append(token)
        i = token.find('.')
        if 0 < i < len(token) - 1:
            new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
            new_query_tokens.extend(new_tokens)

    # check if the code compiles
    tree = parse(code)
    ast_tree = tree_to_ast(tree)
    astor.to_source(ast_tree)

    return new_query_tokens, code, str_map

def deoneline(txt):

    orig_line = ""

    return txt.replace(" DCNL ", '\n').replace(" DCSP ", '\t')

def preprocess_gendataset(annot_file, code_file):

    f_annot = open('annot.all.canonicalized.txt', 'w')
    f_code = open('code.all.canonicalized.txt', 'w')

    # get all queries
    contents = file_contents(annot_file)
    examples = contents.split("\n\n")
    examples = examples[:-1]
    queries = []

    for ex in examples:

        parts = ex.split("\n")
        parts = filter(lambda x: x.strip() != "", parts)
        assert (len(parts) == 2)
        assert (parts[0].startswith("example"))
        eid = int(parts[0].split(" ")[1])
        query = parts[1]
        queries.append(query)


    # get all code
    contents = file_contents(code_file)
    examples = contents.split("\n\n")
    examples = examples[:-1]
    codes = []

    for ex in examples:

        parts = ex.split("\n")
        parts = filter(lambda x: x.strip() != "", parts)
        assert (len(parts) >= 2)
        assert (parts[0].startswith("example"))
        eid = int(parts[0].split(" ")[1])
        code = "\n".join(parts[1:])
        codes.append(code)

    # deoneline all queries and codes
    queries = map(lambda q: deoneline(q), queries)
    codes   = map(lambda c: deoneline(c), codes)

    print "All examples parsed successfully !!"

    examples = []

    err_num = 0

    for idx, (annot, code) in enumerate(zip(queries, codes)):

        annot = annot.strip()
        code  = code.strip()

        #print "Annotation ", annot
        #print "Code       ", code

        #c = raw_input("Press to cont")

        try:
            clean_query_tokens, clean_code, str_map = canonicalize_example(annot, code)
            example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code,
                       'str_map': str_map, 'raw_code': code}
            examples.append(example)

            f_annot.write('example# %d\n' % idx)
            f_annot.write(' '.join(clean_query_tokens) + '\n')
            f_annot.write('%d\n' % len(str_map))
            for k, v in str_map.iteritems():
                f_annot.write('%s ||| %s\n' % (k, v))

            f_code.write('example# %d\n' % idx)
            f_code.write(clean_code + '\n')
        except Exception, e:
            print "Exception raised - ", str(e)
            print "\n\n"
            print code

            err_num += 1

        idx += 1

    f_annot.close()
    f_annot.close()

    # serialize_to_file(examples, 'django.cleaned.bin')

    print 'error num: %d' % err_num
    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples

def batch_deep_phrases(queries):

    n = math.ceil(float(len(queries)) / 18.0)
    n = int(n)

    parts = [ queries[x * n : (x + 1) * n] for x in range(18) ]

    # remove empty parts
    parts = filter(lambda part: part != [], parts)

    # sum of part lengths
    pl_sum = 0
    for part in parts:
        pl_sum = pl_sum + len(part)
    assert (len(queries) == pl_sum)

    # Do parsing for every batch
    qparser = QueryParser()

    result = []

    for part in parts:
        print "Parsing queries ", parts.index(part) * n, " Onwards - ", len(queries)
        result = result + QueryParser.deep_phrases(qparser.parser, part)

    assert (len(result) == len(queries))

    return result

def preprocess_syndataset(annot_file, code_file):

    f_annot = open('annot.all.canonicalized.txt', 'w')
    f_code = open('code.all.canonicalized.txt', 'w')

    queries, codes = fetch_queries_codes(annot_file, code_file)

    queries_codes  = zip(queries, codes)

    #queries_codes = queries_codes[:10]

    print "Pairs fetched ", len(queries_codes)

    queries_codes = map(lambda q_c: (q_c[0], parsable_code(q_c[1])), queries_codes)

    queries_codes = map(lambda q_c: (q_c[0], reproducable_code(q_c[1])), queries_codes)

    queries_codes = filter(lambda q_c: q_c[1].strip() != "", queries_codes)

    queries_codes = filter(lambda q_c: len(q_c[0].split(" ")) < 150, queries_codes)

    queries_codes = filter(lambda q_c: q_c[1] != "", queries_codes)

    print "Parsable and Reproducable and Short :P ", len(queries_codes)

    queries, codes = zip(*queries_codes)

    ref_types, canon_queries, canon_codes = canonicalize_bunch(queries, codes)

    # from here on canon_queries are actual queries and canon_codes are actual codes;
    # queries and codes will be refered to as base queries and base codes; ref_type
    # is for easy conversion
    bqueries = queries
    bcodes   = codes
    queries  = canon_queries
    codes    = canon_codes

    # deoneline all queries and codes
    # annots = map(lambda q: deoneline(q), queries)
    # codes   = map(lambda c: deoneline(c), codes)

    print "All examples parsed successfully !!"

    examples = []

    err_num = 0
    sorted_len = []
    for idx, (annot, code, bannot, bcode, ref_type) in enumerate(zip(queries, codes, bqueries, bcodes, ref_types)):
        annot = annot.strip()
        code = code.strip()

        try:

            clean_query_tokens, clean_code, str_map = canonicalize_example(annot, code)

            if len(clean_query_tokens) > 150:
                raise OSERROR # some error :P

            example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code,
                    'str_map': str_map, 'raw_code': code, 'bannot' : bannot, 'bcode' : bcode, 'ref_type' : ref_type}
            examples.append(example)

        except:
            print "Exception Raised !! "
            print clean_query_tokens
            err_num += 1

        idx += 1


    # get all queries
    queries = []
    for example in examples:
        assert(len(example['query_tokens']) <= 150)
        queries.append(" ".join(example['query_tokens']))

    # get pos information for all the queries
    phrase_pos = batch_deep_phrases(queries)

    assert (len(queries) == len(phrase_pos))

    for idx, example in enumerate(examples):

        qtokens = phrase_pos[idx][0]
        phrase  = phrase_pos[idx][1]
        pos     = phrase_pos[idx][2]

        assert (len(qtokens.split(" ")) == len(phrase) == len(pos))

        # update examples
        example['query_tokens'] = qtokens.split(" ")
        example['phrase']       = phrase
        example['pos']          = pos

        f_annot.write('example# %d\n' % example['id'])
        f_annot.write(' '.join(example['query_tokens']) + '\n')
        f_annot.write('%d\n' % len(example['str_map']))
        for k, v in example['str_map'].iteritems():
            f_annot.write('%s ||| %s\n' % (k, v))

        f_code.write('example# %d\n' % example['id'])
        f_code.write(example['code'] + '\n')


    f_annot.close()
    f_annot.close()

    # serialize_to_file(examples, 'django.cleaned.bin')

    print 'error num: %d' % err_num
    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples


def preprocess_dataset(annot_file, code_file):
    f_annot = open('annot.all.canonicalized.txt', 'w')
    f_code = open('code.all.canonicalized.txt', 'w')

    examples = []

    # get all queries
    contents = file_contents(annot_file)
    examples = contents.split("\n\n")
    examples = examples[:-1]
    annots = []

    for ex in examples:

        parts = ex.split("\n")
        parts = filter(lambda x: x.strip() != "", parts)
        assert (len(parts) == 2)
        assert (parts[0].startswith("example"))
        eid = int(parts[0].split(" ")[1])
        query = parts[1]
        annots.append(query)


    # get all code
    contents = file_contents(code_file)
    examples = contents.split("\n\n")
    examples = examples[:-1]
    codes = []

    for ex in examples:

        parts = ex.split("\n")
        parts = filter(lambda x: x.strip() != "", parts)
        assert (len(parts) >= 2)
        assert (parts[0].startswith("example"))
        eid = int(parts[0].split(" ")[1])
        code = "\n".join(parts[1:])
        codes.append(code)

    annots = annots[:5]
    codes  = codes[:5]

    # deoneline all queries and codes
    # annots = map(lambda q: deoneline(q), queries)
    # codes   = map(lambda c: deoneline(c), codes)

    print "All examples parsed successfully !!"

    examples = []

    err_num = 0
    for idx, (annot, code) in enumerate(zip(annots, codes)):
        annot = annot.strip()
        code = code.strip()
        try:

            clean_query_tokens, clean_code, str_map = canonicalize_example(annot, code)
            example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code,
                       'str_map': str_map, 'raw_code': code}
            examples.append(example)

            print "\n"
            print " ".join(clean_query_tokens)
            print annot

            f_annot.write('example# %d\n' % idx)
            f_annot.write(' '.join(clean_query_tokens) + '\n')
            f_annot.write('%d\n' % len(str_map))
            for k, v in str_map.iteritems():
                f_annot.write('%s ||| %s\n' % (k, v))

            f_code.write('example# %d\n' % idx)
            f_code.write(clean_code + '\n')
        except:
            print code
            err_num += 1

        idx += 1

    f_annot.close()
    f_annot.close()

    # serialize_to_file(examples, 'django.cleaned.bin')

    print 'error num: %d' % err_num
    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples

if __name__== '__main__':
    from nn.utils.generic_utils import init_logging
    init_logging('parse.log')

    # annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
    # code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'

    # preprocess_dataset(annot_file, code_file)

    # parse_django_dataset()
    # check_terminals()

    # print process_query(""" ALLOWED_VARIABLE_CHARS is a string 'abcdefgh"ijklm" nop"%s"qrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.'.""")

    # for i, query in enumerate(open('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno')):
    #     print i, process_query(query)

    # clean_dataset()

    parse_django_dataset()
    # from lang.py.py_dataset import parse_hs_dataset
    # parse_hs_dataset()
