import sys
import os
import numpy as np
import pandas as pd
import torch
from rpunct import RestorePuncts
import nltk
from nltk.tokenize import sent_tokenize
import argparse
from transformers import GPT2Tokenizer, GPT2Model
from fmri_mrh.textgrid import TextGrid
from fmri_mrh.util import *

TG_DIR = os.path.join(DATA_DIR, 'src', 'derivative', 'TextGrids')
OUT_DIR = os.path.join(PREPROCESSED_DIR, 'texts')
SPECIAL_TOKENS = ('', 'sp', '{sl}', '{lg}', '{ns}', '{cg}', '{br}', '{ls}', '{ns', '{sp}',
    '{ls)', '{br', '{{br}', '{ig}', 'um', 'uh') 
PUNCT_MODEL = RestorePuncts()
GPT_TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
GPT_MODEL = GPT2Model.from_pretrained('gpt2')
CHUNK_LENGTH = 512


def get_table(x, *meta):
    out = [[] for _ in range(len(meta) + 1)]
    for i, _x in enumerate(x):
        _x = _x.strip().lower()
        if _x not in SPECIAL_TOKENS:
            out[0].append(_x)
            for j in range(len(meta)):
                out[j+1].append(meta[j][i])
    text = ' '.join(out[0])
    text = PUNCT_MODEL.punctuate(text)
    out[0] = text.split(' ')
    
    sents = sent_tokenize(text)
    i = 0
    sentids = []
    for j, _sent in enumerate(sents):
        _sent = _sent.split(' ')
        for _ in _sent:
            sentids.append(j)
    
    out.append(sentids)
    out = pd.DataFrame({'word': out[0], 'time': out[1], 'sentid': out[2]})
    out['sentpos'] = out.groupby('sentid').cumcount()
    out['textpos'] = list(range(len(out)))

    return out


def get_adjacency_matrix(words, toks):
    M = np.zeros((len(words), len(toks)))
    w_cur = ''
    j = 0
    for i, w in enumerate(words):
        while w_cur != w and len(w_cur) < len(w):
            w_cur += toks[j]
            M[i, j] = 1
            j += 1
        assert w == w_cur, 'Misaligned tokens. Expected %s, got %s.' % (w, w_cur)
        w_cur = ''
    return M / M.sum(axis=1, keepdims=True)


def get_gpt2_states(df):
    words = df.word.to_list()
    text = ' '.join(words)
    toks_src = GPT_TOKENIZER.tokenize(text)
    input_ids = np.array(GPT_TOKENIZER.convert_tokens_to_ids(toks_src))
    toks = [GPT_TOKENIZER.convert_tokens_to_string(x).strip() for x in toks_src]
    states = None

    M = get_adjacency_matrix(words, toks)

    for i in range(0, len(toks), CHUNK_LENGTH):
        s = max(0, i - CHUNK_LENGTH)
        e = i + CHUNK_LENGTH
        if i == 0:
            len_prefix = 0
        else:
            len_prefix = CHUNK_LENGTH
        _input_ids = input_ids[s:e]
        _input_ids = torch.from_numpy(_input_ids[None, ...])
        _states = GPT_MODEL(
            input_ids=_input_ids,
            output_hidden_states=True
        )['hidden_states']
        if states is None:
            states = []
            for layer in _states:
                states.append([])
        for L, layer in enumerate(_states):
            states[L].append(layer.detach().numpy()[0, len_prefix:])

    for L, layer in enumerate(states):
        layer = np.concatenate(layer, axis=0)
        layer = M @ layer
        layer = pd.DataFrame(layer, columns=['D%03d' % i for i in range(layer.shape[1])])
        layer['layer'] = L
        states[L] = layer

    return states


if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''Dump transcriptions of MRH auditory stimuli to CSV's with words and metadata''')
    args = argparser.parse_args()

    for _tg_path in os.listdir(TG_DIR):
        name = _tg_path[:-9]
        _tg_path = os.path.join(tg_path, _tg_path)
        stderr('Processing story: %s...\n' % name)
        with open(_tg_path, 'r') as f:
            tg = f.read()
        tg = TextGrid(tg)
        word_tier = None
        for tier in tg.tiers:
            if tier.nameid == 'word':
                word_tier = tier
                break
        assert word_tier is not None, 'No word tier found for file %s' % _tg_path
        transcript = word_tier.simple_transcript
        # Transcript is list of <onset, offset, word> tuples
        words = [x[2] for x in transcript]
        times = [x[0] for x in transcript]

        meta = get_table(words, times)
        meta['docid'] = name

        states = get_gpt2_states(meta)
        
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        for L, state in enumerate(states):
            out = pd.concat([meta, state], axis=1)
            out.to_csv(os.path.join(OUT_DIR, name + '_L%02d.csv' % L), sep=' ', index=False)

