#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   api2vec.py
@Desc         :   api2vec
'''
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import os
import gc
import random
import utils

def generate_corpus(corpus):
    result_corpus = []
    for corpu in corpus:
        length = len(corpu)
        count = 5
        step = 3
        added_corpus = []
        added_indexes = random.sample(range(length - step), count)
        for index in added_indexes:
            aim_corpu = corpu[index:index+step]
            added_corpus.extend(aim_corpu * int(round(length / 10, 0)))
        result_corpus.append(corpu + added_corpus)
    return result_corpus

def to_api(path, idx2api):
    return [idx2api[idx] for idx in path]


def get_embedding_model(config, corpus, idx2api):

    if('doc2vec' in config.embedding_model_name):
        if(config.pid_flag == 'pid' or config.pid_flag == 'node2vec'):
            corpus = [to_api(path, idx2api) for paths in corpus for path in paths]
            corpus = [TaggedDocument(path, [i]) for i, path in enumerate(corpus)]
        else:
            corpus = [TaggedDocument(path, [i]) for i, path in enumerate(corpus)]
    else:
        if(config.pid_flag == 'pid' or config.pid_flag == 'node2vec'):
            corpus = [path for paths in corpus for path in paths]

    if('doc2vec' in config.embedding_model_name):
        model = Doc2Vec(corpus, vector_size=config.vector_size, window=config.window, min_count=1, workers=config.workers, epochs=config.epochs)
    elif(config.embedding_model_name == 'ft'):
        model = FastText(sentences=corpus, vector_size=config.vector_size, window=config.window, min_count=config.min_count, workers=config.workers)
    elif(config.embedding_model_name == 'w2v'):
        model = Word2Vec(sentences=corpus, vector_size=config.vector_size, sg=config.sg, window=config.window, 
                    epochs=config.epochs, min_count=config.min_count, workers=config.workers)

    return model



def api2vec(config, corpus, labels, datanames, idx2api):
    corpus, _, __ = utils.filter_time(config, corpus, labels, datanames)
    return get_embedding_model(config, corpus, idx2api)