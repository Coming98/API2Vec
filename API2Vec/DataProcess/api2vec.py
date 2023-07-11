#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   api2vec.py
@Desc         :   api2vec
'''
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import random
import utils
import torch
import numpy as np
from tqdm import tqdm

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

def get_embedding_model(config, corpus, idx2api):
    if(config.embedding_model_name == 'doc2vec'):
        if(config.pid_flag == 'pid' or config.pid_flag == 'node2vec'):
            corpus = [item for path in corpus for item in path]
            documents = [TaggedDocument(path, [i]) for i, path in enumerate(corpus)]
        else:
            documents = [TaggedDocument(path, [i]) for i, path in enumerate(corpus)]
        model = Doc2Vec(documents, vector_size=config.vector_size, window=config.window, min_count=1, workers=config.workers, epochs=config.epochs)
    else:
        if(config.pid_flag == 'pid' or config.pid_flag == 'node2vec'):
            corpus = [item for path in corpus for item in path]

    if(config.embedding_model_name == 'w2v'):
        model = Word2Vec(sentences=corpus, vector_size=config.vector_size, sg=config.sg, window=config.window, 
                    epochs=config.epochs, min_count=config.min_count, workers=config.workers)

    return model



def api2vec(config, corpus, labels, datanames, idx2api):
    model = get_embedding_model(config, corpus, idx2api)
    return model, None