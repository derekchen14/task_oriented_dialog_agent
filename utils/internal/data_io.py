'''
Ref: https://github.com/llSourcell/How_to_make_a_chatbot/blob/master/memorynetwork.py
Most of them are helper functions. You may only read and use get_word(), read_from_file(), and
load_glove()
'''
import random
import numpy as np
from scipy.stats import truncnorm
import math

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return sent.split()


def parse_dialogue(lines, only_supporting=False):
    '''
    lines: f.readline(), which is actually a list of lines in txt
    Return [[(u1, r1), (u2, r2)...], [(u1, r1), (u2, r2)...], ...]
    '''
    data = []
    dialogue = []
    for line in lines:
        if line != '\n' and line != lines[-1]:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            line = line.decode('utf-8').strip()
            q, a = line.split('\t')
            q = tokenize(q)
            a = tokenize(a)
            dialogue.append((q, a))
        else:
            data.append(dialogue)
            dialogue = []
    return data


def word_to_glove_vector(glove, word):
    '''
    :param embeddinds: Glove object from pytorchtext
    :param word: str
    :return: the embedding vector of the word
    '''
    return glove.vectors[glove.stoi[word]]


def read_restuarant_data(filename):
    '''
    :param filename: 'directory/file.txt'
    :return:[
        [(u1, r1), (u2, r2)...]
        , [(u1, r1), (u2, r2)...], ...]
    the data is a list of training examples
    each example consists of one dialog
    for each dialog, there are a number of turns
    each turn is made up of a tuple of (u_i, r_i) for up to N turns
        where ui is utterance from the customer
        where ri is a response from an agent
    each ui or ri, is a list of strings, for up to M tokens
        each token is usually a word or punctuation
    if the customer said nothing during their turn,
        special token of <SILENCE> is used
    '''
    with open(filename) as f:
        # max_length = None
        data = parse_dialogue(f.readlines(), only_supporting=False)
    return data

def read_car_data(filename):
    pass


def init_glove_words(name='6B', dim=100):
    '''
    :param name: which glove you want to load
    :param dim: dimension of word vectors
    :return: the glove object in pytorch
    '''
    import torchtext.vocab as vocab
    glove = vocab.GloVe(name='6B', dim=100)
    print('Loaded {} words'.format(len(glove.itos)))
    word_embeddings = glove
    return word_embeddings

def init_normal_words(vocab_size=1229, dim=100):
    # add EOS and SOS and UNK?
    mean = 0
    stddev = 1.0/math.sqrt(2 * math.pi)
    # each word embedding is a column vector
    word_embeddings = truncnorm.rvs(a=mean, b=stddev, size=[dim, vocab_size])
    return word_embeddings





