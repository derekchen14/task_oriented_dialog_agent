'''
Ref: https://github.com/llSourcell/How_to_make_a_chatbot/blob/master/memorynetwork.py
Most of them are helper functions. You may only read and use get_word(), read_from_file(), and
load_glove()
'''
import torchtext.vocab as vocab

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


def get_dialogue(f, only_supporting=False, max_length=None):
    '''
    f: open(file) object
    returns [[(u1, r1), (u2, r2)...], [(u1, r1), (u2, r2)...], ...]
    '''
    data = parse_dialogue(f.readlines(), only_supporting=only_supporting)
    return data


def get_word(glove, word):
    '''
    :param glove: Glove object from pytorchtext
    :param word: str
    :return: the embedding vector of the word
    '''
    return glove.vectors[glove.stoi[word]]


def read_from_file(filename):
    '''
    :param filename: 'directory/file.txt'
    :return:[[(u1, r1), (u2, r2)...], [(u1, r1), (u2, r2)...], ...]
    '''
    f = open(filename)
    data = get_dialogue(f)
    return data

def load_glove(name='6B', dim=100):
    '''
    :param name: which glove you want to load
    :param dim: dimension of word vectors
    :return: the glove object in pytorch
    '''
    glove = vocab.GloVe(name='6B', dim=100)
    print('Loaded {} words'.format(len(glove.itos)))
    return glove


