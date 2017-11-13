import json

vocab = json.load( open("datasets/vocab.json", "r") )
UNK_token = 14
SOS_token = 15
EOS_token = 16

def word_to_index(token):
  return vocab.index(token) + 1

def index_to_word(idx):
  return vocab[idx - 1]