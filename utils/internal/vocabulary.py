import json

vocab = json.load( open("datasets/vocab.json", "r") )
UNK_token = 15
SOS_token = 16
EOS_token = 17

def word_to_index(token):
  return vocab.index(token)

def index_to_word(idx):
  return vocab[idx]