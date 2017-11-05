import json

vocab = json.load( open("datasets/vocab.json", "r") )

def word_to_index(token):
  return vocab.index(token) + 1

def index_to_word(idx):
  return vocab[idx - 1]