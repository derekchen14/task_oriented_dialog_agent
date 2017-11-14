import json

#TODO: turn into a class
vocab = json.load( open("datasets/vocab.json", "r") )
UNK_token = 15
SOS_token = 16
EOS_token = 17

def word_to_index(token):
  return vocab.index(token)

def index_to_word(idx):
  return vocab[idx]

# @staticmethod
def ulary_size():
  return len(vocab)

# @classmethod
# def load_vocab(cls, path)
#   cls(path)