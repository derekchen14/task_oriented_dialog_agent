import json

#TODO: turn into a class
vocab = json.load( open("datasets/res_vocab.json", "r") )
# vocab = json.load( open("datasets/car_vocab.json", "r") )
UNK_token = 15
SOS_token = 16
EOS_token = 17
PHONE_token = 19
ADDR_token = 20

def word_to_index(token):
  if token.startswith("resto"):
    if "phone" in token:
      return PHONE_token
    elif "address" in token:
      return ADDR_token
    else:
      return vocab.index(token)
  else:
    return vocab.index(token)

def index_to_word(idx):
  return vocab[idx]

# @staticmethod
def ulary_size():
  return len(vocab)

# @classmethod
# def load_vocab(cls, path)
#   cls(path)