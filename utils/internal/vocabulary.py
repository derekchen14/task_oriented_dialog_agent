import json
from torch import eye
from torch.autograd import Variable

car_vocab = json.load(open("datasets/car_vocab.json", "r") )
res_vocab = json.load( open("datasets/res_vocab.json", "r") )
match_embeddings = Variable(eye(8))

UNK_token = 15
SOS_token = 16
EOS_token = 17
PHONE_token = 19
POI_token = 19
ADDR_token = 20

# Task independent since car dataset special tokens already replaced
def word_to_index(token, task):
  if task == "car":
    return car_vocab.index(token)
  else:
    if token.startswith("resto"):
      if "phone" in token:
        return PHONE_token
      elif "address" in token:
        return ADDR_token
    if task == "res" and token == 'cantonese':
      return 259 # CHINESE_token
  # if none of the special events occur ...
  return res_vocab.index(token)

def index_to_word(idx, task):
  if task == "car":
    return car_vocab[idx]
  else:
    return res_vocab[idx]

def ulary_size(task):
  if task == "car":
    return len(car_vocab)
  else:
    return len(res_vocab)
