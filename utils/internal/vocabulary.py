import json
from torch import eye
import pdb

def load_vocab(path):
  with open("datasets/" + path, "r") as f:
    vocab = json.load(f)
  print("datasets/{} file loaded!".format(path))
  return vocab

# car_vocab = load_vocab("car_vocab.json")
# babi_vocab = load_vocab("babi_vocab.json")
# dstc2_vocab = load_vocab("dstc2/cleaned/vocab.json")
woz2_vocab = load_vocab("woz2/vocab/vocabulary.json")
# embeddings = load_vocab("woz2/vocab/embeddings.json")
# frames_vocab = load_vocab("frames/cleaned/vocab.json")

# fe_vocab = load_vocab("woz2/vocab/full_enumeration.json")
# po_vocab = load_vocab("woz2/vocab/possible_only.json")
ov_vocab = load_vocab("woz2/vocab/ordered_values.json")
label_vocab = ov_vocab

# special_tokens = ["<SILENCE>", "<T01>","<T02>","<T03>", ... , "<T14>",
#           "UNK", "SOS", "EOS", "api_call","poi", "addr"]
UNK_token = 15
SOS_token = 16
EOS_token = 17
PHONE_token = 19
POI_token = 19
ADDR_token = 20

# Task independent since car dataset special tokens already replaced
def word_to_index(token, task):
  if task == "in-car":
    return car_vocab.index(token)
  elif task == "babi":
    if token.startswith("resto"):
      if "phone" in token:
        return PHONE_token
      elif "address" in token:
        return ADDR_token
    if task == "res" and token == 'cantonese':
      return 259 # CHINESE_token
    # if none of the special events occur ...
    return res_vocab.index(token)
  elif task == "dstc2":
    return dstc2_vocab.index(token)
  elif task == "woz2":
    return woz2_vocab.index(token)

def belief_to_index(belief):
  intent, slot, value = belief
  token = "{}={}".format(slot, value)
  return label_vocab.index(token)

def beliefs_to_index(beliefs):
  intents = ["{}={}".format(slot, value) for _, slot, value in beliefs]
  return [label_vocab.index(x) for x in intents]

def index_to_word(idx, task):
  if task == "woz2":
    return woz2_vocab[idx]
  elif task == "in-car":
    return car_vocab[idx]
  elif task == "babi":
    return res_vocab[idx]
  elif task == "dstc2":
    return dstc2_vocab[idx]
  elif task == "label":
    return label_vocab[idx]

def ulary_size(task):
  if task == "woz2":
    return len(woz2_vocab)
  elif task == "in-car":
    return len(car_vocab)
  elif task == "babi":
    return len(res_vocab)
  elif task == "dstc2":
    return len(dstc2_vocab)

def label_size():
  return len(label_vocab)