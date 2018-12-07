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
fe_vocab = load_vocab("woz2/vocab/full_enumeration.json")
po_vocab = load_vocab("woz2/vocab/possible_only.json")
ov_vocab = load_vocab("woz2/vocab/ordered_values.json")
# frames_vocab = load_vocab("frames/cleaned/vocab.json")

label_vocab = {
  "full_enumeration": fe_vocab,
  "possible_only": po_vocab,
  "ordered_values": ov_vocab
}

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

def index_to_word(idx, task):
  if task == "woz2":
    return woz2_vocab[idx]
  elif task == "in-car":
    return car_vocab[idx]
  elif task == "babi":
    return res_vocab[idx]
  elif task == "dstc2":
    return dstc2_vocab[idx]

def belief_to_index(belief, task):
  intent, slot, value = belief
  token = "{}={}".format(slot, value)
  return label_vocab[task].index(token)
  # high, low, slot, value = belief

  # if high in ["inform", "request", "answer", "question"]:
  #   if slot is None:
  #     token = "{0}({1})".format(high, value)
  #   else:
  #     token = "{0}({1}={2})".format(high, slot, value)
  # else:
  #   token = high
  # return labels.index(token)

def beliefs_to_index(beliefs, kind):
  labels = label_vocab[kind]

  intents = ["{}={}".format(slot, value) for _, slot, value in beliefs]
  if kind == "full_enumeration":
    # intents.sort()
    joined = "+".join(intents)
    return labels.index(joined)
  else:
    return [labels.index(x) for x in intents]

def index_to_word(idx, task):
  if task == "woz2":
    return woz2_vocab[idx]
  elif task == "in-car":
    return car_vocab[idx]
  elif task == "babi":
    return res_vocab[idx]
  elif task == "dstc2":
    return dstc2_vocab[idx]
  else:
    return label_vocab[task][idx]

def ulary_size(task):
  if task == "woz2":
    return len(woz2_vocab)
  elif task == "in-car":
    return len(car_vocab)
  elif task == "babi":
    return len(res_vocab)
  elif task == "dstc2":
    return len(dstc2_vocab)

def label_size(kind):
  return len(label_vocab[kind])