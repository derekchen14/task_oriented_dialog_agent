import json
from torch import eye
import pdb

def load_vocab(path):
  with open("datasets/" + path, "r") as f:
    vocab = json.load(f)
  print("datasets/{} file loaded!".format(path))
  return vocab

# dstc2_vocab = load_vocab("dstc2/cleaned/vocab.json")
woz2_vocab = load_vocab("woz2/vocab/vocabulary.json")
dual_vocab = load_vocab("woz2/vocab/dual.json")
# has keys slot and value, no "s"

# Task independent since car dataset special tokens already replaced
def word_to_index(token, task):
  return woz2_vocab.index(token)

def belief_to_index(belief, kind):
  intent, slot, value = belief
  if kind == "slot":
    return dual_vocab[kind].index(slot)
  elif kind == "value":
    return dual_vocab[kind].index(value)

  # high, low, slot, value = belief
  # intent_idx = intent_vocab.index(high)
  # if slot is None:
  #   if value is None:
  #     sv = "None"
  #   else:
  #     sv = value
  # else:
  #   sv = "{}={}".format(slot, value)

  # sv_idx = sv_vocab.index(sv)
  # return [intent_idx, sv_idx]

def index_to_word(idx, kind):
  if kind == "woz2":
    return woz2_vocab[idx]
  else:
    return dual_vocab[kind][idx]
  # elif kind == "intent":
  #   return intent_vocab[idx]
  # elif kind == "sv":
  #   return sv_vocab[idx]
  # elif kind == "dstc2":
  #   return dstc2_vocab[idx]

def ulary_size(task):
  return len(woz2_vocab)

def label_size(kind):
  return len(dual_vocab[kind])

