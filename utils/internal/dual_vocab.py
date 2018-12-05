import json
from torch import eye
import pdb

def load_vocab(path):
  with open("datasets/" + path, "r") as f:
    vocab = json.load(f)
  print("datasets/{} file loaded!".format(path))
  return vocab

dstc2_vocab = load_vocab("dstc2/cleaned/vocab.json")
intent_vocab = load_vocab("dstc2/slots/intent_vocab.json")
sv_vocab = load_vocab("dstc2/slots/sv_vocab.json")

# Task independent since car dataset special tokens already replaced
def word_to_index(token, task):
  return dstc2_vocab.index(token)

def belief_to_index(belief):
  high, low, slot, value = belief
  intent_idx = intent_vocab.index(high)
  if slot is None:
    if value is None:
      sv = "None"
    else:
      sv = value
  else:
    sv = "{}={}".format(slot, value)

  sv_idx = sv_vocab.index(sv)
  return [intent_idx, sv_idx]

def index_to_word(idx, kind):
  if kind == "intent":
    return intent_vocab[idx]
  elif kind == "sv":
    return sv_vocab[idx]
  elif kind == "dstc2":
    return dstc2_vocab[idx]

def ulary_size(task):
  return len(dstc2_vocab)

def label_size(kind):
  if kind == "intent":
    return len(intent_vocab)
  elif kind == "sv":
    return len(sv_vocab)
  else:
    return len(label_vocab[kind])

