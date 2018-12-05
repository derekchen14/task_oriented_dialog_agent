import json
import numpy as np

def load_vocab(path):
  with open("datasets/" + path, "r") as f:
    vocab = json.load(f)
  print("datasets/{} file loaded!".format(path))
  return vocab

vocabs = load_vocab("dstc2/slots/per_slot_vocab.json")
vocabs["dstc2"] = load_vocab("dstc2/cleaned/vocab.json")
categories = ["intent", "food", "area", "price", "name", "answer", "request"]

def belief_to_index(belief):
  high, low, slot, value = belief
  indexes = np.zeros((7, 1))
  indexes[0] = vocabs["intent"].index(high)
  if slot == "this" and value == "any":
    high, value = "answer", "this_any"

  if slot == "food":
    indexes[1] = vocabs[slot].index(value)
  elif slot == "area":
    indexes[2] = vocabs[slot].index(value)
  elif slot == "price":
    indexes[3] = vocabs[slot].index(value)
  elif slot == "name":
    indexes[4] = vocabs[slot].index(value)
  elif high == "answer":
    indexes[5] = vocabs[high].index(value)
  elif high == "request":
    indexes[6] = vocabs[high].index(value)
  return indexes

def cat_index(category):
  return categories.index(category)

def word_to_index(token):
  return vocabs["dstc2"].index(token)

def index_to_word(idx, kind):
  return vocabs[kind][idx]

def ulary_size(task):
  return len(vocabs["dstc2"])

def label_size(kind):
  return len(vocabs[kind])
