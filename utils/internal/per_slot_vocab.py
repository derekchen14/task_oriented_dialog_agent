import json
import numpy as np

def load_vocab(path):
  with open("datasets/" + path, "r") as f:
    vocab = json.load(f)
  print("datasets/{} file loaded!".format(path))
  return vocab

vocabs = load_vocab("woz2/vocab/per_slot.json")
vocabs["woz2"] = load_vocab("woz2/vocab/vocabulary.json")
categories = ["area", "food", "price", "request"]

def belief_to_index(belief):
  intent, slot, value = belief
  indexes = np.zeros((4, 1))

  if slot == "area":
    indexes[0] = vocabs[slot].index(value)
  elif slot == "food":
    indexes[1] = vocabs[slot].index(value)
  elif slot == "price":
    indexes[2] = vocabs[slot].index(value)
  elif intent == "request":
    indexes[3] = vocabs[intent].index(value)
  return indexes

def cat_index(category):
  return categories.index(category)

def word_to_index(token):
  return vocabs["woz2"].index(token)

def index_to_word(idx, kind):
  return vocabs[kind][idx]

def ulary_size(task):
  return len(vocabs["woz2"])

def label_size(kind):
  return len(vocabs[kind])







# indexes[0] = vocabs["intent"].index(high)
# if slot == "this" and value == "any":
#   high, value = "answer", "this_any"
