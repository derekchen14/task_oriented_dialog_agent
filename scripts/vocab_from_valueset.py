import pickle

vocab = set()

value_set = pickle.load( open("value_set.p", "rb"), encoding="latin1")
for slot, values in value_set.items():
  vocab.add(slot)
  for val in values:
    val = val.replace(".", " ")
    val = val.replace(",", " ")
    val = val.replace("!", " ")
    val = val.replace("#", " ")
    val = val.replace("=", " ")
    val = val.replace("}", " ")
    val = val.replace("{", " ")
    for token in val.lower().split():
      if len(token) <= 14:
        vocab.add(token)

processed = list(vocab)
print(len(processed))

print(processed[10:14])
print(processed[40:50])
print(processed[140:150])

import pdb
pdb.set_trace()

import json

json.dump(processed, open("vocab.json", "w"))
