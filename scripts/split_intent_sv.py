import json
import pdb

with open("label_vocab.json", "r") as f:
  vocab = json.load(f)

  intent = set()
  sv = set()
  for row in vocab["ordered_values"]:
    isv = row.split("(")
    intent.add(isv[0])
    if len(isv) > 1:
      sv.add(isv[1][:-1])  # everything except closing ")" symbol
    else:
      sv.add("None")

  json.dump(list(intent), open("intent_vocab.json", "w") )
  json.dump(list(sv), open("sv_vocab.json", "w") )




