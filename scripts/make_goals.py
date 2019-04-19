import json
import numpy as np
import re, copy
import pdb
import string
from collections import defaultdict

data = json.load(open("goal_set_v2.json", "r"))
ontology = json.load(open("ontology.json", "r"))
valid_slots = list(ontology["slot_values"].keys())
valid_values = ontology["slot_values"]

# samples = np.random.choice(data, size=200, replace=False)
new_values = defaultdict(set)

def check_existing(slot, value):
  if value not in valid_values[slot]:
    new_values[slot].add(value)

def scrub_theater(candidate):
  special_characters = ['-', ':', ',', '.'] + [str(d) for d in range(10)]
  index = -1
  for idx, char in enumerate(candidate):
    if char in special_characters:
      index = idx
      break

  if index > 0:
    candidate = candidate[:index]
    candidate = candidate.rstrip()
  return candidate

old_goals = data
new_goals = []
for goal in old_goals:
  entry = {}
  for slot, value in sample.items():  # ['inform_slots']
    value = value.lower().rstrip()
    if slot == 'theater':
      cleaned = scrub_theater(value)
      check_existing('theater', cleaned)
    elif slot in valid_slots:
      entry[slot] = value
      check_existing(slot, value)
    elif slot == 'starttime_real':
      inserted = value[:-2] + " " + value[-2:]
      entry['starttime'] = inserted
      check_existing('starttime', inserted)
    elif slot == 'state_real':
      entry['state'] = value
      check_existing('state', value)
  new_goals.append(entry)

print(f"We created {len(new_goals)} entries!")
new_data = {
  "all": goals,
  "train": goals[:140],
  "dev": goals[140:180],
  "test": goals[180:],
}

# for slot, values in new_values.items():
#   list_values = list(values)
#   if slot in ['starttime', 'theater']:
#     ontology['slot_values'][slot].extend(list_values)

json.dump(new_goals, open("goals_v2.json", "w"))
# json.dump(ontology, open("ontology_v2.json", "w"))
