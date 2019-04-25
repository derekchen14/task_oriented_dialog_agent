import json
import pickle as pkl
import numpy as np
import re, copy
import pdb
import string, random
from collections import defaultdict

data = json.load(open("goals.json", "r"))
ontology = json.load(open("ontology.json", "r"))
valid_slots = ontology["slots"]
valid_values = ontology["values"]

# valid_slots.remove('state')
# valid_slots.remove('starttime')

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

"""
old_kb = data
new_kb = []
for sample in old_kb:
  entry = {}
  for slot, value in sample.items():  # ['inform_slots']
    value = value.lower().rstrip()
    if slot == 'theater':
      cleaned = scrub_theater(value)
      entry[slot] = cleaned
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
  new_kb.append(entry)

print(f"We created {len(new_kb)} entries!")
"""

counter = 0

for goal in data['all']:
  # new_goal = goal.copy()
  for slot in valid_slots:
    for goal_slot, goal_attr in goal['inform_slots'].items():
      counter += 1
      if slot == goal_slot:
        value_attributes = valid_values[slot]
        if goal_attr not in value_attributes:
          # new_attr = random.choice(value_attributes)
          # new_goal['inform_slots'][slot] = new_attr
          new_values[slot].add(goal_attr)
  # train_goals.append(new_goal)

"""
all_goals = train_goals + test_goals

combined = {
  "all": all_goals,
  "train": train_goals,   # goals[:140],
  "dev": [],              # goals[140:180],
  "test": test_goals      # goals[180:],
}
"""

print(new_values)
print(counter)
# pdb.set_trace()
# json.dump(combined, open("goals.json", "w"))
# json.dump(new_kb, open("kb.json", "w"))
# json.dump(ontology, open("ontology_v2.json", "w"))
