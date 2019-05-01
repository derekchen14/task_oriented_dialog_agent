import json
import pickle as pkl
import numpy as np
import re, copy
import sys, pdb
import string, random

import editdistance
from collections import defaultdict, Counter
from nltk import word_tokenize
import time as tm

"""
___CONVERSION PROCESS___
1) finalize ontology
2) clean kb and goals to fit in ontology
3) clean data splits (use cache) - train, val, test
4) use this to generate vocabulary (word2index)
5) use vocabulary to finalize vectorized embeddings
"""

ontology = json.load(open("ontology_v3.json", "r"))
kb = json.load(open("kb_v2.json", "r"))
goals = json.load(open("goals_v3.json", "r"))
data = json.load(open("data_v2.json", "r"))

valid_acts = ontology["dialogue_acts"]
valid_slots = ontology["slots"]
valid_values = ontology["values"]
valid_values["task"] = ["complete"]

cache = {'la': 'los angeles', 'california': 'ca', '': 'any',
  'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
  'six': '6', 'seven': '7', 'eight': '8', 'nine': '9' }
vocabulary = Counter()

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

def find_closest(candidates, missing):
  # find the value in candidates that is closest to the missing value
  scores = []

  for value in candidates:
    score = 0
    # 2 and 13 are the medians of large number of samples
    score += lcs(missing, value) / 2.0
    score -= editdistance.eval(missing, value) / 13.0
    scores.append(score)

  index_of_closest = np.argmax(scores)
  return candidates[index_of_closest]

def lcs(x, y):
  grid = np.zeros((len(x), len(y)))

  for i, letter in enumerate(reversed(x)):
    for j, symbol in enumerate(y):
      previous = grid[i-1, j+1] if (i > 0) and (j < len(y) - 1) else 0
      score = previous+1 if letter == symbol else 0
      grid[i,j] = score

  return np.max(grid)

class Vocab(object):
  def __init__(self, vocabulary):
    vocabulary['='] += 140
    vocabulary['task'] += 140
    vocabulary['complete'] += 140

    self.vocabulary = vocabulary
    self.counts = dict(vocabulary)
    self.index2word = list(vocabulary)

    self.counts['<sos>'] = len(data)
    self.counts['<eos>'] = len(data)
    self.index2word.insert(0, '<sos>')
    self.index2word.insert(1, '<eos>')

    for num in [2, 3, 4]:
      token = '<special' + str(num) + '>'
      self.counts[token] = 100
      self.index2word.insert(num, token)
    self.word2index = {word: idx for idx, word in enumerate(self.index2word)}

    self.unknowns = []

    self.slot_mapper = {
      'moviename': 'movie name',
      'ticket': 'ticket',
      'theater': 'theater',
      'genre': 'genre',
      'date': 'date',
      'restaurantname': 'restaurant name',
      'reservation': 'reservation',
      'atmosphere': 'atmosphere',
      'rating': 'rating',
      'city': 'city',
      'state': 'state',
      'starttime': 'start time',
      'numberofpeople': 'number of people',
      'price': 'price',
      'food': 'food',
      'car_type': 'car type',
      'pickup_time': 'pickup time',
      'pickup_city': 'pickup city',
      'dropoff_city': 'dropoff city',
      'pickup_location': 'pickup location',
      'dropoff_location': 'dropoff location',
      'ride': 'ride',
      'task': 'task',
      'other': 'other',
      'unknown': 'unknown',
      '<special2>': '<special2>',
    }
    self.act_mapper = {
      'open': 'open = hi hello',
      'close': 'close = goodbye thanks',
      'accept': 'accept = yes ok',
      'reject': 'reject = no not',
      'question': 'question = how what',
      'answer': 'answer = sure can',
      'unknown': 'unknown = other',
      'multiple': 'multiple = available or',
    }

  def to_dict(self):
    results = {}
    results["counts"] = self.counts
    results["index2word"] = self.index2word
    results["word2index"] = self.word2index
    return results

  def vectorize(self, text, style='intent'):
    embedding = []
    embedding.append(self.word2index['<sos>'])

    if style == 'intent':
      x, y = text
      if x == "request":
        slot = self.slot_mapper[y]
        text = f'{x} = {slot}'
      elif x == "act":
        text = self.act_mapper[y]
      else:  # its some type of inform
        slot, value = self.slot_mapper[x], y
        text = f'{slot} = {value}'

    words = text.replace(':', ' ').lower()
    if style == 'intent':
      tokens = words.split()
    elif style == 'utterance':
      tokens = word_tokenize(words)

    for token in tokens:
      try:
        embedding.append(self.word2index[token])
      except(KeyError):
        self.unknowns.append(token)
        embedding.append(self.word2index['<unk>'])

    embedding.append(self.word2index['<eos>'])
    return embedding


before = tm.time()

# Extract vocabulary from ontology
for act in valid_acts:
  vocabulary[act] += 1
for slot in valid_slots:
  vocabulary[slot] += 1
  for val in valid_values[slot]:
    val = val.replace(':', ' ')
    # we use split() here because ontology is cleared of punctuation
    tokens = val.split()
    for token in tokens:
      vocabulary[token.lower()] += 1

# Create the knowledge base
new_kb = []
for entry in kb:
  new_entry = {}
  for slot, old_value in entry.items():
    candidates = valid_values[slot]
    if old_value not in candidates:
      if old_value in cache.keys():
        new_value = cache[old_value]
      else:
        new_value = find_closest(candidates, old_value)
        cache[old_value] = new_value
      new_entry[slot] = new_value
    else:
      new_entry[slot] = old_value
  new_kb.append(new_entry)

json.dump(new_kb, open("trial/kb.json", "w"))
time_past = round(tm.time() - before, 2)
print(f"KB done: {time_past} seconds")

# Build all the goals
goal_informs = np.random.choice(new_kb, size=32, replace=False)
for goal_inf in goal_informs:
  example = {'inform_slots': goal_inf, 'request_slots': {}, 'dialogue_act': 'inform'}
  goals['val'].append(example)
combined = { "train": [], "val": goals['val'], "test": []}

for split in ['train', 'test']:
  new_goals = []
  for goal in goals[split]:
    new_goal = goal.copy()
    for slot, old_value in goal['inform_slots'].items():
      candidates = valid_values[slot]
      if old_value not in candidates:
        if old_value in cache.keys():
          new_value = cache[old_value]
        else:
          new_value = find_closest(candidates, old_value)
          cache[old_value] = new_value
        new_goal['inform_slots'][slot] = new_value
    new_goals.append(new_goal)
  combined[split] = new_goals

combined['all'] = combined['train'] + combined['val'] + combined['test']
json.dump(combined, open("trial/goals.json", "w"))
time_past = round(tm.time() - before, 2)
print(f"Goals done: {time_past} seconds")

for split, split_data in data.items():
  new_split = []
  for example in split_data['dialogues']:
    new_exp = {'dialogue_id': example['dialogue_id'], 'turns': []}
    for turn in example["turns"]:
      new_turn = turn.copy()

      for idx, aa in enumerate(turn['agent_actions']):
        if aa[0] == '<special2>':
          new_turn['agent_actions'][idx] = ['<special2>', '<special2>']
        else:
          aa_slot, aa_value = aa
          candidates = valid_values[aa_slot]
          if aa_value not in candidates:
            if aa_value in cache.keys():
              new_value = cache[aa_value]
            else:
              new_value = find_closest(candidates, aa_value)
              cache[aa_value] = new_value
          else:
            new_value = aa_value
          new_turn['agent_actions'][idx] = [aa_slot, new_value]

      for jdx, ui in enumerate(turn['user_intent']):
        ui_slot, ui_value = ui
        candidates = valid_values[ui_slot]
        if ui_value not in candidates:
          if ui_value in cache.keys():
            new_value = cache[ui_value]
          else:
            new_value = find_closest(candidates, ui_value)
            cache[ui_value] = new_value
        else:
          new_value = ui_value
        new_turn['user_intent'][jdx] = [ui_slot, new_value]

      for kdx, belief in enumerate(turn['belief_state']):
        new_belief = belief.copy()
        # new_belief is a dict, belief["slots"] is a list of lists
        assert len(belief["slots"]) == 1
        bs_slot, bs_value = belief["slots"][0]
        candidates = valid_values[bs_slot]
        if bs_value not in candidates:
          if bs_value in cache.keys():
            new_value = cache[bs_value]
          else:
            new_value = find_closest(candidates, bs_value)
            cache[bs_value] = new_value
        else:
          new_value = bs_value

        new_belief['slots'][0] = [bs_slot, new_value]
        new_turn['belief_state'][kdx] = new_belief

      # we use tokenize() because we are concerned with punctuation
      tokens = word_tokenize(turn['utterance'].replace(':', ' '))
      for token in tokens:
        vocabulary[token.lower()] += 1

      new_exp['turns'].append(new_turn)
    new_split.append(new_exp)

  time_past = round(tm.time() - before, 2)
  print(f"{split} done: {time_past} seconds")
  data[split] = new_split

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

removed = []
for token, count in vocabulary.items():
  if count == 1:
    if len(token) > 12 or len(token) == 1:
      removed.append(token)
for remove in removed:
  vocabulary.pop(remove)
  vocabulary["<unk>"] += 1

print("There are {} unique words in the vocabulary".format(len(vocabulary)))
print("The ten most common words are", vocabulary.most_common(10))
print("{} rare words have been turned into <unk>'s".format(vocabulary["<unk>"]))

vb = Vocab(vocabulary)
json.dump(vb.to_dict(), open("trial/vocab.json", "w"))

for slot, value_set in valid_values.items():
  empirical_values = sorted(list(value_set))
  numbers = [vb.vectorize([slot, val]) for val in value_set]
  ontology['vectorized'][slot] = numbers
  print("We found {} values in the {} slot".format(len(value_set), slot))

print("vectorized ontology")
json.dump(ontology, open(f"trial/ontology.json", "w"))

for split, split_data in data.items():
  final_data = []
  for example in split_data:
    new_example = example.copy()
    new_example["turns"] = []
    for turn in example["turns"]:
      turn['num'] = {
        "agent_actions": [vb.vectorize(aa) for aa in turn['agent_actions']],
        "utterance": vb.vectorize(turn['utterance'], 'utterance')
      }
      new_example["turns"].append(turn)
    final_data.append(new_example)
  json.dump(final_data, open(f"trial/{split}.json", "w"))


print("vectorized data")

"""
make sure that inputs being passed in to NLU and PM are lowered and stripped

utterance in NLU is (preprocessed and) correctly
   - lowercases
   - separates punctuation
   - splits the colons in starttimes
user intent label in NLU is also fed as one hot, so if the label does
  not exist in the ontology exactly, then code will break
  this is actually a great sanity check that your labels
  can be found precisely in the ontology!


user intents (as one hot) fed into PM are embeddings not natural language
  the PM never sees a colon anywhere in either the state, action or reward
the text generation is left untouched


"""



