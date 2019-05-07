import json
import pickle as pkl
import torch
import os, pdb, sys
import numpy as np
import re
from random import seed, shuffle
from tqdm import tqdm as progress_bar

from objects.models import NLU
from objects.models.external import LSTM_model
from utils.internal.arguments import solicit_args
from operators.preprocess import DataLoader

tags = ['actor', 'actress', 'city', 'closing', 'critic_rating', 'date', 'description', 'distanceconstraints', 'genre', 'greeting', 'moviename', 'mpaa_rating', 'numberofpeople', 'other', 'starttime', 'state', 'theater', 'theater_chain', 'video_format', 'zip', 'price']
num_mapper = {'1': 'one','2': 'two','3': 'three','4': 'four',
    '5':'five', '6':'six', '7':'seven', '8': 'eight', '9': 'nine'}


def label_tags(utterance, seq_len, tag, value, module, actual):
  utt = utterance.lower().split()
  tokens = [x.strip('.').strip('?').strip(',').strip('!') for x in utt]
  raw_words = [y.strip(')').strip('!').strip('.') for y in value.lower().split()]

  words = []
  for word in raw_words:
    if '||' in word:
      word = word[1:-1]
      choices = word.split('||')
      words.extend(choices)
    elif word in num_mapper.keys():
      number = word if word in tokens else num_mapper[word]
      words.append(number)
    elif tag == 'starttime':
      if word in tokens:
        words.append(word)
      elif re.search(r'\dpm$', word):
        words.append(word[:-2])
        words.append(word[-2:])
      else:
        words.append(word)
    else:
      words.append(word)

  if len(words) == 0:
    pass
  elif len(words) == 1:
    try:
      position = tokens.index(words[0]) + 1
    except(ValueError):
      return actual
    label = f'B-{tag}'
    tag_idx = module.tag_set[label]
    actual[position, tag_idx] = 1.0
  else:
    for j, word in enumerate(words):
      try:
        position = tokens.index(word) + 1
      except(ValueError):
        return actual

      label = f'B-{tag}' if j == 0 or tag == 'price' else f'I-{tag}'
      try:
        tag_idx = module.tag_set[label]
      except(KeyError):
        continue
      actual[position, tag_idx] = 1.0

  return actual

def extract_slots(internal):
  if len(internal) == 0:
    return {}
  else:
    slot_vals = internal.split(';')
    slots = {}
    for x in slot_vals:
      if x in ['ticket', 'moviename', 'starttime', 'city', 'state', 'date', 'other', 'theater']:
        slot = x
        val = ''
      elif '=' in x:
        slot, val = x.split('=')
      else:
        continue
      slots[slot] = val
    return slots

def extract_intent(candidate):
  try:
    act, internal = candidate.strip(')').split('(')
  except(ValueError):
    act = 'unknown'
    internal = ''

  slot_vals = extract_slots(internal)
  slots = list(slot_vals.keys())
  values = slot_vals
  extracted = []

  if act == 'multiple_choice' or act == 'confirm_question':
    slots.insert(0, act)
    intent = '+'.join(slots)
    extracted.append(intent)
  elif act == 'request':
    if len(slots) < 1:
      pass
    elif len(slots) > 2:
      for slot in slots:
        intent = f'request+{slot}'
        extracted.append(intent)
    else:
      slots.insert(0, act)
      intent = '+'.join(slots)
      extracted.append(intent)
  elif act == 'inform':
    for slot in slots:
      intent = f'{act}+{slot}'
      extracted.append(intent)
  else:
    if len(slots) == 0:
      extracted.append(act)
    elif act == 'thanks' and slots == ['closing']:
      print("used this")
      extracted.append(act)
    else:
      slots.insert(0, act)
      intent = '+'.join(slots)
      extracted.append(intent)

  return extracted, values


if __name__ == "__main__":
  args = solicit_args()
  torch.manual_seed(args.seed)
  seed(args.seed)

  loader = DataLoader(args)
  model_path = args.prefix + args.model + args.suffix
  module = NLU(loader, 'results/manage_policy/ddq/movies/')
  module.load_nlu_model('nlu_1468447442')
  # module = NLU(loader, 'results/track_intent/e2e/movies/')
  # module.load_nlu_model(model_path)

  row_counter = 0
  data = []
  data_path = os.path.join('datasets', args.dataset, 'movie_all.tsv')
  with open(data_path, 'r') as file:
    for line in file:
      row_counter += 1
      if row_counter == 1: continue

      items = line.rstrip('\n').split('\t')
      if items[3] == 'agent': continue
      utterance = items[4]
      seq_len = len(utterance.split()) + 2
      seq_dim = len(module.tag_set)
      actual = np.zeros((seq_len, seq_dim))

      for num in [5,6,7,8]:
        if items[num] != '':
          extracted, slot_values = extract_intent(items[num])
          for intent in extracted:
            try:
              tag_idx = module.tag_set[intent]
              actual[-1, tag_idx] = 1.0
            except(KeyError):
              pass
          for tag, value in slot_values.items():
            if tag in tags:
              actual = label_tags(utterance, seq_len, tag, value, module, actual)
      data.append((utterance, actual))

  shuffle(data)
  end_idx = int(len(data) * 0.7)
  train = data[:end_idx]
  # pkl.dump(train, open('temp_train.pkl', 'wb'))
  print("Finished generating data")

  correct, monitor = 0, 0
  RMSprop = {weight_type: 0 for weight_type in module.model.update}
  for epoch in range(args.epochs):
    for example in train:

      utterance, actual = example
      stripped = utterance.strip('.').strip('?').strip(',').strip('!')
      rep = module.embed_to_one_hot(stripped)

      if args.test_mode:
        preds, cache = module.model.fwdPass(rep, module.params, predict_model=True)
      else:
        outputs, cache = module.model.fwdPass(rep, module.params, predict_model=False)

        # manual softmax
        maxes = np.amax(outputs, axis=1, keepdims=True)
        e = np.exp(outputs - maxes) # for numerical stability shift into good numerical range
        preds = e/np.sum(e, axis=1, keepdims=True)
        if np.all(np.isnan(preds)): preds = np.zeros(preds.shape)

        try:
          # pred_copy = np.zeros_like(preds)
          # pred_copy[-1] = preds[-1]
          # diff = pred_copy - actual
          diff = preds.copy() - actual
        except(ValueError):
          continue
        # {'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd}
        weights = module.model.bwdPass(diff, cache)

        for wt in module.model.update:
          weight_grad = weights[wt]
          RMSprop[wt] = (0.9 * RMSprop[wt]) + 0.1 * (weight_grad**2)
          delta = (args.learning_rate/np.sqrt(RMSprop[wt] + 1e-8)) * weight_grad
          module.model.model[wt] -= delta  # args.learning_rate * weight_grad

      preds_per_seq = np.nanargmax(preds, axis=1)
      pred_index = preds_per_seq[-1]
      if actual[-1, pred_index] == 1:
        correct += 1
      monitor += 1

      if monitor == 1000:
        print(f"{epoch+1}) Got {correct} correct")
        correct, monitor = 0, 0

  if args.save_model:
    save_dir = os.path.join("results", args.task, args.dataset, model_path)
    module.save_nlu_model(save_dir)

# python train_old_nlu.py --task track_intent --epochs 5 --dataset e2e/movies \
#   --prefix old_nlu_ --model lstm --suffix _14 --seed 14 --save-model \
#   --learning-rate 0.00003
