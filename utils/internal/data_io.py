# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy.stats import truncnorm
import math
import json
import pandas as pd

from nltk import word_tokenize
import utils.internal.vocabulary as vocab

def load_dataset(task, split, debug=False):
  if task in ['1', '2', '3', '4', '5']:
    dataset_name = "dialog-babi-task{0}-{1}.txt".format(task, split)
    restaurants, kb = read_restaurant_data(dataset_name)
    candidates = read_restaurant_data("dialog-babi-candidates.txt")
    max_length = 22
    return (restaurants, candidates, max_length)
  elif task in ['schedule','navigate','weather']:
    paths = {'trn': 'in_car/train.json', 'dev': 'in_car/dev.json', 'tst': 'in_car/test.json'}
    data = load_json_dataset(paths[split])
    data = data[0:20] if debug else data
    navigations, weathers, schedules, kbs = load_incar_data(data)
    tasks = {'navigate': navigations, 'weather': weathers, 'schedule': schedules}
    max_length = 42
    return (tasks[task], kbs, max_length)
  elif task == "challenge":
    dataset_name = "dialog-babi-task6-{0}2-{1}.txt".format('dstc', split)
    restaurants, kb = read_restaurant_data(dataset_name)
    candidates = read_restaurant_data("dialog-dstc-candidates.txt")
    max_length = 30
    return (restaurants, candidates, max_length)
  elif task == "dstc2":
    dataset_name = "{0}/cleaned/{1}_v3.json".format(task, split)
    data = vocab.load_vocab(dataset_name)  # actually used to load json
    examples = data[0:20] if debug else data
    max_length = 23
    return (examples, max_length)


def parse_dialogue(lines, tokenizer=True):
  '''
  lines: f.readline(), which is actually a list of lines in txt
  Return [[(u1, r1), (u2, r2)...], [(u1, r1), (u2, r2)...], ...]
  '''
  data = []
  dialogue = []
  kb_dialogue = []
  kb = []
  for line in lines:
    if line != '\n' and line != lines[-1]:
      nid, line = line.split(' ', 1)
      nid = int(nid)
      line = line.strip()  # line.decode('utf-8').strip() for python 2.7

      if len(line.split('\t')) == 1:
        kb_dialogue.append(line.split('\t'))
        continue

      u, r = line.split('\t')
      if tokenizer is True:
        u = word_tokenize(u)
        r = word_tokenize(r)
      u = fix_silence(u) if len(u)>2 else u
      dialogue.append((u, r))
    else:
      data.append(dialogue)
      kb.append(kb_dialogue)
      dialogue = []
      kb_dialogue = []
  return data, kb

def fix_silence(utterance):
  '''
  Special keyword <SILENCE> is being split into three tokens
  So we revert that back to one token as required
  '''
  if (utterance[0] == "<") and (utterance[1] == "SILENCE"):
    utterance[0] = "<SILENCE>"
    utterance.pop(2)
    utterance.pop(1)

  return utterance

def parse_candidates(lines):
  '''
  :param lines: f.readlines()
  :return: list of all candidates ["hello", "A is a good restaurant"]
  '''
  candidates = []
  for line in lines:
    nid, line = line.split(' ', 1)
    line = line.decode('utf-8').strip()
    candidates.append(line.split('\t'))
  return candidates


def parse_dialogue_QA(lines, tokenizer=True):
  '''
  lines: f.readline(), which is actually a list of lines in txt
  Return [[(u1, u2, u3), (c1, c2, c3)], [(u1, u2, u3), (c1, c2, c3)], ...]
  '''
  data = []
  dialogue = []
  kb_dialogue = []
  kb = []
  u = []
  c = []
  for line in lines:
    if line != '\n' and line != lines[-1]:
      nid, line = line.split(' ', 1)
      nid = int(nid)
      line = line.decode('utf-8').strip()

      if len(line.split('\t')) == 1:
        kb_dialogue.append(line.split('\t'))
        continue

      q, a = line.split('\t')
      if tokenizer is True:
        q = tokenize(q)
        a = tokenize(a)
      u.append(q)
      c.append(a)
    else:
      dialogue.append(tuple(u))
      dialogue.append(tuple(c))
      data.append(dialogue)

      kb.append(kb_dialogue)
      dialogue = []
      u = []
      c = []
      kb_dialogue = []

  return data, kb

def read_restaurant_data(filename, restaraunt_prefix = "datasets/babi/"):
  '''
  :param filename: 'directory/file.txt'
  :return:[
    [(u1, r1), (u2, r2)...]
    , [(u1, r1), (u2, r2)...], ...]
  the data is a list of training example, each example consists of one dialog
  for each dialog, there are a number of turns
  each turn is made up of a tuple of (u_i, r_i) for up to N turns
    where ui is utterance from the customer
    where ri is a response from an agent
  each ui or ri, is a list of strings, for up to M tokens
    each token is usually a word or punctuation

  kb: the knowledge base in the format of
  [u'saint_johns_chop_house R_post_code saint_johns_chop_house_post_code',
  u'saint_johns_chop_house R_phone saint_johns_chop_house_phone',
  u'saint_johns_chop_house R_address saint_johns_chop_house_address',
  u'saint_johns_chop_house R_price moderate']
  '''
  restaurant_path = restaraunt_prefix + filename
  with open(restaurant_path) as f:
    # max_length = None
    data, kb = parse_dialogue(f.readlines())
  return data, kb

def select_consecutive_pairs(data, count):
  import pdb
  random_location = (random.random()/2.0) + 0.2  # random number from 0.2 to 0.7
  random_index = int(round(random_location * len(data)))
  random_query = data[random_index][0]  # [0] is query, [1] is response
  turn_index = int(random_query[0].cpu().data.numpy()[0])
  start_index = (random_index - (turn_index - 1))

  dialogues = []
  dialog = []
  while len(dialogues) < count:
    turn_pair = data[start_index]
    # check if we reached the end of the dialog
    turn_count = int(turn_pair[0][0].data.cpu().numpy()[0])
    if turn_count == 1 and (len(dialog) > 0):
      dialogues.append(dialog)
      dialog = []
    dialog.append(turn_pair)
    start_index += 1
  return dialogues

def spit_fire(dialogues):
  words = []
  for dialog in dialogues:
    for turn in dialog:
      for utterance in turn:
        for token in utterance:
          token_index = int(token.data.numpy()[0])
          words.append(vocab.index_to_word(token_index, "res"))
  print(words)

def look4str(u, df):
    a = df['addrs'].apply(lambda x: x in u)
    b = df['pois'].apply(lambda x: x in u)
    a = df[a]['addrs'].as_matrix()
    b = df[b]['pois'].as_matrix()

    if len(a) != 0:
        u = u.replace(a[0], 'addr')
    if len(b) != 0:
        u = u.replace(b[0], 'poi')
    return u


def load_incar_data(data_json):
    '''
    :param data_json: a json file loaded from .json
    :return:
    navigate/weather/schedule_data, three lists for the three tasks
    each list is a list of dialogues, and each dialogue is a list of turns [(u1, r1), (u2, r2)...]
    each utterance/response is a list of tokens
    '''
    lookup = pd.read_csv('datasets/in_car/incar_addr_poi.csv')
    navigate_data = []
    schedule_data = []
    weather_data = []
    kbs = []

    for dialogue in data_json:
        dia = []
        u = None
        r = None
        for turn in dialogue['dialogue']:
            if turn['turn'] == 'driver':
                u = turn['data']['utterance']
                u = look4str(u, lookup)
                u = word_tokenize(u.lower())
            if turn['turn'] == 'assistant':
                r = turn['data']['utterance']
                r = look4str(r, lookup)
                r = word_tokenize(r.lower())
                dia.append((u, r))

        if dialogue['scenario']['task']['intent'] == 'navigate':
            navigate_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'schedule':
            schedule_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'weather':
            weather_data.append(dia)
        else:
            print(dialogue['scenario']['task']['intent'])

        kbs.append(dialogue['scenario']['kb'])
    print('Loaded %i navigate data!'%len(navigate_data))
    print('Loaded %i schedule data!'%len(schedule_data))
    print('Loaded %i weather data!'%len(weather_data))

    return navigate_data, weather_data, schedule_data, kbs


def init_glove_words(name='6B', dim=100):
  '''
  :param name: which glove you want to load
  :param dim: dimension of word vectors
  :return: the glove object in pytorch
  '''
  import torchtext.vocab as vocab
  glove = vocab.GloVe(name='6B', dim=100)
  print('Loaded {} words'.format(len(glove.itos)))
  word_embeddings = glove
  return word_embeddings

def init_normal_words(vocab_size=1229, dim=100):
  # add EOS and SOS and UNK?
  mean = 0
  stddev = 1.0/math.sqrt(2 * math.pi)
  # each word embedding is a column vector
  word_embeddings = truncnorm.rvs(a=mean, b=stddev, size=[dim, vocab_size])
  return word_embeddings

