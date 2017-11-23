# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy.stats import truncnorm
import math
import json

from nltk import word_tokenize
import utils.internal.vocabulary as vocab

def load_car_dataset(task, split):
    prefix = 'datasets/in_car/'
    paths = {'train': prefix+'train.json', 'dev':prefix+'dev.json', 'test':prefix+'test.json'}


def load_dataset(task, split):
  if task in ['1', '2', '3', '4', '5']:
    dataset_name = "dialog-babi-task{0}-{1}.txt".format(task, split)
    restaurants, kb = read_restuarant_data(dataset_name)
    candidates = read_restuarant_data("dialog-babi-candidates.txt")
    max_length = 22
    return (restaurants, candidates, max_length)
  elif task == 'dstc':
    dataset_name = "dialog-babi-task6-{0}2-{1}.txt".format(task, split)
    restaurants, kb = read_restuarant_data(dataset_name)
    candidates = read_restuarant_data("dialog-dstc-candidates.txt")
    max_length = 30
    return (restaurants, candidates, max_length)
  elif task in ['schedule','navigate','weather']:
    dataset_name = "kvret_{1}_public.txt".format(split)
    max_length = None  #TODO
    return read_car_data(dataset_name, task)
  elif task == 'concierge':
    raise ValueError("Sorry, concierge task not supported at this time")
  else:
    raise ValueError("Not a valid task")
  

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
      line = line.decode('utf-8').strip()

      if len(line.split('\t')) == 1:
        kb_dialogue.append(line.split('\t'))
        continue

      u, r = line.split('\t')
      if tokenizer is True:
        u = word_tokenize(u)
        r = word_tokenize(r)
      dialogue.append((u, r))
    else:
      data.append(dialogue)
      kb.append(kb_dialogue)
      dialogue = []
      kb_dialogue = []
  return data, kb


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


def word_to_glove_vector(glove, word):
  '''
  :param glove: Glove object from pytorchtext
  :param word: str
  :return: the embedding vector of the word
  '''
  return glove.vectors[glove.stoi[word]]

def read_restuarant_data(filename, restaraunt_prefix = "datasets/restaurants/"):
  '''
  :param filename: 'directory/file.txt'
  :return:[
    [(u1, r1), (u2, r2)...]
    , [(u1, r1), (u2, r2)...], ...]
  the data is a list of training examples
  each example consists of one dialog
  for each dialog, there are a number of turns
  each turn is made up of a tuple of (u_i, r_i) for up to N turns
    where ui is utterance from the customer
    where ri is a response from an agent
  each ui or ri, is a list of strings, for up to M tokens
    each token is usually a word or punctuation
  if the customer said nothing during their turn,
    special token of <SILENCE> is used

  kb: the knowledge base in the format of
  [u'saint_johns_chop_house R_post_code saint_johns_chop_house_post_code',
  u'saint_johns_chop_house R_cuisine british', u'saint_johns_chop_house R_location west',
  u'saint_johns_chop_house R_phone saint_johns_chop_house_phone',
  u'saint_johns_chop_house R_address saint_johns_chop_house_address',
  u'saint_johns_chop_house R_price moderate']
  '''
  restaurant_path = restaraunt_prefix + filename
  with open(restaurant_path) as f:
    # max_length = None
    data, kb = parse_dialogue(f.readlines())
  return data, kb


def load_json_dataset(path):
    '''
    Load the in-car dataset as it is
    :param path: path to train/validation/test.json
    :return: the json file
    '''
    with open(path) as f:
        data = json.load(f)
    print path + ' file loaded!!'
    return data


def load_incar_data(data_json):
    '''
    :param data_json: a json file loaded from .json
    :return:
    navigate/weather/schedule_data, three lists for the three tasks
    each list is a list of dialogues, and each dialogue is a list of turns [(u1, r1), (u2, r2)...]
    each utterance/response is a list of tokens
    '''
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
                u = word_tokenize(u)
            if turn['turn'] == 'assistant':
                r = turn['data']['utterance']
                r = word_tokenize(r)

                dia.append((u, r))
                u = None
                r = None

        if dialogue['scenario']['task']['intent'] == 'navigate':
            navigate_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'schedule':
            schedule_data.append(dia)
        elif dialogue['scenario']['task']['intent'] == 'weather':
            weather_data.append(dia)
        else:
            print dialogue['scenario']['task']['intent']

        print 'Loaded %i navigate data!'%len(navigate_data)
        print 'Loaded %i schedule data!'%len(schedule_data)
        print 'Loaded %i weather data!'%len(weather_data)

        kbs.append(dialogue['scenario']['kb'])

    return navigate_data, weather_data, schedule_data, kbs


def read_car_data(filename, task):
  car_prefix = "datasets/in_car/"
  car_path = car_prefix + filename
  ent_path = car_prefix + "kvret_entities.json"
  raw_data = json.load(open(car_path, "r") )
  entities = json.load(open(ent_path, "r") )

  car_data = []
  for dialog in car_data:
    parse_it_todo(task)
    car_data.append(parsed_dialog)
  return car_data, entities

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



'''
class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS"}
    self.n_words = 2  # Count SOS and EOS

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
  )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  return s


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
  print("Reading lines...")

  # Read the file and split into lines
  lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
    read().strip().split('\n')

  # Split every line into pairs and normalize
  pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

  # Reverse pairs, make Lang instances
  if reverse:
    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)
  else:
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

  return input_lang, output_lang, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

MAX_LENGTH = 10

eng_prefixes = (
  "i am ", "i m ",
  "he is", "he s ",
  "she is", "she s",
  "you are", "you re ",
  "we are", "we re ",
  "they are", "they re "
)


def filterPair(p):
  return len(p[0].split(' ')) < MAX_LENGTH and \
    len(p[1].split(' ')) < MAX_LENGTH and \
    p[1].startswith(eng_prefixes)


def filterPairs(pairs):
  return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
  input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
  print("Read %s sentence pairs" % len(pairs))
  pairs = filterPairs(pairs)
  print("Trimmed to %s sentence pairs" % len(pairs))
  print("Counting words...")
  for pair in pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])
  print("Counted words:")
  print(input_lang.name, input_lang.n_words)
  print(output_lang.name, output_lang.n_words)
  return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

# Preparing Training Data
# -----------------------
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.

def indexesFromSentence(lang, sentence):
  return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
  indexes = indexesFromSentence(lang, sentence)
  indexes.append(EOS_token)
  result = Variable(torch.LongTensor(indexes).view(-1, 1))
  if use_cuda:
    return result.cuda()
  else:
    return result


def variablesFromPair(pair):
  input_variable = variableFromSentence(input_lang, pair[0])
  target_variable = variableFromSentence(output_lang, pair[1])
  return (input_variable, target_variable)



'''
