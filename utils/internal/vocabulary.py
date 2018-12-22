import os
import json
from torch import eye
import pdb, sys

class Vocabulary(object):
  def __init__(self, args, data_dir):
    self.task = args.task
    self.dataset = args.dataset

    self.vocab = self.load_words(data_dir)
    special_tasks = ["full_enumeration", "ordered_values", "possible_only", "dual", "per_slot"]
    if self.task in special_tasks:
      self.label_vocab = self.load_labels(self.task)
    else:
      self.label_vocab = self.load_labels("label_vocab")

  def load_words(self, path):
    with open("{}/vocab.json".format(path), "r") as f:
      vocab = json.load(f)
    print("{} vocab loaded!".format(path))
    return vocab
  def load_labels(self, task):
    path = os.path.join("datasets", self.dataset)
    with open("{}/{}.json".format(path, task), "r") as f:
      labels = json.load(f)
    print("{} labels loaded!".format(task))
    return labels

  def word_to_index(self, token):
    return self.vocab.index(token)
  def label_to_index(self, label, kind=None):
    act, slot, value = label
    token = "{}={}".format(slot, value)
    if kind is None:
      return self.label_vocab.index(token)
    else:
      return self.label_vocab[kind].index(token)

  def index_to_word(self, idx):
    return self.vocab[idx]
  def index_to_label(self, idx):
    return self.label_vocab[idx]

  def ulary_size(self):
    return len(self.vocab)
  def label_size(self):
    return len(self.label_vocab)

'''
special_tokens = ["<SILENCE>", "<T01>","<T02>","<T03>", ... , "<T14>",
          "UNK", "SOS", "EOS", "api_call","poi", "addr"]
UNK_token = 15
SOS_token = 16
EOS_token = 17
PHONE_token = 19
POI_token = 19
ADDR_token = 20
'''