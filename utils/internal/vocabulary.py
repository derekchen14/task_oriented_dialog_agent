import os
import json
from torch import eye
import pdb, sys

class Vocabulary(object):
  def __init__(self, args, data_dir, kind=None):
    self.vocab = self.load_words(data_dir)
    labels = self.load_labels(data_dir, args.task)
    self.label_vocab = labels[kind] if kind is not None else labels

  def load_words(self, path):
    with open("{}/vocab.json".format(path), "r") as f:
      vocab = json.load(f)
    print("{} vocab loaded!".format(path))
    return vocab
  def load_labels(self, path, task):
    label_path = "{}/{}.json".format(path, task)
    if not os.path.exists(label_path):
      label_path = "{}/{}.json".format(path, "label_vocab")
    with open(label_path, "r") as f:
      labels = json.load(f)
    print("{} labels loaded!".format(label_path))
    return labels

  def word_to_index(self, token):
    return self.vocab.index(token)
  def label_to_index(self, token):
    return self.label_vocab.index(token)

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