import os, pdb, sys
import json
import logging

from vocab import Vocab
from utils.external.glad_dataset import Dataset, Ontology

# Used for loading data, to be fed into the PreProcessor
class DataLoader(object):
  def __init__(self, args):
    self.data_dir = os.path.join('datasets', args.dataset)
    self.clean_dir = os.path.join(self.data_dir, 'clean')
    self.debug_dir = os.path.join(self.data_dir, 'debug')
    self.task = args.task

    if args.pretrained:
      with open(os.path.join(self.data_dir, 'embeddings.json')) as f:
       self.embeddings = json.load(f)

    if self.task == "glad":
      with open(os.path.join(self.data_dir, 'ontology.json')) as f:
        self.ontology = Ontology.from_dict(json.load(f))
      with open(os.path.join(self.data_dir, 'vocab.json')) as f:
        self.vocab = Vocab.from_dict(json.load(f))

    self.set_categories()
    self.load_datasets()

  def set_categories(self):
    self.multitask = True
    if self.task == "end_to_end":
      self.categories = ["intent_tracker", "kb_lookup", "policy_manager", "text_generator"]
    elif self.task == "clarification":
      self.categories = ["belief_tracker", "policy_manager", "user_simulator"]
    elif self.task == "dual":
      self.categories = ["slot", "value"]
    elif self.task == "per_slot":
      self.categories = ["area", "food", "price", "request"]
    else:
      self.categories = self.task
      self.multitask = False

  def load_datasets(self):
    self.datasets = {}
    for split in ['train', 'val', 'test']:
      data_path = os.path.join(self.clean_dir, '{}.json'.format(split))
      with open(data_path, 'r') as f:
        dataset = json.load(f)
        if self.task == 'glad':
          dataset = Dataset.from_dict(dataset)
      self.datasets[split] = dataset
      logging.info("{} loaded with {} items!".format(data_path, len(dataset)))