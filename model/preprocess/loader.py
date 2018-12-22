import os, pdb, sys
import json
import logging

from pprint import pformat
from vocab import Vocab
from utils.internal.vocabulary import Vocabulary
from utils.external.glad_dataset import Dataset, Ontology

# Used for loading data, to be fed into the PreProcessor
class DataLoader(object):
  def __init__(self, args):
    self.data_dir = os.path.join('datasets', args.dataset)
    self.clean_dir = os.path.join(self.data_dir, 'clean')
    self.debug_dir = os.path.join(self.data_dir, 'debug')

    self.multitask = True
    if args.task == "end_to_end":
      self.categories = ["intent_tracker", "kb_lookup", "policy_manager", "text_generator"]
    elif args.task == "clarification":
      self.categories = ["belief_tracker", "policy_manager", "user_simulator"]
    elif args.task == "dual":
      self.categories = ["slot", "value"]
    elif args.task == "per_slot":
      self.categories = ["area", "food", "price", "request"]
    else:
      self.categories = args.task
      self.multitask = False

    if args.pretrained:
      with open(os.path.join(self.data_dir, 'embeddings.json')) as f:
       self.embeddings = json.load(f)

    if args.model == "glad":
      with open(os.path.join(self.data_dir, 'ontology.json')) as f:
        self.ontology = Ontology.from_dict(json.load(f))
      with open(os.path.join(self.data_dir, 'vocab.json')) as f:
        self.vocab = Vocab.from_dict(json.load(f))
      self.populate_dataset()
    else:
      if args.test_mode:
        self.test_data = self.load_dataset('test')
      else:
        self.train_data = self.load_dataset('train')
        self.val_data = self.load_dataset('val')
      self.vocab = Vocabulary(args, self.data_dir)

  def load_dataset(self, split):
    data_path = "{}/{}.json".format(self.clean_dir, split)
    with open(data_path, "r") as f:
      dataset = json.load(f)
    print("{} data loaded!".format(data_path))
    return dataset

  def populate_dataset(self):
    self.dataset = {}
    for split in ["train", "val", "test"]:
      with open(os.path.join(self.data_dir, '{}.json'.format(split))) as f:
        logging.info('Loading {} split'.format(split))
        self.dataset[split] = Dataset.from_dict(json.load(f))

    logging.info('Dataset sizes: {}'.format(pformat({k: len(v) for k, v in self.dataset.items()})))