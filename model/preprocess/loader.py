import os, pdb, sys
import json
import logging

from pprint import pformat
from vocab import Vocab
from utils.external.glad_dataset import Dataset, Ontology

# Used for loading data, to be fed into the PreProcessor
class DataLoader(object):
  def __init__(self, args):
    if args.model == "glad":
      self.data_dir = os.path.join('datasets', args.task, 'ann')
      with open(os.path.join(self.data_dir, 'ontology.json')) as f:
        self.ontology = Ontology.from_dict(json.load(f))
      with open(os.path.join(self.data_dir, 'vocab.json')) as f:
        self.vocab = Vocab.from_dict(json.load(f))
      with open(os.path.join(self.data_dir, 'emb.json')) as f:
        self.embeddings = json.load(f)
      self.populate_dataset()
    else:
      self.description = "only glad uses this so far"

  def populate_dataset(self):
    self.dataset = {}
    for split in ["train", "dev", "test"]:
      with open(os.path.join(self.data_dir, '{}.json'.format(split))) as f:
        logging.info('Loading {} split'.format(split))
        self.dataset[split] = Dataset.from_dict(json.load(f))

    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in self.dataset.items()})))
