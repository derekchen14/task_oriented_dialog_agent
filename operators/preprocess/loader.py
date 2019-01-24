import os, pdb, sys
import json
import pickle as pkl
import logging

from vocab import Vocab
from utils.external.reader import text_to_dict, get_glove_name
from objects.blocks import Dataset, Ontology

class ModuleLoader(object):
  def __init__(self, args):
    pass

  def load_intent_predictor(self, model_path):
    pass

  def load_policy_manager(self, model_path):
    pass

  def load_text_generator(self, model_path, template_path):
    model_params = pkl.load(open(model_path, 'rb'))
    text_generator = TextGenerator.from_params(model_params)
    templates = json.load(open(path, 'rb'))
    text_generator.set_templates(templates)

# Used for loading data, to be fed into the PreProcessor
class DataLoader(object):
  def __init__(self, args):
    self.data_dir = os.path.join('datasets', args.dataset)
    self.clean_dir = os.path.join(self.data_dir, 'clean')
    self.debug_dir = os.path.join(self.data_dir, 'debug')
    self.task = args.task

    if args.pretrained:
      self.embeddings = json.load(self.path('embeddings.json'))

    if self.task == "glad":
      self.ontology = Ontology.from_path(self.path('ontology.json'))
      self.vocab = Vocab.from_dict(json.load(self.path('vocab.json')))
    elif self.task == "rule":
      self.intent_sets = {
        "act": text_to_dict(self.path("act_set.txt")),
        "slot": text_to_dict(self.path("slot_set.txt")),
        "value": pkl.load(open(self.path("value_set.p"), "rb"))
      }
      self.kb = pkl.load(open(self.path("knowledge_base.p")))

    self.load_datasets()

  def path(self, filename, kind=None):
    if kind is None:
      return open(os.path.join(self.data_dir, filename))
    else:
      return open(os.path.join(self.data_dir, filename), kind)

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

  def load_entity_embeddings(self, vocab):
    # made for EntNet
    if self.opt.pt != "none":
      name = get_glove_name(self.opt, "entities", "entpt")
      print("Loading entity embeddings from {}".format(name))
      entity_words = pkl.load( open(name, "r") )

      for i, word in vocab.iteritems():
        if word in ["<unk>", "<start>", "<end>", "<pad>"]:
          self.key_init.weight.data[i].zero_()
          continue
        vec = tensor(entity_words[word]).to(device)
        self.key_init.weight.data[i] = vec


  """
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
  """