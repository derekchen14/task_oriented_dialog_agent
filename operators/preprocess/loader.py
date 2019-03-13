import os, pdb, sys
import json
import pickle as pkl
import torch

from vocab import Vocab
from utils.internal.vocabulary import Vocabulary
from utils.external.reader import text_to_dict, get_glove_name
from objects.blocks import Dataset
from utils.internal.ontology import Ontology

# Used for loading data, to be fed into the PreProcessor
class DataLoader(object):
  def __init__(self, args):
    self.data_dir = os.path.join('datasets', args.dataset)
    self.clean_dir = os.path.join(self.data_dir, 'clean')
    self.debug_dir = os.path.join(self.data_dir, 'debug')
    self.task = args.task

    if args.pretrained:
      self.embeddings = json.load(self.path('embeddings.json'))

    if self.task == "track_intent":
      self.vocab = Vocab.from_dict(json.load(self.path('vocab.json')))
      self.ontology = Ontology.from_dict(json.load(self.path('ontology.json')))
    elif self.task == "manage_policy":
      self.kb = pkl.load(self.path("knowledge_base.p", "rb"), encoding="latin1")
      self.ontology = Ontology.from_path(self.data_dir)
      self.vocab = Vocabulary(args, self.data_dir)
    elif self.task == 'end_to_end':
      self.kb = pkl.load(self.path("knowledge_base.p", "rb"), encoding="latin1")
      self.ontology = Ontology.from_path(self.data_dir)
      self.vocab = Vocabulary(args, self.data_dir)
      self.act_set = self.text_data('dia_acts')
      self.slot_set = self.text_data('slot_set')

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
        if self.task == 'track_intent':
          dataset = Dataset.from_dict(dataset)
      self.datasets[split] = dataset
      print("{} loaded with {} items!".format(data_path, len(dataset)))

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

  def restore_checkpoint(self, filepath):
    model_checkpoint = torch.load(filepath)
    print('Loaded model from {}'.format(filepath))
    return model_checkpoint

  def json_data(self, filename):
    file_path = self.path(filename + '.json', 'r')
    return json.load(file_path)

  def pickle_data(self, filename, directory=None):
    if directory is None:
      file_path = self.path(filename + '.pkl', 'rb')
      return pkl.load(file_path, encoding='latin1')
    else:
      file_path = os.path.join(directory, filename + '.pkl')
      return pkl.load(open(file_path, 'rb'), encoding='latin1')

  def text_data(self, filename):
    full_set = {}
    with self.path(filename + '.txt', 'r') as f:
      index = 0
      for line in f.readlines():
        full_set[line.strip('\n').strip('\r')] = index
        index += 1
    return full_set


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
