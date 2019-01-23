import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from objects.blocks.selectors import *
from objects.blocks.attention import DoubleAttention
from objects.blocks.basics import WeightedBOW
from objects.models.encoders import MatchEncoder
from objects.components import device

class EntNet(nn.Module):
  """
  Recurrent Entity Networks wrapper class that wraps around the
  individual components of the network
  (Paper from Mike Henaff, original code from Antoine Bosselut)

  Initialization Args:
    max_words: maximum number of words in input sentences
           (useful for positional mask in EntNet paper)
    opt: options for all model components
    opt.ents: number of entities to initialize
    opt.eSize: size of entity embeddings to initialize

  Input:
    sentence: list of two components
    sentence[0]: Tensor of sentences (batch_size x seq_len)
    sentence[1]: tensor of sentences lengths (batch_size) for
           sentences in batch
    question: same as for sentence
    answers: tensor of answer values
    prev_attn: attention for selecting entities from previous step
           of updating entities. Recurrent Attention in paper

  Output:
    new entity values
    attention over new entities
    loss with respect to entity selection if it is supervised
  """
  def __init__(self, opt, max_words=None):
    super(EntNet, self).__init__()
    self.encoder = SentenceEncoder(opt, max_words)
    self.entity_selector = EntitySelector(opt)
    self.entity_updater = EntityUpdater(opt)

    self.key_init = nn.Embedding(opt.ents, opt.eSize)

    self.crit = nn.BCEWithLogitsLoss()

    self.opt = opt

    self.is_cuda = False

  def initialize_entities(self, entity_ids=None, batch_size=32):
    # If no special entities are being initialized, just select n entities
    # In this work, entities are usually tied -> entity_ids should be given
    if entity_ids is None:
      entity_ids = Variable(torch.LongTensor(range(self.opt.ents)).view(
        1, self.opt.ents).expand(batch_size, self.opt.ents))
      entity_ids = entity_ids.to(device)
    keys = self.key_init(entity_ids)

    # Should we lock the keys to their starting value (useful if keys
    # are initialized with Glove vectors or something of the like)
    self.keys = keys.detach() if self.opt.lk else keys

    # Initialize entity memory values the same as keys
    entities = self.key_init(entity_ids.detach())

    return self.keys, entities, None

  def forward(self, sentences, sentence_lengths,
            entity_labels=None, entity_init=None):
    # print entity_init
    bs = sentences[sentences.keys()[0]].size(0)
    keys, entities, _ = self.initialize_entities(
      entity_ids=entity_init, batch_size=bs)

    sel_loss = 0

    for i in sorted(sentences.keys()):
      sentence = sentences[i]
      # Encode sentence, sentence[0] = sentence tensors
      # sentence[1] = sentence length tensors (need this for padding)
      sent_emb = self.encoder(Variable(sentence), sentence_lengths[i])

      # Select entities
      _, attn_dist, attn_acts = self.entity_selector(
        sent_emb, entities, self.keys)

      if entity_labels is not None:  # and entity_labels[i].sum() != 0:
        sel_loss += self.crit(attn_acts, Variable(entity_labels[i]))

      # Upate entities
      entities, n_dist = self.entity_updater(
        entities, None, attn_dist, self.keys, sent_emb)

      joint = (n_dist * entities).sum(1)

    return entities, joint, sel_loss

  def cuda(self, device_id):
    super(EntNet, self).cuda(device_id)
    self.encoder.cuda(device_id)
    self.entity_selector.cuda(device_id)
    self.entity_updater.cuda(device_id)
    self.key_init.cuda(device_id)
    # self.entity_init.cuda(device_id)
    self.crit.cuda(device_id)
    self.is_cuda = True


class NPN(nn.Module):
  """
  Neural Process Network wrapper class that wraps around the
  individual components of the network

  Initialization Args:
    max_words: maximum number of words in input sentences
           (useful for positional mask in REN paper)
    q_max_words: maximum number of words in questions
          (useful for positional mask in REN paper)
    opt: options for all model components
    opt.ent: number of entities to initialize
    opt.eSize: size of entity embeddings to initialize

  Input:
    sentence: list of two components
    sentence[0]: Tensor of sentences (batch_size x seq_len)
          -- might be other way around
    sentence[1]: tensor of sentences lengths (batch_size) for
           sentences in batch
    question: same as for sentence
    answers: tensor of answer values
    prev_attn: attention for selecting entities from previous step
           of updating entities. Recurrent Attention in paper

  Output:
    If sentence:
    new entity values
    combined output
    selection loss if relevant



  """
  def __init__(self, opt, max_words=None):
    super(NPN, self).__init__()
    # Initialize encoder for entity selection
    self.encoder = SentenceEncoder(opt, max_words)
    # Initialize separate action encoder
    self.action_enc = SentenceEncoder(opt, max_words)
    # Initialize action selector
    self.action_selector = ActionSelector(opt)
    # Initialize applicator for agents and themes
    self.theme_applicator = BilinearApplicator(opt)
    # Initialize entity selector
    self.entity_selector = EntitySelector(opt)
    # Initialize entity updater
    self.entity_updater = EntityUpdater(opt)
    # Initialize entity embeddings
    self.key_init = nn.Embedding(opt.ents, opt.eSize)

    self.crit = nn.BCEWithLogitsLoss()

    self.opt = opt

    self.is_cuda = False

  def initialize_entities(self, entity_ids=None, batch_size=32):
    if entity_ids is None:
      entity_ids = Variable(torch.LongTensor(range(
        self.opt.ents)).view(
        1, self.opt.ents).expand(
        batch_size, self.opt.ents))
      if self.is_cuda:
        entity_ids = entity_ids.cuda(cfg.device)
    keys = self.key_init(entity_ids)

    if self.opt.lk:
      keys = keys.detach()

    self.keys = keys

    entities = self.key_init(entity_ids.detach())
    return self.keys, entities, None

  def forward(self, sentences, sentence_lengths, entity_labels=None, entity_init=None):
    bs = sentences[sentences.keys()[0]].size(0)
    keys, entities, _ = self.initialize_entities(
      entity_ids=entity_init, batch_size=bs)

    sel_loss = 0

    attn_dist = None

    for i in sorted(sentences.keys()):
      sentence = sentences[i]

      # Encode sentence, sentence[0] = sentence tensors
      # sentence[1] = sentence length tensors (need this for padding)
      sent_emb = self.encoder(
        Variable(sentence), sentence_lengths[i])

      act_emb = self.action_enc(
        Variable(sentence), sentence_lengths[i])

      # Select actions
      actions = self.action_selector(act_emb)

      # Select entities
      selected_entities, attn_dist, attn_acts = self.entity_selector(
        sent_emb, entities, self.keys, attn_dist)

      if entity_labels is not None:  # and entity_labels[i].sum() != 0:
        sel_loss += self.crit(attn_acts, Variable(entity_labels[i]))

      # Apply action to entities
      changed_entities = self.theme_applicator(
        actions, selected_entities)

      # Upate entities
      entities, n_dist = self.entity_updater(
        entities, changed_entities, attn_dist, self.keys, sent_emb)

      joint = (n_dist * entities).sum(1)

    return entities, joint, sel_loss

  def load_entity_embeddings(self, vocab):
    if self.opt.pt != "none":
      name = "data/{}/entities.{}{}.vocab".format(
        self.opt.pt, self.opt.entpt,
        self.opt.eSize)

      print("Loading entity embeddings from {}".format(name))
      with open(name, "r") as f:
        entity_words = pickle.load(f)

      for i, word in vocab.iteritems():
        if word in ["<unk>", "<start>", "<end>", "<pad>"]:
          self.key_init.weight.data[i].zero_()
          continue
        if self.is_cuda:
          vec = torch.cuda.FloatTensor(entity_words[word])
        else:
          vec = torch.FloatTensor(entity_words[word])
        self.key_init.weight.data[i] = vec

  def initialize_actions(self, init, vocab):
    if init == "n":
      # Xavier initialization
      stdv = 1. / math.sqrt(self.action_selector.actions.size(1))
      self.action_selector.actions.data.uniform_(-stdv, stdv)
    elif "gauss" in init:
      # Gaussian initialization
      deets = init.split("+")
      mean = deets[1]
      stddev = deets[2]
      print("Using Gaussian Initialization with Mean = {} and \
        Standard Deviation = {} for actions").format(
        float(mean), float(stddev))
      self.action_selector.actions.data.normal_(
        float(mean), float(stddev))
    else:
      raise

  def cuda(self, device_id):
    super(NPN, self).cuda(device_id)
    self.encoder.cuda(device_id)
    self.action_selector.cuda(device_id)
    self.theme_applicator.cuda(device_id)
    self.entity_selector.cuda(device_id)
    self.entity_updater.cuda(device_id)
    self.key_init.cuda(device_id)

    self.action_enc.cuda(device_id)
    # self.entity_init.cuda(device_id)
    self.crit.cuda(device_id)
    self.is_cuda = True

