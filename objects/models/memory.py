import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from objects.models.basics import WeightedBOW
from objects.models.attention import DoubleAttention
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


class SentenceEncoder(nn.Module):
  """
  Encode sentence using a weighted bag of words where the
  word embeddings are weighted by a positional mask

  Input:
    input: word indices based on vocabulary
    lengths: length of each set of words in the batchs
    embed: word embeddings in case we use shared embeddings
        with another module

  Output:
    output: hidden state from encoding the words
  """
  def __init__(self, opt, max_words=None):
    super(SentenceEncoder, self).__init__()
    self.opt = opt
    self.sentence_encoder = WeightedBOW(opt, max_words)

  def forward(self, input, lengths, embed=None):
    # Encode sentences
    output, hidden = self.sentence_encoder(input, lengths)

    return output

  def cuda(self, device_id):
    super(SentenceEncoder, self).cuda(device_id)
    self.sentence_encoder.cuda(device_id)
    self.is_cuda = True


class EntitySelector(nn.Module):
  """
  Select slot_values from memory using predicted intent as condition and
      hidden state from encoder as an additional gating mechanism
  Completely a wrapper around DoubleAttention for semantic purposes

  Input:
    hidden: hidden state of the encoder, used the condition vector
    memory_slots: the memory slots values at start of EntNet cycle. In the
          case of dialogue, these are the slots in the ontology we are tracking
          shape = (batch_size x num_slots x slot_vector)
          slot_vector is equivalent to the size of slot hidden_dim
    memory_values: the memory values associated with the slots, these values
          are generically called "entities".
          shape = (batch_size x num_slots x value_vector)
          based on convention the slot_vector dim == value_vector dim

  Output:
    a tuple of weights, normalized, scores which represent
    weights: selected value vectors from attention distribution
    normalized: normalized distribution of activation scores over values
    scores: raw activation scores of attention
  """
  def __init__(self, config):
    super().__init__()
    self.attend = DoubleAttention(act="sigmoid")
    self.config = config

  def forward(self, hidden, memory_slots, memory_values):
    return self.attend(hidden, memory_slots, memory_values)



class EntityUpdater(nn.Module):
  """
  Update entities using entity values changed by applicator

  Initialization Args:
    opt.eSize: size of entities

  Input:
    entities: entity vals at start of EntNet cycle
          (batch_size x num_entities x entity_size)
    new_entities: new_entity from applicator
          (batch_size x entity_size)
    dist: attention distribution from selecting entities for change
    keys: key vector for entities
    ctx (default=None): context from sentence encoder

  Output:
    updated entity vectors (batch_size x num_entities x entity_size)

  """
  def __init__(self, opt):
    super(EntityUpdater, self).__init__()
    self.opt = opt

    # Update entities with projected key (See EntNet Paper)
    self.key_applicator = nn.Linear(opt.eSize, opt.eSize, False)

    # Update entities with projected value (See EntNet Paper)
    self.val_applicator = nn.Linear(opt.eSize, opt.eSize, False)

    # Update entities with projected context (See EntNet Paper)
    self.ctx_applicator = nn.Linear(opt.hSize, opt.eSize, False)

    self.act = nn.PReLU(opt.eSize)
    self.act.weight.data.fill_(1)

  def forward(self, entities, new_entities, dist, keys, ctx=None):
    batch_size = entities.size(0)
    num_items = entities.size(1)
    hidden_size = entities.size(2)

    ok, oe, oc = self.compute_update_components(
      entities, keys, ctx)

    # Format attention weights
    n_dist = dist.view(batch_size, num_items, 1).expand(
      batch_size, num_items, hidden_size)

    new_ents = self.update_entities(ok, oe, oc, n_dist, entities)

    return new_ents * (1 / new_ents.norm(2, 2)).unsqueeze(2).expand(
      batch_size, num_items, hidden_size), n_dist

  def update_entities(self, ok, oe, oc, n_dist, entities):
    batch_size = entities.size(0)
    num_items = entities.size(1)
    hidden_size = entities.size(2)

    # Update entities "n" = don't interpolate
    if "n" not in self.opt.afunc:
      if self.opt.act == "I":
        new_ents = ((n_dist * (oe + ok + oc)) + (1 - n_dist) *
              entities)
      else:
        pre_act = (oe + ok + oc).view(-1, hidden_size)
        new_ents = ((n_dist * self.act(pre_act).view(
          batch_size, num_items, -1)) + (1 - n_dist) * entities)
    else:
      if self.opt.act == "I":
        new_ents = entities + n_dist * (oe + ok + oc)
      else:
        pre_act = (oe + ok + oc).view(-1, hidden_size)
        new_ents = (n_dist * self.act(pre_act).view(
          batch_size, num_items, -1) + entities)

    return new_ents

  def compute_update_components(self, entities, keys, ctx):
    batch_size = entities.size(0)
    num_items = entities.size(1)
    hidden_size = entities.size(2)

    ok = 0  # key contribution        (V w)
    oe = 0  # old entity contribution (U h)
    oc = 0  # context contribution    (W s)

    # Key contribution from EntNet paper
    ok = self.key_applicator(
      keys.view(batch_size * num_items, hidden_size)).view(
        batch_size, num_items, hidden_size)

    # Value contribution from EntNet paper
    oe = self.val_applicator(
      entities.view(batch_size * num_items, hidden_size)).view(
        batch_size, num_items, hidden_size)

    # Context contribution from EntNet paper
    oc = self.ctx_applicator(ctx).view(batch_size, 1, -1).repeat(
      1, num_items, 1)

    return ok, oe, oc

  def cuda(self, device_id):
    super(EntityUpdater, self).cuda(device_id)
    self.is_cuda = True
    self.key_applicator.cuda(device_id)
    self.val_applicator.cuda(device_id)
    self.ctx_applicator.cuda(device_id)
    self.act.cuda(device_id)

