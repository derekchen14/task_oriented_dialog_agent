import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from objects.components import var, device
from objects.blocks.base import BaseBeliefTracker
from utils.external.reader import get_glove_name

class BasicClassifer(BaseBeliefTracker):
  def __init__(self, encoder, ff_network, args):
    super().__init__(args)
    self.encoder = encoder.to(device)
    self.decoder = ff_network.to(device)
    self.model_type = "basic"

  def forward(self, sources, hidden):
    self.encoder.rnn.flatten_parameters()
    encoder_outputs, hidden = self.encoder(sources, hidden)
    return self.decoder(encoder_outputs[0])

class PretrainedEmbeddingsModel(nn.Module):
  """
  Base class that allows pretrained embedding to be loaded from
  a vocabulary. Assumes the wrapper class will initialize the embeddings
  """
  def load_embeddings(self, vocab):
    if self.opt.pt != "none":
      # Get glove vector file name
      name = get_glove_name(self.opt, "tokens")
      print("Loading embeddings from {}".format(name))

      # Load glove vectors
      with open(name, "r") as f:
        token_words = pickle.load(f)

      # Assign glove vectors to correct word in vocab
      for i, word in vocab.iteritems():

        # If vocab word is meaning less, set embedding to 0
        if word in ["<unk>", "<start>", "<end>", "<pad>"]:
          self.embeddings.weight.data[i].zero_()
          continue

        if self.is_cuda:
          vec = torch.cuda.FloatTensor(token_words[word])
        else:
          vec = torch.FloatTensor(token_words[word])

        # Set embedding in embedding module
        self.embeddings.weight.data[i] = vec

class WeightedBOW(PretrainedEmbeddingsModel):
  """
  Indexes a set of word embeddings for a sequence of words it receives
  as input. Weighs each of these embeddings by a learned mask
  and returns the sum

  Initialization Args:
    opt.vSize: number of embeddings to initialize
    opt.hSize: size of embeddings to initialize
    opt.dpt: dropout probability after the embedding layer (default = 0)
    max_size: maximum number of masking functions

  Input:
    input: 3-dimensional tensor of batch_size x seq_len x embed_size

  Output:
    2-dimensional tensor of size batch_size x embed_size

  """
  def __init__(self, opt, max_size):
    super(WeightedBOW, self).__init__()
    self.embeddings = nn.Embedding(opt.vSize, opt.hSize, padding_idx=0)

    self.weights = nn.Parameter(
      torch.FloatTensor(max_size, opt.hSize).fill_(1))

    self.dropout = nn.Dropout(opt.dpt)

    self.is_cuda = False
    self.max_size = max_size
    self.opt = opt

  def forward(self, input):
    batch_size = input.size(0)
    length = input.size(1)

    embed = self.embeddings(input)
    dropped = self.dropout(embed)

    # Multiply dropped embeddings by mask weights
    w_dropped = dropped * self.weights[:length, :].unsqueeze(0).repeat(
      batch_size, 1, 1)

    return torch.sum(w_dropped, 1).view(batch_size, self.opt.hSize), None

  def cuda(self, device_id):
    super(WeightedBOW, self).cuda(device_id)
    self.weights.cuda(device_id)
    self.is_cuda = True