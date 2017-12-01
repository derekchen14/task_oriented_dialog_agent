# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.external.preprocessers import match_embedding

# ------- Encoders ----------
# The encoder of a seq2seq network is a RNN that outputs some value for every
# word from the input sentence. For every input word the encoder outputs a vector
# and a hidden state, and uses the hidden state for the next input word.
class Match_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(GRU_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size + 8  # extended dim for the match features
    self.use_cuda = use_cuda
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.embedding = match_embedding(vocab_size, hidden_size)

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    # if we want to change into one-hot vectors
    output = embedded
    for i in range(self.n_layers):
      output, hidden = self.gru(output, hidden)
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class GRU_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(GRU_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)

  def forward(self, input, hidden):
    # matched_input = add_match_features(input)
    # add_match_features
    embedded = self.embedding(input).view(1, 1, -1)
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    # if we want to change into one-hot vectors
    # c = torch.eye(8)
    output = embedded
    for i in range(self.n_layers):
      output, hidden = self.gru(output, hidden)
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class LSTM_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(LSTM_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size)

  # Written different from GRU on purpose, http://pytorch.org/docs/master/nn.html
  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    return self.lstm(embedded, hidden, n_layers)

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class RNN_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(RNN_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.rnn = nn.RNN(hidden_size)

  def forward(self, input, hidden):
    # matched_input = add_match_features(input)
    embedded = self.embedding(input).view(1, 1, -1)
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    # if we want to change into one-hot vectors
    output = embedded
    for i in range(self.n_layers):
      output = self.rnn(output)
    return output

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result