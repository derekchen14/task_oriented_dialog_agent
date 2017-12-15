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
class GRU_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(GRU_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size # dim of object passed into IFOG gates
    self.input_size = hidden_size # serves double duty
    self.use_cuda = use_cuda
    self.embedding = nn.Embedding(vocab_size, hidden_size)

    self.gru = nn.GRU(self.input_size, self.hidden_size)

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
      #  output (1, 1, 256),  hidden  (1, 1, 256)
    return output, hidden

  def initHidden(self):
    # args are (num_layers * num_directions, batch_size, hidden_size)
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class Bid_GRU_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(Bid_GRU_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size // 2, bidirectional=True)

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    for i in range(self.n_layers):
      output, hidden = self.gru(output, hidden)
      #  output (1, 1, 256),  hidden  (2, 1, 128)
    return output, hidden

  def initHidden(self):
    # During bi-directional encoding, we split up the word embedding in half
    # and use then perform a forward pass into two directions.  In code,
    # this is interpreted as 2 layers at half the size. Thus, x = 2.  Next,
    # we still perform stochastic gradient descent, so batch_size = y = 1.
    # Finally, recall that the output of a bi-directional GRU is the
    # concat of the two hidden layers, so in order to maintain the same
    # output size, we split each of the hidden layers in half, z = h // 2
    result = Variable(torch.zeros(2, 1, self.hidden_size // 2))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class Match_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(Match_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size + 8  # extended dim for the match features
    self.use_cuda = use_cuda
    self.gru = nn.GRU(self.hidden_size, self.hidden_size // 2, bidirectional = True)
    self.embedding = match_embedding(vocab_size, hidden_size)

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    for i in range(self.n_layers):
      output, hidden = self.gru(output, hidden)
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(2, 1, self.hidden_size // 2))
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

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    for i in range(self.n_layers):
      output, hidden = self.lstm(output, hidden)
      #  output (1, 1, 256),  hidden  (1, 1, 256)
    return output, hidden
    # return self.lstm(embedded, hidden, n_layers)

  def initHidden(self):
    # you need two variables since LSTMs have
    # (1) hidden state and (2) candidate cell state
    # whereas GRU have only the hidden state which does both
    hidden = Variable(torch.zeros(1, 1, self.hidden_size))
    cell = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return (hidden.cuda(), cell.cuda())
    else:
      return (hidden, cell)

class RNN_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(RNN_Encoder, self).__init__()
    self.n_layers = n_layers
    self.input_size = hidden_size
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.rnn = nn.RNN(self.input_size, self.hidden_size)

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

    # dimensions are timesteps, batch_size, input_size
    # timesteps = sequence_length, number of words in your sentence
    # batch_size = mini-batch size Ex. 1 means we are doing SGD
    # input_size = feature_dim  Ex. 256 means your word embedding is 256 dim
