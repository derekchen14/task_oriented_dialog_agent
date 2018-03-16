# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import Transformer
from model.components import *
from utils.external.preprocessers import match_embedding

# ------- Encoders ----------
# The encoder of a seq2seq network is a RNN that outputs some value for every
# word from the input sentence. For every input word the encoder outputs a vector
# and a hidden state, and uses the hidden state for the next input word.
class Match_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(Match_Encoder, self).__init__()
    self.hidden_size = hidden_size + 8  # extended dim for the match features
    self.gru = nn.GRU(self.hidden_size, self.hidden_size // 2, \
      num_layers=n_layers, bidirectional=True)
    self.embedding = match_embedding(vocab_size, hidden_size)

  def forward(self, word_inputs, hidden):
    seq_len = len(word_inputs)
    embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
    output, hidden = self.gru(embedded, hidden)
    return output, hidden

  def initHidden(self):
    return smart_variable(torch.zeros(2, 1, self.hidden_size // 2))

class Transformer_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=6):
    super(Transformer_Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.scale_factor = math.sqrt(hidden_size)
    self.embedding = match_embedding(vocab_size, hidden_size)
    self.transformer = Transformer(hidden_size, n_layers)

  def forward(self, word_inputs):
    seq_len = len(word_inputs)
    embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
    final_output = self.transformer(embedded)
    return final_output

class Bid_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(Bid_Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.gru = nn.GRU(hidden_size, hidden_size // 2, \
      num_layers=n_layers, bidirectional=True)
    self.embedding = nn.Embedding(vocab_size, hidden_size)

  def forward(self, word_inputs, hidden):
    seq_len = len(word_inputs)  # now a matrix multiplication
    embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
    # output is seq_len, batch, hidden_size * num_directions: (8,1,264)
    # seq_len = 8, batch = 1, hidden_size = 128 + 4, num_directions = 2
    # hidden is num_layers * num_directions, batch, hidden_size: (2,1,132)
    # the two pieces in hidden are actually the bottom half of the last word
    #     and the top half of the *first* word since we are bidirectional
    output, hidden = self.gru(embedded, hidden)
    return output, hidden

  def initHidden(self):
    # During bi-directional encoding, we split up the word embedding in half
    # and use then perform a forward pass into two directions.  In code,
    # this is interpreted as 2 layers at half the size. Thus, x = 2.  Next,
    # we still perform stochastic gradient descent, so batch_size = y = 1.
    # Finally, recall that the output of a bi-directional GRU is the
    # concat of the two hidden layers, so in order to maintain the same
    # output size, we split each of the hidden layers in half, z = h // 2
    return smart_variable(torch.zeros(2, 1, self.hidden_size // 2))

class GRU_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(Attn_Encoder, self).__init__()
    self.hidden_size = hidden_size # dim of object passed into IFOG gates
    self.input_size = hidden_size # serves double duty

    self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=n_layers)
    self.embedding = nn.Embedding(vocab_size, hidden_size)

  def forward(self, word_inputs, hidden):
    seq_len = len(word_inputs)
    embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    # if we want to change into one-hot vectors
    # c = torch.eye(8)
    output, hidden = self.gru(embedded, hidden)
    return output, hidden
    # dimensions are timesteps, batch_size, input_size
    # timesteps = sequence_length, number of words in your sentence
    # batch_size = mini-batch size Ex. 1 means we are doing SGD
    # input_size = feature_dim  Ex. 256 means your word embedding is 256 dim

  def initHidden(self):
    # args are (num_layers * num_directions, batch_size, hidden_size)
    return smart_variable(torch.zeros(1, 1, self.hidden_size))

class LSTM_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(LSTM_Encoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size)

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    for i in range(self.n_layers):
      output, hidden = self.lstm(output, hidden)
    return output, hidden

  def initHidden(self):
    # you need two variables since LSTMs have
    # (1) hidden state and (2) candidate cell state
    # whereas GRU have only the hidden state which does both
    hidden = smart_variable(torch.zeros(1, 1, self.hidden_size))
    cell = smart_variable(torch.zeros(1, 1, self.hidden_size))
    return (hidden, cell)

class RNN_Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(RNN_Encoder, self).__init__()
    self.n_layers = n_layers
    self.input_size = hidden_size
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.rnn = nn.RNN(self.input_size, self.hidden_size)

  def forward(self, input, hidden):
    embedded = self.embedding(input).view(1, 1, -1)
    output = embedded
    for i in range(self.n_layers):
      output = self.rnn(output)
    return output

  def initHidden(self):
    return smart_variable(torch.zeros(1, 1, self.hidden_size))