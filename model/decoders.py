# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# ------- Decoders ----------
# Decoder is given an input token and hidden state. The initial input token is
# the start-of-string <SOS> token, and the first hidden state is the context
# vector (the encoder's last hidden state).
class GRU_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(GRU_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.input_size = hidden_size #serves double duty
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(self.input_size, self.hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    output = self.embedding(input).view(1, 1, -1)
    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class RNN_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(RNN_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.rnn = nn.RNN(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    output = self.embedding(input).view(1, 1, -1)
    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.rnn(output, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class LSTM_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1):
    super(LSTM_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.use_cuda = use_cuda

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.lstm = nn.LSTM(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    output = self.embedding(input).view(1, 1, -1)
    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.lstm(output, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result

class GRU_Attn_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, use_cuda, n_layers=1,
        dropout_p=0.1, max_length=8):
    super(GRU_Attn_Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
    self.use_cuda = use_cuda
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.max_length = max_length

    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
    self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(self, input, hidden, encoder_output, encoder_outputs):
    embedded = self.embedding(input).view(1, 1, -1)
    embedded = self.dropout(embedded)

    attn_weights = F.softmax(
      self.attn(torch.cat((embedded[0], hidden[0]), 1)))
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                 encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)

    output = F.log_softmax(self.out(output[0]))
    return output, hidden, attn_weights

  def initHidden(self):
    result = Variable(torch.zeros(1, 1, self.hidden_size))
    if self.use_cuda:
      return result.cuda()
    else:
      return result