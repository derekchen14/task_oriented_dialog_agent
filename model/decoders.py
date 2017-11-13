import torch
import torch.nn as nn

######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class RNN_Decoder(nn.Module):
  def __init__(self, hidden_size, output_size, n_layers=1):
    super(DecoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(output_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
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
    if use_cuda:
      return result.cuda()
    else:
      return result

class RNN_Attn_Decoder(nn.Module):
  def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=8):
    super(AttnDecoderRNN, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.max_length = max_length

    self.embedding = nn.Embedding(self.output_size, self.hidden_size)
    self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.output_size)

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
    if use_cuda:
      return result.cuda()
    else:
      return result