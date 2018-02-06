# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from components import smart_variable
from utils.external.preprocessers import match_embedding

# ------- Decoders ----------
# Decoder is given an input token and hidden state. The initial input token is
# the start-of-string <SOS> token, and the first hidden state is the context
# vector (the encoder's last hidden state, not the last output!).

# During the forward pass:
# input: scalar, used for indexing into vocab embedding
#    producing output with shape [1, 1, 256]
# hidden: [1, 1, 256] with dimensions:
#    depth = 1 because we are using words
#         for images they might be 3 for the RGB channels
#         sometimes depth is 2 because we have a bi-directional layer
#         if doing many words at once, this could be "sequence length"
#    batch size = 1
#    dim = 256 for word embeddings
#         for images this might be 784 for a 28x28 pixel image

class Copy_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1,
        dropout_p=0.1, max_length=8, verbose=False):
    super(Copy_Decoder, self).__init__()
    self.vocab_size = vocab_size  # check extend_vocab method below
    self.hidden_size = hidden_size + 8
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.max_length = max_length
    self.verbose = verbose

    self.attention_W = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attention_U = nn.Linear(self.hidden_size * 2, self.hidden_size)

    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    self.copy_mode = nn.Linear(self.hidden_size, self.hidden_size)
    self.generate_mode = nn.Linear(self.hidden_size, self.vocab_size)

  def extend_vocab(self, input_sentence):
    num_words_to_be_copied = len(input_sentence)
    self.vocab_size += num_words_to_be_copied
    self.out = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(self, decoder_input, hidden_state, encoder_outputs, input_variable):
    if self.verbose:
      print("decoder_input: {}".format(decoder_input.size()) )
      print("hidden_state: {}".format(hidden_state.size()) )
      print("encoder_outputs: {}".format(encoder_outputs.size()) )
      print("input_variable: {}".format(input_variable.size()) )

    input_length = input_variable.size()[0]
    batch_size = 1
    if (hidden_state.size()[0] == (2 * decoder_input.size()[0])):
      hidden_state = hidden_state.view(batch_size, 1, -1)

    embedded = self.embedding(decoder_input).view(batch_size, 1, -1)
    embedded = self.dropout(embedded)

    attn_weights = F.softmax(                                         # (1x30)
            self.attention_W(torch.cat((embedded[0], hidden_state[0]), 1)))
    attn_applied = torch.bmm(                 # [b x 1 x 30] x [b x 30 x 264]
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) # (1x1x264)
    joined_input = torch.cat((embedded[0], attn_applied[0]), 1)      # (1x528)
    rnn_input = self.attention_U(joined_input).unsqueeze(0)          # (1x1x264)

    rnn_input = F.relu(rnn_input)   # should order be switched? TODO: is this needed?
    # in a GRU and LSTM the encoder output is the same as encoder hidden state
    # you are confusing hidden state with the LSTM cell state which is different
    output, final_hidden = self.gru(rnn_input, hidden_state)         # (1x1x264)

    # 2.1) score_g is unnormalized probability right before a softmax
    score_g = self.generate_mode(final_hidden).squeeze(1)     # [b x vocab_size]
    # enc_outs = encoder_outputs.contiguous().view(-1, self.hidden_size*2)
    # before: torch.Size([30, 264])   after: torch.Size([15, 528])
    # basically, pass through affine transform and a non-linearity
    # all the view and squeezing is just to line up the matrix multplication

    # 2.2) score_c is the score for the copy mode
    trimmed_outputs = encoder_outputs[:input_length, :]     # to remove padding
    score_c = F.tanh(self.copy_mode(trimmed_outputs))               # (8x264)
    score_c = score_c.view(batch_size, -1, self.hidden_size)        # (bx8x264)
    hidden = hidden_state.view(batch_size, self.hidden_size, 1)     # (bx264x1)
    score_c = torch.bmm(score_c, hidden).view(batch_size, -1)       # (bx8)

    # 2.3) get softmax-ed probabilities
    probs = F.softmax(torch.cat([score_g,score_c],1))              # (1x 1000)
    prob_g = probs[:,:self.vocab_size]                             # [b x vocab]
    prob_c = probs[:,self.vocab_size:]                             # [b x 8]

    # 2.4) append some OOV slots to the end pof prob_generate, i.e. 12 slots
    # oovs = smart_variable(torch.zeros((batch_size, 12)) )+1e-4
    # prob_g = torch.cat([prob_g,oovs], 1)                      # [b x vocab+12]

    # 2.5) add prob_c to prob_g
    input_indexes = input_variable.data.long().t().unsqueeze_(2)    # (bx8x1)
    one_hot = torch.zeros((batch_size, input_length, self.vocab_size))
    one_hot.scatter_(2, input_indexes, 1)                    # (b x 8 x 1000)
    # one_hot = one_hot.cuda()
    prob_c_to_g = torch.bmm(prob_c.unsqueeze(1), smart_variable(one_hot)) # [b x 1 x vocab]
    prob_c_to_g = prob_c_to_g.squeeze(1)                        # [b x vocab]
    final_output = prob_g + prob_c_to_g                         # [1 x 2165]

    # 3. get weighted attention to use for predicting next word
    # 3.1) tensor indicating whether decoder input appeared anywhere in encoder
    din = decoder_input.squeeze()                     # Variable scalar
    matched = torch.cat([(word==din) for word in input_variable]).float()
    # for i in range(batch_size):   not needed because we have batch_size 1
    #   total = matched[i].sum().data[0]
    #   matched[i] = matched[i]/total if total > 1 else matched[i]

    # 3.2) multiply with prob_c to get final weighted representation
    updated_attn = prob_c * matched                         # [b,8]x[8] = (bx8)
    final_weights = torch.mm(updated_attn, trimmed_outputs)      # [b x hidden]
    final_weights = final_weights.unsqueeze(1)                      # (bx1x264)

    if self.verbose:
      print("final_output: {}".format(final_output.size()) )
      print("final_hidden: {}".format(final_hidden.size()) )
      print("final_weights: {}".format(final_weights.size()) )

    return final_output, final_hidden, final_weights

class Match_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1,
        dropout_p=0.1, max_length=8):
    super(Match_Decoder, self).__init__()
    self.hidden_size = hidden_size + 8   # extended dim for the match features
    self.vocab_size = vocab_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.max_length = max_length
    self.embedding = nn.Embedding(vocab_size, self.hidden_size)

    self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers)
    self.out = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(self, input, hidden, encoder_outputs):
    if (hidden.size()[0] == (2 * input.size()[0])):
      hidden = hidden.view(1, 1, -1)

    embedded = self.embedding(input).view(1, 1, -1)
    embedded = self.dropout(embedded)
    attn_weights = F.softmax(
      self.attn(torch.cat((embedded[0], hidden[0]), 1)))
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                 encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    output = F.relu(output)
    output, hidden = self.gru(output, hidden)

    output = F.log_softmax(self.out(output[0]))
    return output, hidden, attn_weights

class Bid_GRU_Attn_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1,
        dropout_p=0.1, max_length=8):
    super(Bid_GRU_Attn_Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
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
    if (hidden.size()[0] == (2 * input.size()[0])):
      hidden = hidden.view(1, 1, -1)

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

class Bid_GRU_Decoder(nn.Module):
  '''
  During bi-directional encoding, we split up the word embedding in half
  and use then perform a forward pass into two directions.  In code,
  this is interpreted as 2 layers at half the size. Based on the way we
  produce the encodings, we need to merge the context vectors together in
  order properly init the hidden state, but then everything else is the same
  '''
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(Bid_GRU_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.input_size = hidden_size #serves double duty

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(self.input_size, self.hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    # if we are processing initial time step
    if (hidden.size()[0] == (2 * input.size()[0])):
      hidden = hidden.view(1, 1, -1)
    output = self.embedding(input).view(1, 1, -1)
    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden

class GRU_Attn_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1,
        dropout_p=0.1, max_length=8):
    super(GRU_Attn_Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size
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

    attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)

    output = F.log_softmax(self.out(output[0]))
    return output, hidden, attn_weights

class GRU_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(GRU_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.input_size = hidden_size #serves double duty

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.gru = nn.GRU(self.input_size, self.hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    output = self.embedding(input).view(1, 1, -1)
    # input: scalar, hidden: [1, 1, 256], output:[1, 1, 256]
    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.gru(output, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden

class LSTM_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(LSTM_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size

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
    return smart_variable(torch.zeros(1, 1, self.hidden_size))

class RNN_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(RNN_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.input_size = hidden_size

    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.rnn = nn.RNN(self.input_size, self.hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.softmax = nn.LogSoftmax()

  def forward(self, input, hidden):
    output = self.embedding(input).view(1, 1, -1)
    for i in range(self.n_layers):
      output = F.relu(output)
      output, hidden = self.rnn(output, hidden)
    output = self.softmax(self.out(output[0]))
    return output, hidden