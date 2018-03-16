# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pdb  # set_trace

from model.components import smart_variable
from model.modules import Attention, Transformer
from utils.external.preprocessers import match_embedding

# ------- Decoders ----------
# Decoder is given an input token and hidden state. The initial input token is
# the start-of-string <SOS> token, and the first hidden state is the context
# vector (the encoder's last hidden state, not the last output!).

class Transformer_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=6):
    super(Transformer_Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.arguments_size = "small"

    self.embedding = nn.Embedding(vocab_size, self.hidden_size) # will be replaced
    self.transformer = Transformer(vocab_size, hidden_size, n_layers, True)
    self.out = nn.Linear(hidden_size, vocab_size)

  def forward(self, word_inputs, encoder_outputs, di):
    embedded = self.embedding(word_inputs)
    # dropped_embed = self.dropout(embedded)
    transformer_output = self.transformer(embedded, encoder_outputs, di)
    final_output = F.log_softmax(self.out(transformer_output), dim=1)  # (1x2N) (2NxV) = 1xV
    return final_output

class Copy_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, attn_method, drop_prob, max_length):
    super(Copy_Decoder, self).__init__()
    self.hidden_size = hidden_size + 8
    self.input_size = self.hidden_size * 2
    self.vocab_size = vocab_size  # check extend_vocab method below
    self.dropout = nn.Dropout(drop_prob)
    self.max_length = max_length
    self.arguments_size = "large"

    # max_length is added to help train the location based addressing
    self.embedding = nn.Embedding(vocab_size, self.hidden_size + max_length)
    self.gru = nn.GRU(self.input_size + max_length, self.hidden_size)
    self.attn = Attention(attn_method, self.hidden_size)
    self.out = nn.Linear(self.input_size, vocab_size) # will be replaced

    self.copy_mode = nn.Linear(self.hidden_size, self.hidden_size)
    self.generate_mode = nn.Linear(self.input_size, self.vocab_size)

  def extend_vocab(self, input_sentence):
    num_words_to_be_copied = len(input_sentence)
    self.vocab_size += num_words_to_be_copied
    self.out = nn.Linear(self.input_size, self.vocab_size)

  def forward(self, word_input, prev_context, prev_hidden, encoder_outputs, \
        sources, targets, ti, use_teacher_forcing):
    batch_size = 1      # prev_hidden: [b,1,264]  encoder_outputs: [9,1,264]
    if (prev_hidden.size()[0] == (2 * word_input.size()[0])):
      prev_hidden = prev_hidden.view(batch_size, 1, -1)

    # ------------ Perform Embedding for State Update -------------------------
    embedded = self.embedding(word_input).view(batch_size, 1, -1)
    embedded = self.dropout(embedded)
    # 0a) Get index of the decoder input: y_{t-1} (representing what would be
    #     the correct prediction in the previous timestep).  Alternatively,
    #     if we are not using teacher forcing, then get the predicted word
    di = targets[ti].squeeze() if use_teacher_forcing else word_input.squeeze()
    # 0b) Tensor indicating whether decoder word appeared in the encoder words
    #     (ie. whether or not we should have performed a copy action)
    locations = torch.cat([(word==di) for word in sources]).float()
    locations = smart_variable(locations, dtype="var")            # a list (7,)
    # default of dim=1 will throw error since "locations" only has one dimension
    embedded[:, :, :len(locations)] = F.normalize(locations, p=2, dim=0)

    # 1a) Calculate basic decoder output
    rnn_input = torch.cat((embedded, prev_context), dim=2)
    rnn_output, current_hidden = self.gru(rnn_input, prev_hidden)
    # 1b) Apply attention score to encoder outputs to get attention context
    decoder_hidden = current_hidden.squeeze(0)
    attn_weights = self.attn(decoder_hidden, encoder_outputs)  # 1 x 1 x S
    attn_context = attn_weights.bmm(encoder_outputs.transpose(0,1))
    # 1c) Join the results of decoder hidden state and Attention context
    joined_hidden = torch.cat((current_hidden, attn_context), dim=2).squeeze(0)

    # 2a) Get a score for generating text from vocabulary
    score_g = self.generate_mode(joined_hidden)              # (b x vocab_size)
    # 2b) Get a score for copying text from user utterance   assume seq_len = 7
    c_part = F.tanh(self.copy_mode(encoder_outputs))                  # (7x264)
    c_part = c_part.view(batch_size, -1, self.hidden_size)          # (bx7x264)
    h_part = prev_hidden.transpose(1,2)                  # (bx1x264 => bx264x1)
    score_c = torch.bmm(c_part, h_part).squeeze(2)             # (bx7x1 => bx7)

    # 3a) Normalize probabilities with softmax
    probs = F.log_softmax(torch.cat([score_g,score_c], 1), dim=1)    # (b, v+7)
    # 3b) Split the probability up for each option
    prob_g = probs[:,:self.vocab_size]              # (batch_size x vocab_size)
    prob_c = probs[:,self.vocab_size:]                 # (batch_size x seq_len)

    # ----------- Merge COPY-prob and GENERATE-prob together --------------
    # 4a) Extract indexes of words in the encoder input sequence
    input_indexes = sources.data.cpu().t().unsqueeze_(2)    # (b,7,1)
    # pdb.set_trace()
    batch_size, seq_len, _ = input_indexes.size()
    # 4b) Create a vector of zeros to store all the encoder indices
    # Only necessary because we might multiple identical words in the encoder
    one_hot = torch.zeros((batch_size, seq_len, self.vocab_size))
    # 4c) In the third dimension, put ones where the input words show up
    one_hot.scatter_(2, input_indexes, 1)                           # (b,7,v)
    # 4d) Pull out and sum together the encoder words probabilities
    # (b,1,7) x (b,7,v)  => (b x 1 x vocab_size)
    carrier = torch.bmm(prob_c.unsqueeze(1), smart_variable(one_hot))
    # 4e) Transfer probabilities from copy to generate
    #        (b,v)  +   (b,1,v => b,v)
    output = prob_g + carrier.squeeze(1)          # (batch_size, vocab_size)

    return output, attn_context, current_hidden, attn_weights

    ''' Merge the COPY-prob into the GENERATE prob
    We slightly simplify by assuming that while there may be OOV (words that
    appear in the encoder during testing that never appeared anywhere during
    training), we live in a world where there are no UNKs in the vocabulary.
    Thus, there is no need to extend the vocabulary size of prob_g

    max_num_oov = 12  # we think that at most there will be 12 OOV terms
    oovs = smart_variable(torch.zeros((batch_size, max_num_oov)) )+1e-4
    prob_g = torch.cat([prob_g, oovs], 1)                     # [b x vocab+12]

    We also simplify when calculating the state update by assuming a batch
    size of one for all training examples

    for i in range(batch_size):   not needed because we have batch_size 1
      total = matched[i].sum().data[0]
      matched[i] = matched[i]/total if total > 1 else matched[i]
    '''

class Match_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, method, drop_prob=0.1):
    super(Match_Decoder, self).__init__()
    self.hidden_size = hidden_size + 8   # extended dim for the match features
    self.input_size = self.hidden_size * 2
    self.vocab_size = vocab_size
    self.dropout = nn.Dropout(drop_prob)
    self.arguments_size = "medium"

    self.embedding = nn.Embedding(vocab_size, self.hidden_size) # will be replaced
    self.gru = nn.GRU(self.input_size, self.hidden_size)
    self.attn = Attention(method, self.hidden_size)
    self.out = nn.Linear(self.input_size, vocab_size)

  def forward(self, word_input, last_context, prev_hidden, encoder_outputs):
    if (prev_hidden.size()[0] == (2 * word_input.size()[0])):
      prev_hidden = prev_hidden.view(1, 1, -1)

    embedded = self.embedding(word_input).view(1, 1, -1)        # 1 x 1 x N
    embedded = self.dropout(embedded)
    rnn_input = torch.cat((embedded, last_context), dim=2)
    rnn_output, current_hidden = self.gru(rnn_input, prev_hidden)

    decoder_hidden = current_hidden.squeeze(0)
    attn_weights = self.attn(decoder_hidden, encoder_outputs)  # 1 x 1 x S
    attn_context = attn_weights.bmm(encoder_outputs.transpose(0,1))

    joined_hidden = torch.cat((current_hidden, attn_context), dim=2).squeeze(0)
    output = F.log_softmax(self.out(joined_hidden), dim=1)  # (1x2N) (2NxV) = 1xV
    return output, attn_context, current_hidden, attn_weights

class Bid_Decoder(nn.Module):
  '''
  During bi-directional encoding, we split up the word embedding in half
  and use then perform a forward pass into two directions.  In code,
  this is interpreted as 2 layers at half the size. Based on the way we
  produce the encodings, we need to merge the context vectors together in
  order properly init the hidden state, but then everything else is the same
  '''
  def __init__(self, vocab_size, hidden_size, method, drop_prob=0.1):
    super(Bid_Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = self.hidden_size * 2  # since we concat input and context
    self.vocab_size = vocab_size                # |V| is around 2100
    self.dropout = nn.Dropout(drop_prob)
    self.arguments_size = "medium"
    # num_layers is removed since decoder always has one layer
    self.embedding = nn.Embedding(vocab_size, self.hidden_size) # will be replaced
    self.gru = nn.GRU(self.input_size, self.hidden_size) # dropout=drop_prob)
    self.attn = Attention(method, self.hidden_size)  # adds W_a matrix
    self.out = nn.Linear(self.hidden_size * 2, vocab_size)
    # we need "* 2" since we concat hidden state and attention context vector

  def forward(self, word_input, last_context, prev_hidden, encoder_outputs):
    if (prev_hidden.size()[0] == (2 * word_input.size()[0])):
      prev_hidden = prev_hidden.view(1, 1, -1)

    # Get the embedding of the current input word (i.e. last output word)
    embedded = self.embedding(word_input).view(1, 1, -1)        # 1 x 1 x N
    embedded = self.dropout(embedded)
    # Combine input word embedding and previous hidden state, run through RNN
    rnn_input = torch.cat((embedded, last_context), dim=2)
    pdb.set_trace()
    rnn_output, current_hidden = self.gru(rnn_input, prev_hidden)

    # Calculate attention from current RNN state and encoder outputs, then apply
    # Drop first dimension to line up with single encoder_output
    decoder_hidden = current_hidden.squeeze(0)    # (1 x 1 x N) --> 1 x N
    attn_weights = self.attn(decoder_hidden, encoder_outputs)  # 1 x 1 x S
     # [1 x (1xS)(SxN)] = [1 x (1xN)] = 1 x 1 x N)   where S is seq_len of encoder
    attn_context = attn_weights.bmm(encoder_outputs.transpose(0,1))

    # Predict next word using the decoder hidden state and context vector
    joined_hidden = torch.cat((current_hidden, attn_context), dim=2).squeeze(0)
    output = F.log_softmax(self.out(joined_hidden), dim=1)  # (1x2N) (2NxV) = 1xV
    return output, attn_context, current_hidden, attn_weights

class Attn_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, method, drop_prob=0.1):
    super(Attn_Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = self.hidden_size * 2  # since we concat input and context
    self.vocab_size = vocab_size                # |V| is around 2100
    self.dropout = nn.Dropout(drop_prob)
    self.arguments_size = "medium"

    self.embedding = nn.Embedding(vocab_size, self.hidden_size) # will be replaced
    self.gru = nn.GRU(self.input_size, self.hidden_size) # dropout=drop_prob)
    self.attn = Attention(method, self.hidden_size)  # adds W_a matrix
    self.out = nn.Linear(self.hidden_size * 2, vocab_size)
    # we need "* 2" since we concat hidden state and attention context vector

  def forward(self, word_input, last_context, prev_hidden, encoder_outputs):
    # Get the embedding of the current input word (i.e. last output word)
    embedded = self.embedding(word_input).view(1, 1, -1)        # 1 x 1 x N
    embedded = self.dropout(embedded)
    # Combine input word embedding and previous hidden state, run through RNN
    rnn_input = torch.cat((embedded, last_context), dim=2)
    # pdb.set_trace()
    rnn_output, current_hidden = self.gru(rnn_input, prev_hidden)

    # Calculate attention from current RNN state and encoder outputs, then apply
    # Drop first dimension to line up with single encoder_output
    decoder_hidden = current_hidden.squeeze(0)    # (1 x 1 x N) --> 1 x N
    attn_weights = self.attn(decoder_hidden, encoder_outputs)  # 1 x 1 x S
     # [1 x (1xS)(SxN)] = [1 x (1xN)] = 1 x 1 x N)   where S is seq_len of encoder
    attn_context = attn_weights.bmm(encoder_outputs.transpose(0,1))

    # Predict next word using the decoder hidden state and context vector
    joined_hidden = torch.cat((current_hidden, attn_context), dim=2).squeeze(0)
    output = F.log_softmax(self.out(joined_hidden), dim=1)  # (1x2N) (2NxV) = 1xV
    return output, attn_context, current_hidden, attn_weights

class GRU_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1):
    super(GRU_Decoder, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.input_size = hidden_size # serves double duty
    self.arguments_size = "medium"

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
    self.arguments_size = "medium"

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
    self.arguments_size = "medium"

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