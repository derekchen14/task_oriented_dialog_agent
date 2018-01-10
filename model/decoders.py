# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

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
        dropout_p=0.1, max_length=8):
    super(Copy_Decoder, self).__init__()
    self.vocab_size = vocab_size  # check extend_vocab method below
    self.hidden_size = hidden_size + 8
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.max_length = max_length

    self.attention_W = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attention_U = nn.Linear(self.hidden_size * 2, self.hidden_size)

    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    self.copy_mode = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.generate_mode = nn.Linear(self.hidden_size, self.vocab_size)

  def extend_vocab(self, input_sentence):
    num_words_to_be_copied = len(input_sentence)
    self.vocab_size += num_words_to_be_copied
    self.out = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(self, input, hidden, encoder_outputs):
    # encoder_outputs     (input_max_length x hidden_size + 8) (30, 264)
    input_sentence_length = encoder_outputs.size()[0]
    print input_sentence_length
    sys.exit()

    if (hidden.size()[0] == (2 * input.size()[0])):
      hidden = hidden.view(1, 1, -1)

    embedded = self.embedding(input).view(1, 1, -1)
    embedded = self.dropout(embedded)

    attn_input = torch.cat((embedded[0], hidden[0]), 1)
    # context_vector = softmax(W(i+h))
    attn_weights = F.softmax(self.attention_W(attn_input))
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
    # attn = context_vector x encoder_outputs
    joined_input = torch.cat((embedded[0], attn_applied[0]), 1)
    # combine = U(i+attn)
    rnn_input = self.attention_U(joined_input).unsqueeze(0)

    rnn_input = F.relu(rnn_input)   # should order be switched? TODO: is this needed?
    # in a GRU and LSTM the encoder output is the same as encoder hidden state
    # you are confusing hidden state with the LSTM cell state which is different
    output, hidden = self.gru(rnn_input, hidden)
    # thus, the "output" here can be ignored

    '''
    # score_g is unnormalized probability right before a softmax
    score_g = self.generate_mode(hidden) # [b x vocab_size]
    # contiguous operates on computer memory, it does not change the math
    enc_outs = encoder_outputs.contiguous().view(-1,hidden_size*2)
    # basically, pass through affine transform and a non-linearity
    score_c = F.tanh(self.copy_mode(enc_outs)) # [b*seq x hidden_size]
    # all the view and squeezing is just to line up the matrix multplication
    score_c = score_c.view(b,-1,hidden_size) # [b x seq x hidden_size]
    # reshaping doesn't change the math, so feel free to ignore its impact
    score_c = torch.bmm(score_c, hidden.unsqueeze(2)).squeeze() # [b x seq]

    # TODO: find out if we have padding, and if so, what is its index?
    # below assumes padding_idx = 0
    # encoded_mask = (np.array(encoded_idx==padding_idx, dtype=float)*(-1000)) # [b x seq]
    # padded parts will get close to 0 when applying softmax
    # score_c = score_c + smart_variable(encoded_mask)
    # score_c = F.tanh(score_c)

    # 2-3) get softmax-ed probabilities
    score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
    probs = F.softmax(score)
    prob_g = probs[:,:vocab_size] # [b x vocab]
    prob_c = probs[:,vocab_size:] # [b x seq]

    # combined prob is size of the original_vocab + OOV
    combined_prob = smart_variable(torch.zeros((b,vocab_size)) )
    assume batch size of 1
    for s in range(input_sentence_length):  # for each word in the input sentence
      word_index = encoded_idx[s]
      some_zero_vector = combined_prob[word_index]
      combined_prob[word_index] += prob_c[s]
    out = prob_g + combined_prob
    out = out.unsqueeze(1) # [b x 1 x vocab]

    output = F.log_softmax(self.out(output[0]))
    '''
    return output, hidden #, attn_weights

class Match_Decoder(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers=1,
        dropout_p=0.1, max_length=8):
    super(Match_Decoder, self).__init__()
    self.hidden_size = hidden_size + 8
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

    for i in range(self.n_layers):
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
    self.use_cuda = use_cuda

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