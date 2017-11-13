# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import json
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import utils.internal.data_io as data_io
from utils.external.clock import *
from utils.external.preprocessers import *

from model.encoders import RNN_Encoder # LSTM_Encoder, GRU_Encoder
from model.decoders import RNN_Decoder # RNN_Attn_Decoder

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 8

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
  encoder_hidden = encoder.initHidden()

  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  input_length = input_variable.size()[0]
  target_length = target_variable.size()[0]
  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(
      input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

  decoder_input = Variable(torch.LongTensor([[SOS_token]]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
  decoder_hidden = encoder_hidden

  for di in range(target_length):
    decoder_output, decoder_hidden, decoder_attention = decoder(
      decoder_input, decoder_hidden, encoder_output, encoder_outputs)
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    loss += criterion(decoder_output, target_variable[di])
    if ni == EOS_token:
      break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_length

def track_progress(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
  start = time.time()
  # plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  training_pairs = [variablesFromPair(random.choice(pairs))
            for i in range(n_iters)]
  criterion = nn.NLLLoss()

  for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    loss = train(input_variable, target_variable, encoder,
           decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                     iter, iter / n_iters * 100, print_loss_avg))
    # if iter % plot_every == 0:
    #   plot_loss_avg = plot_loss_total / plot_every
    #   plot_losses.append(plot_loss_avg)
    #   plot_loss_total = 0

  return plot_losses

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
  parser.add_argument('-t', '--task-name', help='Choose the task to train on', \
    choices=['one','two','three','four','five','dstc','concierge','schedule','navigate','weather'])
  parser.add_argument('--hidden-size', default=256, type=int, help='Number of hidden units in each LSTM')
  parser.add_argument('-v', '--verbose', default=False, action='store_true', help='whether or not to have verbose prints')
  args = parser.parse_args()
  # ----- LOAD DATA -----
  trial = data_io.dialog_to_vec("tasty")
  # dataset = data_io.load_dataset(args.task_name)
  # ---- BUILD MODEL ----
  # encoder = EncoderRNN(10, args.hidden_size)
  # decoder = DecoderRNN(args.hidden_size, 10, 1, dropout_p=0.1)
  # if use_cuda:
  #   encoder = encoder.cuda()
  #   decoder = decoder.cuda()
  # ---- TRAIN MODEL ----
  # losses = track_progress(encoder, decoder, 75000, print_every=5000)
  # --- MANAGE RESULTS ---
  # showPlot(losses)

'''
def forward(self, input, hidden, encoder_outputs):
  embedded = self.embedding(input).view(1, 1, -1)
  embedded = self.dropout(embedded)

  # concat takes two (1x128) and joins together to become (1x256)
  score_input = torch.cat((embedded[0], hidden[0]), 1)

  attention_score1 = self.attn_Bahdanau(score_input)    # https://arxiv.org/abs/1409.0473
  # attention_score2 = self.attn_Luong(score_input)       # https://arxiv.org/abs/1508.04025
  # attention_score3 = self.attn_Vinyals(score_input)     # https://arxiv.org/abs/1412.7449
  # the attn is an affine that goes from (1x128) to (1x9) assuming we have
  # chosen a maximum sequence length of 9 words
  attn_weights = self.softmax( attention_score1 )
  # performs batch matrix-matrix product, we know the attention weights are (1x9)
  # from above, which implies the encoder outputs are (9x128) (with seq_len of
  # 9 and hidden size of 128) then context_vector will be (1x128) again
  attention_weights = attn_weights.unsqueeze(0)  # a_t
  encoder_outputs = encoder_outputs.unsqueeze(0)  # h_s or h-bar, s = source
  context_vector = torch.bmm(attention_weights, encoder_outputs) # c_t

  # by concat, we once again go from two (1x128) vectors to (1x256)
  output = torch.cat((embedded[0], context_vector[0]), 1)
  # attn_combine shrinks this down to half the size, which is (1x128) again
  output = self.attn_combine(output).unsqueeze(0)

  for i in range(self.n_layers):
    # decoder_input = F.relu(output) # (or tanH)
    decoder_input = output
    output, hidden = self.gru(decoder_input, hidden)

  # self.out is a Linear to reshape to the number of classes to predict
  output = self.log_softmax(self.out(output[0]))
  return output, hidden, attn_weights
'''