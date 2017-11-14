# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import json
import argparse
import sys
import time as tm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import utils.internal.data_io as data_io
import utils.internal.vocabulary as vocab
from utils.external.clock import *
from utils.external.preprocessers import *

from model.encoders import GRU_Encoder # LSTM_Encoder, RNN_Encoder
from model.decoders import GRU_Decoder # RNN_Attn_Decoder

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

def validate(input_variable, target_variable, encoder, decoder):
  encoder_hidden = encoder.initHidden()

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
    decoder_output, decoder_hidden, = decoder(
      decoder_input, decoder_hidden, encoder_output, encoder_outputs)
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    loss += criterion(decoder_output, target_variable[di])
    if ni == EOS_token:
      break

  return loss.data[0] / target_length

def track_progress(encoder, decoder, dialogs, n_iters, print_every=1000, \
      plot_every=100, val_every=150, learning_rate=0.01):
  start = tm.time()
  # plot_losses = []
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
  training_pairs = [random.choice(dialogs) for i in xrange(n_iters)]
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
    if iter % val_every == 0:
      print_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0
    # if iter % plot_every == 0:
    #   plot_loss_avg = plot_loss_total / plot_every
    #   plot_losses.append(plot_loss_avg)
    #   plot_loss_total = 0

  return plot_losses

def solicit_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
  parser.add_argument('-t', '--task-name', help='Choose the task to train on', \
    choices=['1','2','3','4','5','dstc','concierge','schedule','navigate','weather'])
  parser.add_argument('--hidden-size', default=256, type=int, help='Number of hidden units in each LSTM')
  parser.add_argument('-v', '--verbose', default=False, action='store_true', help='whether or not to have verbose prints')
  return parser.parse_args()

if __name__ == "__main__":
  # -- PARSE ARGUMENTS --
  args = solicit_args()
  # ----- LOAD DATA -----
  now = tm.time()
  train, candidates = data_io.load_dataset(args.task_name, "trn")
  development, candidates = data_io.load_dataset(args.task_name, "dev")
  train_variables = collect_dialogues(train)
  development_variables = collect_dialogues(development)
  time_past(now)
  sys.exit()
  # ---- BUILD MODEL ----
  # encoder = EncoderRNN(10, args.hidden_size)
  # decoder = DecoderRNN(args.hidden_size, 10, 1, dropout_p=0.1)
  # if use_cuda:
  #   encoder = encoder.cuda()
  #   decoder = decoder.cuda()
  # ---- TRAIN MODEL ----
  losses = track_progress(encoder, decoder, train_variables, \
         n_iters=75000, print_every=5000)
  # --- MANAGE RESULTS ---
  # showPlot(losses)