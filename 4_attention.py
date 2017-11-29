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

import utils.internal.data_io as data_io
import utils.internal.vocabulary as vocab
import utils.internal.display as display
from utils.external.clock import *
from utils.external.preprocessers import *

from model.encoders import GRU_Encoder # LSTM_Encoder, RNN_Encoder
from model.decoders import RNN_Attn_Decoder

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 8
n_layers = 1

def train(input_variable, target_variable, encoder, decoder, \
        encoder_optimizer, decoder_optimizer, criterion, max_length):
  encoder.train()
  decoder.train()

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

  decoder_input = Variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
  # encoder's last hidden state is the decoder's intial hidden state
  decoder_hidden = encoder_hidden

  for di in range(target_length):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    # decoder_input, decoder_hidden, last_enc_hidden_state, encoder_outputs)  To be used later for attention
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    loss += criterion(decoder_output, target_variable[di])
    if ni == vocab.EOS_token:
      break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_length

def validate(input_variable, target_variable, encoder, decoder, criterion):
  encoder.eval()
  decoder.eval()

  encoder_hidden = encoder.initHidden()
  input_length = input_variable.size()[0]
  target_length = target_variable.size()[0]
  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

  decoder_input = Variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
  # encoder's last hidden state is the decoder's intial hidden state
  decoder_hidden = encoder_hidden

  for di in range(target_length):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    # decoder_input, decoder_hidden, last_enc_hidden_state, encoder_outputs)  To be used later for attention
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    loss += criterion(decoder_output, target_variable[di])
    if ni == vocab.EOS_token:
      break

  return loss.data[0] / target_length

def track_progress(encoder, decoder, train_data, val_data, max_length=8, \
      n_iters=75000, print_every=5000, plot_every=100, val_every=50, \
      learning_rate=0.01):
  start = tm.time()
  # plot_losses = []
  v_iters = int(len(val_data)/500) - 1

  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

  encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
  training_pairs = [random.choice(train_data) for i in xrange(n_iters)]
  validation_pairs = [random.choice(val_data) for j in xrange(v_iters)]
  criterion = nn.NLLLoss()

  for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]
    output_variable = training_pair[1]

    loss = train(input_variable, output_variable, encoder, decoder, \
           encoder_optimizer, decoder_optimizer, criterion, max_length)
    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%d%% complete %s, Train Loss: %.4f' % ((iter / n_iters * 100),
          timeSince(start, iter / n_iters), print_loss_avg))

    if iter % val_every == 0:
        # for iter in range(1, v_iters + 1):
        validation_pair = validation_pairs
        val_loss = validate(input_variable, output_variable, encoder, decoder, criterion)
        print('Validation loss = ', val_loss)


    # if iter % val_every == 0:
    #   for iter in range(1, v_iters + 1):
    #     training_pair = training_pairs[iter - 1]
    #     input_variable = training_pair[0]
    #     output_variable = training_pair[1]
    #     val_loss = validate(input_variable, output_variable, encoder,
    #          decoder, encoder_optimizer, decoder_optimizer, criterion)
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
  train_data, candidates, m = data_io.load_dataset(args.task_name, "trn")
  val_data, val_candidates, _ = data_io.load_dataset(args.task_name, "dev")

  train_variables = collect_dialogues(train_data)
  val_variables = collect_dialogues(val_data)

  # ---- BUILD MODEL ----
  encoder = GRU_Encoder(vocab.ulary_size(), args.hidden_size, use_cuda)
  decoder = RNN_Attn_Decoder(vocab.ulary_size(), args.hidden_size, use_cuda, n_layers, attn_param)
  # ---- TRAIN MODEL ----
  max_length = m if m else MAX_LENGTH
  losses = track_progress(encoder, decoder, train_variables, val_variables, max_length, n_iters=750, print_every=50)
  # --- MANAGE RESULTS ---
  # display.plot_results(losses)
  # save_results(losses)
