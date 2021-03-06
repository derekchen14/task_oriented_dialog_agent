# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division
from io import open
import unicodedata
import string
import re
import random
import json
import sys
import time as tm
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import utils.internal.data_io as data_io
import utils.internal.vocabulary as vocab
import utils.internal.display as display
from utils.internal.arguments import solicit_args
from utils.external.clock import *
from utils.external.preprocessers import *

from model.encoders import Bid_GRU_Encoder, GRU_Encoder # LSTM_Encoder, RNN_Encoder
from model.decoders import Bid_GRU_Decoder, GRU_Decoder # RNN_Attn_Decoder

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

  for ei in range(min(max_length, input_length)):
    encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

  # encoder's last hidden state is the decoder's intial hidden state
  # last_enc_hidden_state = encoder_hidden[-1]
  decoder_input = Variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
  # encoder's last hidden state is the decoder's intial hidden state
  decoder_hidden = encoder_hidden

  for di in range(target_length):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]   # the index of the top predicted word
    # the [0][0] is simply to extract a scalar from a 1x1x1 3d tensor
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    loss += criterion(decoder_output, target_variable[di])
    if ni == vocab.EOS_token:
      break

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_length

def validate(input_variable, target_variable, encoder, decoder, criterion, max_length):
  encoder.eval()
  decoder.eval()

  encoder_hidden = encoder.initHidden()

  input_length = input_variable.size()[0]
  target_length = target_variable.size()[0]
  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
  loss = 0

  for ei in range(min(max_length, input_length)):
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

    loss += criterion(decoder_output, target_variable[di]) #############????
    if ni == vocab.EOS_token:
      break

  return loss.data[0] / target_length

def track_progress(encoder, decoder, train_data, val_data, task, max_length=8, \
      n_iters=75000, print_every=5000, plot_every=100, val_every=150, \
      learning_rate=0.01, ):
  start = tm.time()
  plot_losses_train = []
  plot_losses_validation = []
  plot_steps_train = []
  plot_steps_validation = []

  v_iters = len(val_data) if task == 'car' else int(len(val_data)/500) - 1
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

  if args.optimizer == 'SGD':
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

  elif args.optimizer == 'Adam':
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate * 0.01)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * 0.01)
  else:
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)

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
      # plot_losses_train.append(print_loss_avg)
      plot_losses_train.append(print_loss_avg)
      plot_steps_train.append(iter)

    if iter % val_every == 0:
        plot_steps_validation.append(iter)
        validation_losses = []
        for iter in range(1, v_iters + 1):
          validation_pair = validation_pairs[iter - 1]
          validation_input = validation_pair[0]
          validation_output = validation_pair[1]
          val_loss = validate(validation_input, validation_output, encoder, decoder, criterion, max_length)
          validation_losses.append(val_loss)
        print('Validation loss = ', sum(validation_losses) * 1.0 / len(validation_losses))
        plot_losses_validation.append(sum(validation_losses) * 1.0 / len(validation_losses))

  return plot_losses_train, plot_losses_validation, plot_steps_train, plot_steps_validation

if __name__ == "__main__":
  # ---- PARSE ARGS -----
  args = solicit_args()
  task = task_simplification(args.task_name)
  # ----- LOAD DATA -----
  train_data, candidates, max_length = data_io.load_dataset(args.task_name, \
    "trn", args.debug)
  train_variables = collect_dialogues(train_data, task)
  val_data, val_candidates, _ = data_io.load_dataset(args.task_name, "dev")
  val_variables = collect_dialogues(val_data, task)
  # ---- BUILD MODEL ----
  encoder = Bid_GRU_Encoder(vocab.ulary_size(task), args.hidden_size, use_cuda)
  decoder = Bid_GRU_Decoder(vocab.ulary_size(task), args.hidden_size, use_cuda, n_layers)
  # ---- TRAIN MODEL ----
  ltrain, lval, strain, sval = track_progress(encoder, decoder, train_variables,
      val_variables, task, max_length, n_iters=args.n_iters, print_every=150)
  # --- MANAGE RESULTS ---
  if args.save_results:
    torch.save(encoder, args.encoder_path)
    torch.save(decoder, args.decoder_path)
    print('Model saved!')
    errors = pd.DataFrame(data={'train_steps':strain, 'valid_steps':sval, 'train_error': ltrain, 'validation_error':lval})
    errors.to_csv(args.error_path, index=False)
    print('Error saved!')
  if args.plot_results:
    display.plot([strain, sval], [ltrain, lval], 'Training curve', 'Iterations', 'Loss')


