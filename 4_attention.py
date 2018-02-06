# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division
from utils.internal.arguments import solicit_args
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
from torch.optim.lr_scheduler import StepLR as StepLR

import utils.internal.data_io as data_io
import utils.internal.display_loss as display
from utils.external.clock import *
from utils.external.preprocessers import *
from model.components import *

from model.encoders import Match_Encoder, Bid_GRU_Encoder #GRU_Encoder # LSTM_Encoder, RNN_Encoder
from model.decoders import Match_Decoder, Bid_GRU_Attn_Decoder # GRU_Attn_Decoder # RNN_Attn_Decoder

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 8

def train(input_variable, target_variable, encoder, decoder, \
        encoder_optimizer, decoder_optimizer, criterion, max_length, teacher_forcing_ratio):
  encoder.train()   # affects the performance of dropout
  decoder.train()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  loss = 0

  # input_length = input_variable.size()[0]
  # encoder_outputs = smart_variable(torch.zeros(max_length, encoder.hidden_size))
  # for ei in range(min(max_length, input_length)):
  # encoder_outputs[ei] = encoder_output[0][0]
  target_length = target_variable.size()[0]

  encoder_hidden = encoder.initHidden()
  encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

  # encoder's last hidden state is the decoder's intial hidden state
  # last_enc_hidden_state = encoder_hidden[-1]
  decoder_input = smart_variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_hidden = encoder_hidden
  decoder_context = smart_variable(torch.zeros(1, decoder.hidden_size))

  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  if use_teacher_forcing:
    # Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs)

      loss += criterion(decoder_output, target_variable[di])
      decoder_input = target_variable[di]  # Teacher forcing
  else:
    for di in range(target_length):
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
          decoder_input, decoder_context, decoder_hidden, encoder_outputs)
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      decoder_input = smart_variable(torch.LongTensor([[ni]]))
      loss += criterion(decoder_output, target_variable[di])
      if ni == vocab.EOS_token:
        break

  loss.backward()
  clip_gradient([encoder, decoder], clip=5)
  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_length

def validate(input_variable, target_variable, encoder, decoder, criterion, max_length):
  encoder.eval()  # affects the performance of dropout
  decoder.eval()
  loss = 0

  target_length = target_variable.size()[0]
  encoder_hidden = encoder.initHidden()
  encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

  decoder_input = smart_variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_hidden = encoder_hidden
  decoder_context = smart_variable(torch.zeros(1, decoder.hidden_size))

  for di in range(target_length):
    decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs)

    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    decoder_input = smart_variable(torch.LongTensor([[ni]]))

    loss += criterion(decoder_output, target_variable[di])
    if ni == vocab.EOS_token :
      break

  return loss.data[0] / target_length


def track_progress(encoder, decoder, train_data, val_data, task, verbose, debug, \
      max_length=8, n_iters=75600, learning_rate=0.01, \
      teacher_forcing_ratio=0.0, weight_decay=0.0):
  start = tm.time()
  plot_losses_train = []
  plot_losses_validation = []
  plot_steps_train = []
  plot_steps_validation = []

  v_iters = len(val_data) if task == 'car' else int(len(val_data)/500) - 1
  n_iters = 600 if debug else n_iters
  print_every, plot_every, val_every = print_frequency(verbose, debug)
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

  encoder_optimizer, decoder_optimizer = init_optimizers(args.optimizer,
        encoder.parameters(), decoder.parameters(), learning_rate, weight_decay)

  training_pairs = [random.choice(train_data) for i in xrange(n_iters)]
  validation_pairs = [random.choice(val_data) for j in xrange(v_iters)]
  criterion = nn.NLLLoss()
  scheduler = StepLR(encoder_optimizer, step_size=n_iters/(args.decay_times+1), gamma=0.2)
  scheduler2 = StepLR(decoder_optimizer, step_size=n_iters / (args.decay_times+1), gamma=0.2)

  for iter in range(1, n_iters + 1):
    scheduler.step()
    scheduler2.step()

    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]
    output_variable = training_pair[1]

    if iter == 1: print("Starting to train ...")
    loss = train(input_variable, output_variable, encoder, decoder, \
           encoder_optimizer, decoder_optimizer, criterion, max_length, teacher_forcing_ratio=teacher_forcing_ratio)
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
      val_losses = []
      for iter in range(1, v_iters + 1):
        validation_pair = validation_pairs[iter - 1]
        val_input = validation_pair[0]
        val_output = validation_pair[1]
        validation_loss = validate(val_input, val_output, encoder, decoder, criterion, max_length)
        val_losses.append(validation_loss)
      print('Validation loss: {:2.4f}'.format(sum(val_losses) * 1.0 / len(val_losses)) )
      plot_losses_validation.append(sum(val_losses) * 1.0 / len(val_losses))

  time_past(start)
  return plot_losses_train, plot_losses_validation, plot_steps_train, plot_steps_validation

if __name__ == "__main__":
  # ---- PARSE ARGS -----
  args = solicit_args()
  task = 'car' if args.task_name in ['navigate', 'schedule', 'weather'] else 'res'
  # ----- LOAD DATA -----
  train_data, candidates, max_length = data_io.load_dataset(args.task_name, \
    "trn", args.debug)
  train_variables = collect_dialogues(train_data, task=task)
  val_data, val_candidates, _ = data_io.load_dataset(args.task_name, "dev", args.debug)
  val_variables = collect_dialogues(val_data, task=task)
  # ---- BUILD MODEL ----
  encoder = Match_Encoder(vocab.ulary_size(task), args.hidden_size)
  decoder = Match_Decoder(vocab.ulary_size(task), args.hidden_size,
      args.n_layers, args.drop_prob, max_length)
  # decoder.embedding.weight = encoder.embedding.weight
  # ---- TRAIN MODEL ----
  ltrain, lval, strain, sval = track_progress(encoder, decoder, train_variables,
      val_variables, task, args.verbose, args.debug, max_length, n_iters=args.n_iters,
      teacher_forcing_ratio=args.teacher_forcing, weight_decay=args.weight_decay)
  if args.debug: sys.exit()
  # --- MANAGE RESULTS ---
  errors = pd.DataFrame(data={'train_steps': strain, 'valid_steps': sval, 'train_error': ltrain, 'validation_error': lval})
  errors.to_csv(args.error_path, index=False)
  print('Error saved!')

  if args.save_results:
    torch.save(encoder, args.encoder_path)
    torch.save(decoder, args.decoder_path)
    print('Model saved!')
    errors = pd.DataFrame(data={'train_steps':strain, 'valid_steps':sval, 'train_error': ltrain, 'validation_error':lval})
    errors.to_csv(args.error_path, index=False)
    print('Error saved!')

  if args.plot_results:
    display.plot([strain, sval], [ltrain, lval], 'Training curve', 'Iterations', 'Loss')
