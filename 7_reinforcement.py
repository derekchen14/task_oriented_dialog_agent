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
import utils.internal.display as display
from utils.external.clock import *
from utils.external.preprocessers import *

from model.encoders import GRU_Encoder #GRU_Encoder # LSTM_Encoder, RNN_Encoder
from model.decoders import GRU_Attn_Decoder # GRU_Attn_Decoder # RNN_Attn_Decoder

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 8

def evaluate(input_variable, target_variable, encoder, decoder, max_length):
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

  predicts = []
  print("target: {}".format(target_length) )
  for di in range(target_length):
    decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, \
      encoder_output, encoder_outputs)

    # decoder_input, decoder_hidden, last_enc_hidden_state, encoder_outputs)  To be used later for attention
    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    decoder_input = Variable(torch.LongTensor([[ni]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    predicts.append(ni)
    # decoder_output = predicted output distribution (1x2165) Float Tensor
    # target_variable[di] = correct word? Long Tensor (scalar 1)

  target = target_variable.data.tolist()
  accuracy = 0
  for pred, tar in zip(predicts, targets):
    print(pred, tar)
    print(pred == tar)
    if pred == tar:
      accuracy += 1
  sys.exit()

  turn_success = accuracy == (target_length)
  return turn_success

if __name__ == "__main__":
  # ---- PARSE ARGS -----
  args = solicit_args()
  task = 'car' if args.task_name in ['navigate', 'schedule', 'weather'] else 'res'
  # ----- LOAD DATA -----
  val_data, val_candidates, max_length = data_io.load_dataset(args.task_name, "dev", False)
  val_variables = collect_dialogues(val_data, task=task)[0:1000]
  # ---- BUILD MODEL ----
  encoder = torch.load('datasets/old_encoder.pt')
  decoder = torch.load('datasets/old_decoder.pt')
  # ------ EVALUATE ------
  per_turn_success = []
  per_dialog_success = []
  for pair in val_variables:
     turn_success = evaluate(pair[0], pair[1], encoder, decoder, max_length)
     per_turn_accuracy.append(turn_success)
     # if new_dialog:
     #    per_dialog_success.append(acc_so_far)

  per_turn_accuracy = float(per_turn_success) / len(val_variables)
  per_dialog_accuracy = float(per_dialog_success) / 300
  print round(per_turn_accuracy, 3)