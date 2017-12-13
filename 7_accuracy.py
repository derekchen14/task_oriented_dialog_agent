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

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.internal.data_io as data_io
from utils.external.clock import *
from utils.external.preprocessers import *

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 8

def evaluate(input_variable, target_variable, encoder, decoder, max_length):
  encoder_hidden = encoder.initHidden()

  input_length = input_variable.size()[0]
  target_length = target_variable.size()[0]
  encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

  for ei in range(min(max_length, input_length)):
    encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

  decoder_input = Variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_input = decoder_input.cuda() if use_cuda else decoder_input
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

  targets = target_variable.data.tolist()
  turn_success = [pred == tar for pred, tar in zip(predicts, targets)]
  return all(turn_success)

if __name__ == "__main__":
  args = solicit_args()
  task = 'car' if args.task_name in ['navigate', 'schedule', 'weather'] else 'res'
  val_data, val_candidates, max_length = data_io.load_dataset(args.task_name, "dev", False)
  val_variables = collect_dialogues(val_data, task=task)[0:10]
  encoder = torch.load('datasets/old_encoder.pt')
  decoder = torch.load('datasets/old_decoder.pt')

  # single_dialog_success and single_turn_success are scalars with values 0 or 1
  # dialog_success and turn_success are lists that hold "single success"
  # overall_dialog and overall_turn are lists that hold accuracy totals
  # per_dialog_accuracy and per_turn_accuracy are floats between 0% to 100%
  overall_dialog = []
  overall_turn = []
  dialog_success = None

  for input_var, output_var in val_variables:
    first_token = input_var[0].data[0]
    # first token is a turn counter, so when it equals 1, that means we are in a new dialogue
    if first_token == 1 and dialog_success:
      single_dialog_success = all(dialog_success)
      overall_dialog.append(single_dialog_success)
      dialog_success = []

    single_turn_success = evaluate(input_var, output_var, encoder, decoder, max_length)
    dialog_success.append(single_turn_success)
    overall_turn.append(single_turn_success)

  per_turn_accuracy = float(sum(overall_turn)) / len(overall_turn)
  print("per_turn_accuracy: {}".format(round(per_turn_accuracy, 3)) )
  per_dialog_accuracy = float(sum(overall_dialog)) / len(overall_dialog)
  print("per_dialog_accuracy: {}".format(round(per_dialog_accuracy, 3)) )
