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
import pdb
import time as tm

import torch
from torch import optim
from torch.nn import NLLLoss as NegLL_Loss
from torch.optim.lr_scheduler import StepLR as StepLR

import utils.internal.data_io as data_io
import utils.internal.evaluate as evaluate
from utils.internal.bleu import BLEU
from utils.external.clock import *
from utils.external.preprocessers import *
from model.components import *

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 8

def train(input_variable, target_variable, encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion, teach_ratio):
  encoder.train()   # affects the performance of dropout
  decoder.train()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  loss, _ = run_inference(encoder, decoder, input_variable, target_variable, \
                        criterion, teach_ratio)
  loss.backward()
  clip_gradient([encoder, decoder], clip=10)
  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0] / target_variable.size()[0]

def validate(input_variable, target_variable, encoder, decoder, criterion,
          verbose, task):
  encoder.eval()  # affects the performance of dropout
  decoder.eval()

  loss, predictions = run_inference(encoder, decoder, input_variable, \
                    target_variable, criterion, teach_ratio=0)

  queries = input_variable.data.tolist()
  targets = target_variable.data.tolist()
  predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
  target_tokens = [vocab.index_to_word(y[0], task) for y in targets]

  avg_loss = loss.data[0] / target_variable.size()[0]
  bleu_score = BLEU.compute(predicted_tokens, target_tokens)
  turn_success = [pred == tar[0] for pred, tar in zip(predictions, targets)]

  return avg_loss, bleu_score, all(turn_success)

def track_progress(encoder, decoder, train_data, val_data, task, verbose, debug, \
      learning_rate=0.01, n_iters=75600, teacher_forcing=0.0, weight_decay=0.0):
  start = tm.time()
  train_steps, train_losses = [], []
  val_steps, val_losses = [], []
  bleu_scores, accuracy = [], []

  v_iters = len(val_data) if task == 'car' else int(len(val_data)/500)
  n_iters = 600 if debug else n_iters
  print_every, plot_every, val_every = print_frequency(verbose, debug)
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

  encoder_optimizer, decoder_optimizer = init_optimizers(args.optimizer,
        encoder.parameters(), decoder.parameters(), learning_rate, weight_decay)

  training_pairs = [random.choice(train_data) for i in range(n_iters)]
  validation_pairs = [random.choice(val_data) for j in range(v_iters)]
  criterion = NegLL_Loss()
  scheduler = StepLR(encoder_optimizer, step_size=n_iters/(args.decay_times+1), gamma=0.2)
  scheduler2 = StepLR(decoder_optimizer, step_size=n_iters / (args.decay_times+1), gamma=0.2)

  for iter in range(1, n_iters + 1):
    scheduler.step()
    scheduler2.step()

    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]
    output_variable = training_pair[1]

    starting_checkpoint(iter)
    loss = train(input_variable, output_variable, encoder, decoder, \
           encoder_optimizer, decoder_optimizer, criterion, teach_ratio=teacher_forcing)
    print_loss_total += loss
    plot_loss_total += loss

    if iter % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print('%d%% complete %s, Train Loss: %.4f' % ((iter / n_iters * 100),
          timeSince(start, iter / n_iters), print_loss_avg))
      # train_losses.append(print_loss_avg)
      train_losses.append(print_loss_avg)
      train_steps.append(iter)

    if iter % val_every == 0:
      val_steps.append(iter)
      batch_val_loss, batch_bleu, batch_success = [], [], []
      for iter in range(1, v_iters + 1):
        val_pair = validation_pairs[iter - 1]
        val_input = val_pair[0]
        val_output = val_pair[1]
        val_loss, bleu_score, turn_success = validate(val_input, val_output, \
            encoder, decoder, criterion, verbose, task)
        batch_val_loss.append(val_loss)
        batch_bleu.append(bleu_score)
        batch_success.append(turn_success)

      avg_val_loss, avg_bleu, avg_success = evaluate.batch_processing(
                                    batch_val_loss, batch_bleu, batch_success)
      val_losses.append(avg_val_loss)
      bleu_scores.append(avg_bleu)
      accuracy.append(avg_success)

  time_past(start)
  return train_steps, train_losses, val_steps, val_losses, bleu_scores, accuracy

if __name__ == "__main__":
  # ---- PARSE ARGS -----
  args = solicit_args()
  task = 'car' if args.task_name in ['navigate', 'schedule', 'weather'] else 'res'
  # ----- LOAD DATA -----
  train_data, candidates, max_length = data_io.load_dataset(args.task_name, "trn", args.debug)
  train_variables = collect_dialogues(train_data, task=task)
  val_data, val_candidates, _ = data_io.load_dataset(args.task_name, "dev", args.debug)
  val_variables = collect_dialogues(val_data, task=task)
  # ---- BUILD MODEL ----
  encoder, decoder = choose_model(args.model_type, vocab.ulary_size(task),
      args.hidden_size, args.attn_method, args.n_layers, args.drop_prob)
  # ---- TRAIN MODEL ----
  results = track_progress(encoder, decoder, train_variables, val_variables,
      task, args.verbose, args.debug, args.learning_rate, n_iters=args.n_iters,
      teacher_forcing=args.teacher_forcing, weight_decay=args.weight_decay)
  # --- MANAGE RESULTS ---
  if args.save_model:
    torch.save(encoder, args.encoder_path)
    torch.save(decoder, args.decoder_path)
    print('Model saved!')
  if args.report_results:
    evaluate.create_report(results, args)
  if args.plot_results:
    evaluate.plot([strain, sval], [ltrain, lval], 'Training curve', 'Iterations', 'Loss')
