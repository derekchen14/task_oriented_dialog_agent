# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division
from utils.internal.arguments import solicit_args
from io import open
import unicodedata
import string
import re
import random
import json
import pdb, sys
import time as tm
import pickle

import torch
from torch import optim
from torch.nn import NLLLoss as NegLL_Loss
from torch.optim.lr_scheduler import StepLR as StepLR

import utils.internal.data_io as data_io
import utils.internal.evaluate as evaluate
# from utils.internal.evaluate import LossTracker, Evaluator
from utils.internal.bleu import BLEU
from utils.external.clock import *
from utils.external.preprocessers import *
from model.components import *

MAX_LENGTH = 14
torch.manual_seed(14)

def train(input_variable, target_variable, encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion, teach_ratio):
  encoder.train()   # affects the performance of dropout
  decoder.train()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  loss, _, _ = run_inference(encoder, decoder, input_variable, target_variable, \
                        criterion, teach_ratio)
  loss.backward()
  clip_gradient([encoder, decoder], clip=10)
  encoder_optimizer.step()
  decoder_optimizer.step()

  return loss.item() / target_variable.shape[0]

def validate(input_variable, target_variable, encoder, decoder, criterion, task):
  encoder.eval()  # affects the performance of dropout
  decoder.eval()

  loss, predictions, visual = run_inference(encoder, decoder, input_variable, \
                    target_variable, criterion, teach_ratio=0)

  queries = input_variable.data.tolist()
  targets = target_variable.data.tolist()

  # when task is not specified, it defaults to index_to_label
  kind = "ordered_values"  # "possible_only", "full_enumeration"
  predicted_tokens = [vocab.index_to_word(predictions, kind)]
  query_tokens = [vocab.index_to_word(y, task) for y in queries]
  target_tokens = [vocab.index_to_word(z, kind) for z in targets]

  avg_loss = loss.item() / target_variable.shape[0]
  bleu_score = 1 # BLEU.compute(predicted_tokens, target_tokens)
  turn_success = (predictions.item() == targets[0])

  return avg_loss, bleu_score, turn_success

''' Modified since predictions are now single classes rather than sentences
predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
query_tokens = [vocab.index_to_word(y[0], task) for y in queries]
target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

turn_success = [pred == tar[0] for pred, tar in zip(predictions, targets)]
return avg_loss, bleu_score, all(turn_success)
'''

def track_progress(args, encoder, decoder, verbose, debug, train_data, val_data,
                  task, epochs, teacher_forcing=0.0, weight_decay=0.0):
  bleu_scores, accuracy = [], []
  learner = LossTracker(args.early_stopping)

  n_iters = 600 if debug else len(train_data)
  print_every, plot_every, val_every = print_frequency(verbose, debug)
  print_loss_total = 0  # Reset every print_every
  plot_loss_total = 0  # Reset every plot_every

  enc_optimizer, dec_optimizer = init_optimizers(args.optimizer, weight_decay,
        encoder.parameters(), decoder.parameters(), args.learning_rate)

  # training_pairs = [random.choice(train_data) for i in range(n_iters)]
  # validation_pairs = [random.choice(val_data) for j in range(v_iters)]
  criterion = NegLL_Loss()
  enc_scheduler = StepLR(enc_optimizer, step_size=n_iters/(args.decay_times+1), gamma=0.2)
  dec_scheduler = StepLR(dec_optimizer, step_size=n_iters/(args.decay_times+1), gamma=0.2)

  for epoch in range(epochs):
    start = tm.time()
    starting_checkpoint(epoch, epochs)
    for iteration, training_pair in enumerate(train_data):
      enc_scheduler.step()
      dec_scheduler.step()
      input_variable = training_pair[0]
      output_variable = training_pair[1]

      loss = train(input_variable, output_variable, encoder, decoder, \
             enc_optimizer, dec_optimizer, criterion, teach_ratio=teacher_forcing)
      print_loss_total += loss
      plot_loss_total += loss

      if iteration > 0 and iteration % print_every == 0:
        learner.train_steps.append(iteration + 1)
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0  # reset the print loss
        print('{1:3.1f}% complete {2}, Train Loss: {0:.4f}'.format(print_loss_avg,
            (iteration/n_iters * 100.0), timeSince(start, iteration/n_iters )))
        learner.update_loss(print_loss_avg, "train")

      if iteration > 0 and iteration % val_every == 0:
        learner.val_steps.append(iteration + 1)
        batch_val_loss, batch_bleu, batch_success = [], [], []
        for val_input, val_output in val_data:
          val_loss, bleu_score, turn_success = validate(val_input, \
                val_output, encoder, decoder, criterion, task)
          batch_val_loss.append(val_loss)
          batch_bleu.append(bleu_score)
          batch_success.append(turn_success)

        avg_val_loss, avg_bleu, avg_success = evaluate.batch_processing(
                                      batch_val_loss, batch_bleu, batch_success)
        learner.update_loss(avg_val_loss, "val")
        bleu_scores.append(avg_bleu)
        accuracy.append(avg_success)
        if learner.should_early_stop():
          print("Early stopped at val epoch {}".format(learner.val_epoch))
          learner.completed_training = False
          break

  time_past(start)
  return learner, bleu_scores, accuracy

if __name__ == "__main__":
  # ---- PARSE ARGS -----
  args = solicit_args()
  task = args.task_name
  # ----- LOAD DATA -----
  if args.debug:
    debug_data = pickle.load( open( "datasets/debug_data.pkl", "rb" ) )
    train_variables, val_variables, max_length = debug_data
  if args.test_mode:
    test_data, max_length = data_io.load_dataset(args.task_name, "test", args.debug)
    test_variables = prepare_examples(test_data, args.context, task=task)
    encoder = torch.load("results/enc_{0}_{1}.pt".format(args.model_path, args.suffix))
    decoder = torch.load("results/dec_{0}_{1}.pt".format(args.model_path, args.suffix))
    print("model loaded!")
    # show_dialogues(test_variables, encoder, decoder, task)
    # grab_attention(val_data, encoder, decoder, task, 3)
    # evaluate.show_save_attention(visualizations, args.attn_method, args.verbose)
    # results = test_mode_run(test_variables, encoder, decoder, task)
    # print("Done with visualizing.")
    criterion = NegLL_Loss()
    rate_of_success = []
    rate_of_loss = []
    for iteration, data_pair in enumerate(test_variables):
      if iteration % 31 == 0:
        test_input, test_output = data_pair
        loss, _, success = validate(test_input, test_output, encoder, decoder, criterion, task)
        if success:
          rate_of_success.append(1)
        else:
          rate_of_success.append(0)

        rate_of_loss.append(loss)
    ros = np.average(rate_of_success)
    rol = np.average(rate_of_loss)
    print("Loss: {} and Success: {:.3f}".format(rol, ros))
    sys.exit()
  else:
    train_data, max_length = data_io.load_dataset(args.task_name, "train", args.debug)
    train_variables = prepare_examples(train_data, args.context, task=task)
    val_data, _ = data_io.load_dataset(args.task_name, "dev", args.debug)
    val_variables = prepare_examples(val_data, args.context, task=task)
  # ---- BUILD MODEL ----
  print("Running model {0}_{1}".format(args.model_path, args.suffix))
  encoder, decoder = choose_model(args.model_type, vocab.ulary_size(task),
      args.hidden_size, args.attn_method, args.n_layers, args.drop_prob, max_length)
  # ---- TRAIN MODEL ----
  results = track_progress(args, encoder, decoder, args.verbose, args.debug,
      train_variables, val_variables, task, args.epochs,
      teacher_forcing=args.teacher_forcing, weight_decay=args.weight_decay)
  # --- MANAGE RESULTS ---
  if args.save_model and results[0].completed_training:
    torch.save(encoder, "results/enc_{0}_{1}.pt".format(args.model_path, args.suffix))
    torch.save(decoder, "results/dec_{0}_{1}.pt".format(args.model_path, args.suffix))
    print('Model saved at results/model_{}!'.format(args.model_path))
  if args.report_results and results[0].completed_training:
    evaluate.qual_report(encoder, decoder, val_variables, args)
    evaluate.quant_report(results, args)
  if args.plot_results and results[0].completed_training:
    evaluate.plot([strain, sval], [ltrain, lval], 'Training curve', 'Iterations', 'Loss')
  if args.visualize > 0:
    visualizations = grab_attention(val_variables, encoder, decoder, task, args.visualize)
    evaluate.show_save_attention(visualizations, args.attn_method, args.verbose)
