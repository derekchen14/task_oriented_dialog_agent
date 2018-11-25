# -*- coding: utf-8 -*-
import random
import pdb, sys
import time as tm

import torch
from torch import optim
from torch.nn import NLLLoss as NegLL_Loss
from torch.optim.lr_scheduler import StepLR as StepLR

from utils.external.bleu import BLEU
from utils.internal.arguments import solicit_args
from utils.internal.clock import *

from model.components import *
from model.preprocess import PreProcessor
# from modules.learn import Builder, Learner
from model.evaluate import LossTracker, Evaluator

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

def track_progress(args, encoder, decoder, verbose, debug, processor, task,
              epochs, weight_decay=0.0):
  bleu_scores, accuracy = [], []
  tracker = LossTracker(args.early_stopping)

  n_iters = 600 if debug else len(processor.train_data)
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
    starting_checkpoint(epoch, epochs, use_cuda)
    for iteration, training_pair in enumerate(processor.train_data):
      enc_scheduler.step()
      dec_scheduler.step()
      input_variable = training_pair[0]
      output_variable = training_pair[1]

      loss = train(input_variable, output_variable, encoder, decoder, \
             enc_optimizer, dec_optimizer, criterion, teach_ratio=args.teacher_forcing)
      print_loss_total += loss
      plot_loss_total += loss

      if iteration > 0 and iteration % print_every == 0:
        tracker.train_steps.append(iteration + 1)
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0  # reset the print loss
        print('{1:3.1f}% complete {2}, Train Loss: {0:.4f}'.format(print_loss_avg,
            (iteration/n_iters * 100.0), timeSince(start, iteration/n_iters )))
        tracker.update_loss(print_loss_avg, "train")

      if iteration > 0 and iteration % val_every == 0:
        tracker.val_steps.append(iteration + 1)
        batch_val_loss, batch_bleu, batch_success = [], [], []
        for val_input, val_output in processor.val_data:
          val_loss, bleu_score, turn_success = validate(val_input, \
                val_output, encoder, decoder, criterion, task)
          batch_val_loss.append(val_loss)
          batch_bleu.append(bleu_score)
          batch_success.append(turn_success)

        avg_val_loss, avg_bleu, avg_success = Evaluator.batch_processing(
                                      batch_val_loss, batch_bleu, batch_success)
        tracker.update_loss(avg_val_loss, "val")
        bleu_scores.append(avg_bleu)
        accuracy.append(avg_success)
        if tracker.should_early_stop():
          print("Early stopped at val epoch {}".format(tracker.val_epoch))
          tracker.completed_training = False
          break

  time_past(start)
  return tracker, bleu_scores, accuracy

if __name__ == "__main__":
  args = solicit_args()
  # ----- LOAD MODULES -----
  processor = PreProcessor(args)
  # builder = Builder(args)
  # learner = Learner(args)
  evaluator = Evaluator(args)

  if args.debug:
    debug_data = pickle_loader("datasets/debug_data")
    train_variables, val_variables, max_length = debug_data
  if args.test_mode:
    encoder = torch.load("results/enc_{0}_{1}.pt".format(args.model_path, args.suffix))
    decoder = torch.load("results/dec_{0}_{1}.pt".format(args.model_path, args.suffix))
  # ---- BUILD MODEL ----
  print("Running model {0}_{1}".format(args.model_path, args.suffix))
  encoder, decoder = choose_model(args.model_type, vocab.ulary_size(args.task_name),
      args.hidden_size, args.attn_method, args.n_layers, args.drop_prob)
  # ---- TRAIN MODEL ----
  results = track_progress(args, encoder, decoder, args.verbose, args.debug,
      processor, args.task_name, args.epochs, weight_decay=args.weight_decay)
  # --- MANAGE RESULTS ---
  if args.save_model and results[0].completed_training:
    torch.save(encoder, "results/enc_{0}_{1}.pt".format(args.model_path, args.suffix))
    torch.save(decoder, "results/dec_{0}_{1}.pt".format(args.model_path, args.suffix))
    print('Model saved at results/model_{}!'.format(args.model_path))
  if args.report_results and results[0].completed_training:
    evaluator.quant_report(*results)
    evaluator.qual_report(encoder, decoder, processor.val_data)
  if args.plot_results and results[0].completed_training:
    evaluator.plot([strain, sval], [ltrain, lval], 'Training curve', 'Iterations', 'Loss')
  if args.visualize > 0:
    visualizations = grab_attention(processor.val_data, encoder, decoder, args.task_name, args.visualize)
    evaluator.show_save_attention(visualizations, args.attn_method, args.verbose)

'''
    criterion = NegLL_Loss()
rate_of_success = []
rate_of_loss = []
for iteration, data_pair in enumerate(processor.test_data):
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
'''