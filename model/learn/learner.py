import time as tm
import pdb, sys
import random
import logging

from torch.nn import NLLLoss as NegLL_Loss
from torch.optim.lr_scheduler import StepLR

from utils.external.bleu import BLEU
from utils.internal.clock import *
from model.components import *

class Learner(object):
  def __init__(self, args, model, processor, tracker, task):
    self.verbose = args.verbose
    self.debug = args.debug
    self.epochs = args.epochs
    self.decay_times = args.decay_times
    self.teach_ratio = args.teacher_forcing

    self.processor = processor
    self.tracker = tracker
    self.vocab = processor.vocab
    self.model = model

    self.task = args.task
    self.model_idx = processor.loader.categories.index(self.task)   # order matters, do not switch

  def train(self, input_var, output_var):
    self.model.train()   # affects the performance of dropout
    self.model.zero_grad()

    loss, _, _ = run_inference(self.model, input_var, output_var, \
                          self.criterion, self.teach_ratio)
    loss.backward()
    clip_gradient(self.model, clip=10)
    self.model.optimizer.step()

    return loss.item() / output_var.shape[0]

  def validate(self, input_var, output_var, task):
    self.model.eval()  # val period has no training, so teach ratio is 0
    loss, predictions, visual = run_inference(self.model, input_var, \
                      output_var, self.criterion, teach_ratio=0)

    queries = input_var.data.tolist()
    targets = output_var.data.tolist()

    # when task is not specified, it defaults to index_to_label
    predicted_tokens = [self.vocab.index_to_label(predictions[0])]
    query_tokens = [self.vocab.index_to_word(y) for y in queries]
    target_tokens = [self.vocab.index_to_label(z) for z in targets]

    avg_loss = loss.item() / output_var.shape[0]
    # bleu_score = 1 BLEU.compute(predicted_tokens, target_tokens)
    exact_success = (predictions[0].item() == targets[0])
    rank_success = targets[0] in predictions
    # return avg_loss, bleu_score, turn_success
    return avg_loss, exact_success, rank_success

  ''' Modified since predictions are now single classes rather than sentences
  predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
  query_tokens = [vocab.index_to_word(y[0], task) for y in queries]
  target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

  turn_success = [pred == tar[0] for pred, tar in zip(predictions, targets)]
  return avg_loss, bleu_score, all(turn_success)
  '''

  def learn(self, task):
    self.learn_start = tm.time()
    logging.info('Starting to learn ...')
    self.model.init_optimizer()
    self.criterion = NegLL_Loss()

    n_iters = 600 if self.debug else len(self.processor.train_data)
    print_every, plot_every, val_every = print_frequency(self.verbose, self.debug)
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # step_size = n_iters/(self.decay_times+1)
    # enc_scheduler = StepLR(enc_optimizer, step_size=step_size, gamma=0.2)
    # dec_scheduler = StepLR(dec_optimizer, step_size=step_size, gamma=0.2)

    for epoch in range(self.epochs):
      start = tm.time()
      starting_checkpoint(epoch, self.epochs, use_cuda)
      for iteration, training_pair in enumerate(self.processor.train_data):
        # enc_scheduler.step()
        # dec_scheduler.step()
        input_var, output_var = training_pair
        if self.task == "per_slot":
          output_var = output_var[self.model_idx]

        loss = self.train(input_var, output_var)
        print_loss_total += loss
        plot_loss_total += loss

        if iteration > 0 and iteration % print_every == 0:
          self.tracker.train_steps.append(iteration + 1)
          print_loss_avg = print_loss_total / print_every
          print_loss_total = 0  # reset the print loss
          print('{1:3.1f}% complete {2}, Train Loss: {0:.4f}'.format(print_loss_avg,
              (iteration/n_iters * 100.0), timeSince(start, iteration/n_iters )))
          self.tracker.update_loss(print_loss_avg, "train")

        if iteration > 0 and iteration % val_every == 0:
          self.tracker.val_steps.append(iteration + 1)
          batch_val_loss, batch_bleu, batch_success = [], [], []
          for val_input, val_output in self.processor.val_data:
            if self.task == "per_slot":
              val_output = val_output[self.model_idx]
            val_loss, bs, ts = self.validate(val_input, val_output, task)
            batch_val_loss.append(val_loss)
            batch_bleu.append(bs)
            batch_success.append(ts)

          self.tracker.batch_processing(batch_val_loss, batch_bleu, batch_success)
          if self.tracker.should_early_stop(iteration):
            print("Early stopped at val epoch {}".format(self.tracker.val_epoch))
            self.tracker.completed_training = False
            break
      if self.tracker.best_so_far():
        summary = self.tracker.generate_summary()
        identifier = "epoch={0}_success={1:.4f}_recall@two={2:.4f}".format(
              summary["train_epoch"], summary["accuracy"], summary["recall@k=2"])
        self.model.save(summary, identifier)

    logging.info("Done training {}".format(task))
    time_past(self.learn_start)