import os, pdb, sys
import numpy as np
import time as tm
import logging
from collections import defaultdict

from utils.internal.clock import *
from objects.components import unique_identifier

class MonitorBase(object):

  def summarize_results(self):
    """ should summarize all relevant metrics to final score as a scalar
    floating point value, rather than a list of values per epoch"""
    raise(NotImplementedError)

  def best_so_far(self):
    # returns a boolean on whether this latest model is the best seen so far
    raise(NotImplementedError)

  def build_logger(self, save_dir):
    self.logger = logging.getLogger(self.__class__.__name__)
    self.logger.setLevel(logging.DEBUG)  # set root to be very relaxed

    file_handler = logging.FileHandler(os.path.join(save_dir, 'results.log'))
    file_handler.setLevel(logging.INFO)  # log anything info or above
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')
    file_handler.setFormatter(file_formatter)
    self.logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # print out only warning or above
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    self.logger.addHandler(console_handler)

class LossMonitor(MonitorBase):
  def __init__(self, threshold, metrics, early_stop):
    self.completed_training = True
    if threshold > 0.0:
      self._prepare_early_stop(threshold)

    self.best = {}
    self.status = {}

    self.metrics = metrics
    self.early_stop_metric = early_stop

  def start_epoch(self, debug, epoch, logger):
    self.debug = debug
    self.epoch = epoch
    self.logger = logger
    self._print_frequency(debug)
    self.epoch_start_time = tm.time()

    self.iteration = 0
    self.train_losses = []
    self.val_losses = []

  def end_epoch(self):
    no_operation = 0
    time_past(self.epoch_start_time)

  def update_train(self, loss):
    it = self.iteration
    self.train_losses.append(loss)
    if it > 0 and it % self.print_every == 0:
      avg_loss = np.average(self.train_losses)
      self.logger.info('{}) Train Loss: {:.4f}'.format(it, avg_loss))
      self.train_losses = []  # reset the monitor
    self.iteration += 1

  def update_val(self, val_results, train_results):
    self.status.update(val_results)
    for tr_key, tr_value, in train_results.items():
      train_key = "train_" + tr_key
      self.status[train_key] = tr_value
    # for metric in metrics:
    #   if metric == 'bleu':
    #     score = self.calculate_bleu(batch)
    #   if metric == 'val_loss':
    #     score = loss
    #   self.status[metric].append(score)

   # TODO, move all these function to Evaluator
  def calculate_bleu(self, batch):
    input_var, output_var = batch
    queries = input_var.data.tolist()
    targets = output_var.data.tolist()
    # when task is not specified, it defaults to index_to_label
    predicted_tokens = [self.vocab.index_to_label(predictions[0])]
    query_tokens = [self.vocab.index_to_word(y) for y in queries]
    target_tokens = [self.vocab.index_to_label(z) for z in targets]

    return BLEU.compute(predicted, target)

  def calculate_rouge(self, batch):
    return 0.0
  def calculate_meteor(self, batch):
    return 0.0
  def calculate_accuracy(self, batch):
    return 0.0

  def _print_frequency(self, debug):
    self.print_every = 14
    self.val_every = 50
    if debug:  # print more often
      self.print_every /= 2
      self.val_every /= 2

  def _prepare_early_stop(self, threshold):
    # Minimum loss we are willing to accept for calculating absolute loss
    self.threshold = threshold
    # Trailing average storage for calculating relative loss
    self.trailing_average = []
    self.absolute_range = 4
    self.epochs_per_avg = 3
    self.relative_range = 2

  def time_to_validate(self):
    return self.iteration > 0 and self.iteration % self.val_every == 0

  def best_so_far(self):
    candidate_result = self.status[self.early_stop_metric]
    best_result = self.best.get(self.early_stop_metric, 0)
    if self.early_stop_metric in ['val_loss', 'avg_turn']:
      candidate_result *= -1  # reverse  since lower is better
      best_result *= -1

    if candidate_result > best_result:
      self.best.update(self.status)
      return True
    return False

  def should_early_stop(self):
    # if self.iteration > 800:
    #   return True
    if self.threshold < 0:  # we turn off early stopping
      return False
    # if the absolute or relative loss has exploded, we stop early
    end_trail = len(self.val_losses)
    start_trail = end_trail - self.epochs_per_avg
    if start_trail > 0:
      avg = np.average(self.val_losses[start_trail:end_trail])
      self.trailing_average.append(float(avg))
      if self._check_absolute(avg) or self._check_relative(avg, start_trail):
        print("Early stopped at val epoch {}".format(end_trail))
        self.completed_training = False
        return True
    # if nothing causes an alarm, then we should just continue
    return False

  def _check_absolute(self, current_avg):
    if self.val_epoch == (10 - self.absolute_range):
      if current_avg > (self.threshold * 1.5):
        return True
    elif self.val_epoch == 10:
      if current_avg > self.threshold:
        return True
    elif self.val_epoch == (10 + self.absolute_range):
      if current_avg > (self.threshold * 0.9):
        return True
    return False

  def _check_relative(self, current_avg, trail_idx):
    if self.val_epoch >= (self.epochs_per_avg + self.relative_range):
      lookback_avg = self.trailing_average[trail_idx - self.relative_range]
      if (current_avg / lookback_avg) > 1.1:
        return True
    return False

  def summarize_results(self, verbose=False):
    self.logger.info("Epoch {}, iteration {}:".format(self.epoch, self.iteration))
    summary = self.status.copy()
    for metric, metric_value in self.best.items():
      summary["best_{}".format(metric)] = metric_value
    for metric, metric_value in summary.items():
      self.logger.info("{}: {:.4f}".format(metric, metric_value))

    unique_id = unique_identifier(summary, self.epoch, self.iteration, self.early_stop_metric)
    return summary, unique_id
    # for metric in self.metrics:
    #   if metric == "val_loss":
    #     metric_value = np.average(self.status[metric])
    #   else:
    #     metric_value = self.status[metric][-1]

class RewardMonitor(MonitorBase):
  """ Tracks global learning status across episodes. """
  def __init__(self, metrics, threshold=0.0):
    self.rewards = []
    self.turns = []
    self.num_successes = 0
    self.num_episodes = 0
    self.success_rate = 0.0

    self.metrics = metrics
    self.success_threshold = threshold  # 0.3
    self.best_success_rate = -1.0
    # self.warm_start_epochs = 100
    # self.save_check_point = 5   # save the last X checkpoints
  def start_episode(self):
    self.status = {'turn_count': 0, 'success': False, 'episode_reward': 0}

  def end_episode(self):
    self.rewards.append(self.status["episode_reward"])
    self.turns.append(self.status["turn_count"])
    if self.status["success"]:
      self.num_successes += 1
    self.num_episodes += 1

  def summarize_results(self, verbose=False, prefix=None):
    self.success_rate = self.num_successes / float(self.num_episodes)
    self.avg_reward = np.average(self.rewards)
    self.avg_turn = np.average(self.turns)
    self.unique_id =  "episode_{}_best_success_{:.4f}_current_{:.4f}".format(
                  self.num_episodes, self.best_success_rate, self.success_rate)
    if verbose:
      result_str = "Success Rate: {:.4f}, Average Reward: {:.4f}, Average Turns: {:.4f}".format(
        self.success_rate, self.avg_reward, self.avg_turn)
      if prefix is not None:
        print(prefix + result_str)
      else:
        print("Epoch: {}, ".format(self.num_episodes) + result_str)

  def best_so_far(self, simulator_success_rate):
    if simulator_success_rate >= self.best_success_rate:
      self.best_success_rate = simulator_success_rate
      return True if simulator_success_rate >= self.success_threshold else False
