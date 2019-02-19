import os, pdb
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

  def build_logger(self, save_dir, split='train'):
    logger = logging.getLogger('{}-{}'.format(split, self.__class__.__name__))
    log_filename = os.path.join(save_dir, '{}.log'.format(split))
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

class LossMonitor(MonitorBase):
  def __init__(self, threshold, metrics, early_stop):
    self.best = {'val_loss': np.inf}
    self.completed_training = True
    if threshold > 0.0:
      self._prepare_early_stop(threshold)
    # self.build_logger()
    self.status = defaultdict(list)
    self.metrics = metrics
    self.early_stop_metric = early_stop

  def start_epoch(self, debug, epoch):
    self.debug = debug
    self.epoch = epoch
    self._print_frequency(debug)
    self.epoch_start_time = tm.time()

    self.iteration = 0
    self.train_losses = []
    self.val_losses = []

  def end_epoch(self):
    no_operation = 0
    time_past(self.epoch_start_time)

  def update_train(self, loss):
    self.train_losses.append(loss)
    if self.iteration > 0 and self.iteration % self.print_every == 0:
      percent_complete = (self.iteration / self.iters_per_epoch) * 100.0
      avg_loss = np.average(self.train_losses)
      time_left = timeSince(self.epoch_start_time, percent_complete)
      print('{:3.1f}% complete {}, Train Loss: {0:.4f}'.format(
              percent_complete, time_left, avg_loss))
      self.train_losses = []  # reset the monitor
    self.iteration += 1

  def update_val(self, results):
    print("results being passed to val update:")
    print(results)
    print("status before")
    print(self.status)
    self.status.udpate(results)
    print("status after")
    print(self.status)
    # for metric in metrics:
    #   if metric == 'bleu':
    #     score = self.calculate_bleu(batch)
    #   if metric == 'accuracy':
    #     score = self.calculate_accuracy(batch)
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

  def best_so_far(self, early_stop_metric):
    candidate_result = self.status[early_stop_metric]
    best_result = self.best.get(early_stop_metric, 0)
    if early_stop_metric in ['val_loss', 'avg_turn']:
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

  def summarize_results(self, logger):
    logger.info("Epoch {}, iteration {}:".format(self.epoch, self.iteration))
    summary = self.status.copy()
    for metric, metric_value in self.best.items():
      summary["best_{}".format(metric)] = metric_value
    for metric, metric_value in self.summary.items():
      logger.info("{}: {:.4f}".format(metric, metric_value))

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

    self.metrics = metrics
    self.success_rate_threshold = threshold  # 0.3
    self.best_success_rate = -1
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

  def summarize_results(self, verbose):
    self.success_rate = self.num_successes / float(self.num_episodes)
    self.avg_reward = np.average(self.rewards)
    self.avg_turn = np.average(self.turns)
    if verbose:
      print("Success_rate: {}, Average Reward: {:.4f}, Average Turns: {:.4f}".format(
        self.success_rate, self.avg_reward, self.avg_turn))

  def best_so_far(self, simulator_success_rate):
    if simulator_success_rate > self.success_rate_threshold:
      if simulator_success_rate > self.best_success_rate:
        self.best_success_rate = simulator_success_rate
        return True
    return False

