import numpy as np

class MonitorBase(object):
  def __init__(self, args):
    self.metrics = args.metrics
    self.phase = "test" if args.test_mode else "val"

    self.train_steps = []
    self.train_epoch = 0
    self.val_steps = []
    self.val_epoch = 0

  def generate_summary(self):
    """ should summarize all relevant metrics to final score as a scalar
    floating point value, rather than a list of values per epoch"""
    raise(NotImplementedError)

class LossMonitor(MonitorBase):
  def __init__(self, args):
    super().__init__(args)
    self.train_losses = []
    self.val_losses = []

    self.completed_training = True
    # Minimum loss we are willing to accept for calculating absolute loss
    self.threshold = args.early_stop
    self.absolute_range = 4
    # Trailing average storage for calculating relative loss
    self.trailing_average = []
    self.epochs_per_avg = 3
    self.lookback_range = 2

    self.bleus_scores = []
    self.accuracy = []

  def best_so_far(self):
    return True

  def update_loss(self, loss, split):
    if split == "train":
      self.train_losses.append(loss)
      self.train_epoch += 1
    elif split == "val":
      self.val_losses.append(loss)
      self.val_epoch += 1

  def update_stats(self, bleu, acc):
    # takes in a dict of keyword arguments of metrics
    # for metric_name, metric_value in metrics.items():
    #   setattr(self, metric_name, metric_value)
    self.bleu_scores.append(bleu)
    self.accuracy.append(acc)

  def batch_processing(self, batch_val_loss, batch_bleu, batch_success):
    avg_val_loss = sum(batch_val_loss) * 1.0 / len(batch_val_loss)
    avg_bleu = 100 * float(sum(batch_bleu)) / len(batch_bleu)
    avg_success = 100 * float(sum(batch_success)) / len(batch_success)

    print('Validation Loss: {0:2.4f}, BLEU Score: {1:.2f}, Per Turn Accuracy: {2:.2f}'.format(
            avg_val_loss, avg_bleu, avg_success))

    self.update_loss(avg_val_loss, "val")
    self.bleu_scores.append(avg_bleu)
    self.accuracy.append(avg_success)

  def should_early_stop(self, iteration):
    # if iteration > 800:
    #   return True
    if self.threshold < 0:  # we turn off early stopping
      return False

    trail_idx = self.val_epoch - self.epochs_per_avg
    if trail_idx >= 0:
      avg = np.average(self.val_losses[trail_idx : self.val_epoch])
      self.trailing_average.append(float(avg))

      if self._check_absolute(avg) or self._check_relative(avg, trail_idx):
        return True
    # if nothing causes an alarm, then we should continue
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
    if self.val_epoch >= (self.epochs_per_avg + self.lookback_range):
      lookback_avg = self.trailing_average[trail_idx - self.lookback_range]
      if (current_avg / lookback_avg) > 1.1:
        return True
    return False

  def generate_summary(self):
    self.bleu = self.bleu_scores[-1] if len(self.bleu_scores) > 0 else 0.0
    self.rouge = self.rouge_scores[-1] if len(self.rouge_scores) > 0 else 0.0
    self.meteor = self.meteors[-1] if len(self.meteors) > 0 else 0.0
    self.accuracy = self.accuracies[-1] if len(self.accuracies) > 0 else 0.0
    self.eval_loss = self.eval_losses[-1] if len(self.eval_losses) > 0 else 0.0

    if 'macro_f1' in self.metrics or 'micro_f1' in self.metrics:
      self.calculate_f1()


class RewardMonitor(MonitorBase):
  def __init__(self, args):
    super().__init__(args)
    self.rewards = []
    self.turns = []
    self.num_successes = 0
    self.num_episodes = 1

    self.simulation_epoch_size = 100
    self.warm_start_epochs = 100
    self.batch_size = args.batch_size # default = 16
    self.warm_start = args.warm_start
    self.success_rate_threshold = args.threshold  # 0.3
    self.save_check_point = 5   # save the last X checkpoints


    """ Best Model and Performance Records
    self.best_model = {}
    self.best_res = {'success_rate': 0, 'ave_reward':float('-inf'), 'ave_turns': float('inf'), 'epoch':0}
    # self.best_model['model'] = copy.deepcopy(agent)
    self.best_res['success_rate'] = 0
    """
  def start_episode(self):
    self.status = {'turn_count': 0, 'success': False, 'cumulative_reward': 0}

  def end_episode(self):
    self.rewards.append(self.status["cumulative_reward"])
    self.turns.append(self.status["turn_count"])
    if self.status["success"]:
      self.num_successes += 1
    self.num_episodes += 1

  def generate_summary(self):
    self.avg_reward = np.average(self.rewards)
    self.avg_turn = np.average(self.turns)
    self.success_rate = self.num_successes / float(self.num_episodes)
