import numpy as np

class LossTracker(object):
  def __init__(self, args):
    self.train_steps = []
    self.train_losses = []
    self.train_epoch = 0

    self.val_steps = []
    self.val_losses = []
    self.val_epoch = 0

    self.completed_training = True
    # Minimum loss we are willing to accept for calculating absolute loss
    self.threshold = args.early_stop
    self.absolute_range = 4
    # Trailing average storage for calculating relative loss
    self.trailing_average = []
    self.epochs_per_avg = 3
    self.lookback_range = 2

    self.bleu_scores = []
    self.accuracy = []

  def update_loss(self, loss, split):
    if split == "train":
      self.train_losses.append(loss)
      self.train_epoch += 1
    elif split == "val":
      self.val_losses.append(loss)
      self.val_epoch += 1

  def update_stats(self, bleu, acc):
    self.bleu_scores = bleu
    self.accuracy = acc

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
