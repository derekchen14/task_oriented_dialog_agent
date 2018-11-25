import time as tm
import pdb, sys
import random

from torch import save
from torch.nn import NLLLoss as NegLL_Loss
from torch.optim.lr_scheduler import StepLR

from utils.external.bleu import BLEU
from utils.internal import vocabulary as vocab
from utils.internal.clock import *
from model.components import *

class Learner(object):
  def __init__(self, args, processor, builder, tracker, kind):
    self.verbose = args.verbose
    self.debug = args.debug
    self.epochs = args.epochs

    self.decay_times = args.decay_times
    self.teach_ratio = args.teacher_forcing
    self.model_name = "{0}_{1}".format(args.model_path, args.suffix)

    self.processor = processor
    self.builder = builder
    self.tracker = tracker

    self.kind = kind

  def train(self, input_var, output_var, enc_optimizer, dec_optimizer):
    self.model.train()   # affects the performance of dropout
    self.model.zero_grad()

    loss, _, _ = run_inference(self.model, input_var, output_var, \
                          self.criterion, self.teach_ratio)
    loss.backward()
    clip_gradient(self.model, clip=10)
    enc_optimizer.step()
    dec_optimizer.step()

    return loss.item() / output_var.shape[0]

  def validate(self, input_var, output_var, task):
    self.model.eval()  # val period has no training, so teach ratio is 0
    loss, predictions, visual = run_inference(self.model, input_var, \
                      output_var, self.criterion, teach_ratio=0)

    queries = input_var.data.tolist()
    targets = output_var.data.tolist()

    # when task is not specified, it defaults to index_to_label
    predicted_tokens = [vocab.index_to_word(predictions, self.kind)]
    query_tokens = [vocab.index_to_word(y, task) for y in queries]
    target_tokens = [vocab.index_to_word(z, self.kind) for z in targets]

    avg_loss = loss.item() / output_var.shape[0]
    bleu_score = BLEU.compute(predicted_tokens, target_tokens)
    turn_success = (predictions.item() == targets[0])

    return avg_loss, bleu_score, turn_success

  ''' Modified since predictions are now single classes rather than sentences
  predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
  query_tokens = [vocab.index_to_word(y[0], task) for y in queries]
  target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

  turn_success = [pred == tar[0] for pred, tar in zip(predictions, targets)]
  return avg_loss, bleu_score, all(turn_success)
  '''

  def learn(self, model, task):
    self.model = model
    self.learn_start = tm.time()
    n_iters = 600 if self.debug else len(self.processor.train_data)
    print_every, plot_every, val_every = print_frequency(self.verbose, self.debug)
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    enc_optimizer, dec_optimizer = self.builder.init_optimizers(self.model)
    # training_pairs = [random.choice(train_data) for i in range(n_iters)]
    # validation_pairs = [random.choice(val_data) for j in range(v_iters)]
    self.criterion = NegLL_Loss()
    step_size = n_iters/(self.decay_times+1)
    enc_scheduler = StepLR(enc_optimizer, step_size=step_size, gamma=0.2)
    dec_scheduler = StepLR(dec_optimizer, step_size=step_size, gamma=0.2)

    for epoch in range(self.epochs):
      start = tm.time()
      starting_checkpoint(epoch, self.epochs, use_cuda)
      for iteration, training_pair in enumerate(self.processor.train_data):
        enc_scheduler.step()
        dec_scheduler.step()
        input_var = training_pair[0]
        output_var = training_pair[1]

        loss = self.train(input_var, output_var, enc_optimizer, dec_optimizer)
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
            val_loss, bs, ts = self.validate(val_input, val_output, task)
            batch_val_loss.append(val_loss)
            batch_bleu.append(bs)
            batch_success.append(ts)

          self.tracker.batch_processing(batch_val_loss, batch_bleu, batch_success)
          if self.tracker.should_early_stop():
            print("Early stopped at val epoch {}".format(tracker.val_epoch))
            self.tracker.completed_training = False
            break
    time_past(self.learn_start)

  def save_model(self):
    model_path = "results/{}.pt".format(self.model_name)
    save(self.model, model_path)
    print('Model saved at {}!'.format(model_path))
