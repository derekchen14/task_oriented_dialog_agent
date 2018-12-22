# -*- coding: utf-8 -*-
import torch
import logging
from random import seed

from utils.internal.arguments import solicit_args
from model.preprocess import PreProcessor, DataLoader
from model.learn import Builder, Learner, UserSimulator
from model.evaluate import LossTracker, Evaluator

class System():
  def __init__(self, args, loader, processor, builder, tracker):
    self.tasks = loader.categories
    vocab = loader.vocab

    for task in self.tasks:
      model = builder.get_model(vocab.ulary_size(), vocab.label_size())
      setattr(self, '{}_model'.format(task), model)
      if not args.test_mode:
        self.set_save_directory(model, task, builder.dir)
        learner = Learner(args, model, processor, tracker, task)
        setattr(self, '{}_learner'.format(task), learner)

  def set_save_directory(self, model, task, save_directory):
    if len(self.tasks) == 1:
      model.save_dir = save_directory
    else:
      model.save_dir = "{}_{}".format(task, save_directory)

  def run_main(self):
    for task in self.tasks:
      learner = getattr(self, '{}_learner'.format(task))
      learner.learn(task)

if __name__ == "__main__":
  args = solicit_args()
  # ------ BASIC SYSTEM SETUP ------
  torch.manual_seed(args.seed)
  seed(args.seed)
  logging.basicConfig(level=logging.INFO)
  # ----- INITIALIZE OPERATORS -----
  loader = DataLoader(args)
  processor = PreProcessor(args, loader)
  tracker = LossTracker(args)
  builder = Builder(args, loader)
  # -------- RUN THE SYSTEM --------
  system = System(args, loader, processor, builder, tracker)
  if not args.test_mode:
    system.run_main()
    logging.info(args)
  # ----- EVALUATE PERFORMANCE -----
  evaluator = Evaluator(args, processor, system)
  evaluator.run_report(args.metrics)