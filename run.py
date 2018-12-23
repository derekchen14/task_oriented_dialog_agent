# -*- coding: utf-8 -*-
import torch
import logging
from random import seed

from utils.internal.arguments import solicit_args
from operators.preprocess import DataLoader
from operators.learn import Builder
from operators.evaluate import LossTracker, Evaluator
from operators.system import SingleSystem, MultiSystem

if __name__ == "__main__":
  args = solicit_args()
  # ------ BASIC SYSTEM SETUP ------
  torch.manual_seed(args.seed)
  seed(args.seed)
  logging.basicConfig(level=logging.INFO)
  # ----- INITIALIZE OPERATORS -----
  loader = DataLoader(args)
  tracker = LossTracker(args)
  builder = Builder(args, loader)
  # -------- RUN THE SYSTEM --------
  if loader.multitask:
    system = MultiSystem(args, loader, builder, tracker)
  else:
    system = SingleSystem(args, loader, builder, tracker)
  if not args.test_mode:
    system.run_main()
    logging.info(args)
  # ----- EVALUATE PERFORMANCE -----
  evaluator = Evaluator(args, system, loader.multitask)
  evaluator.run_report(args.metrics)