# -*- coding: utf-8 -*-
import torch
import logging
from random import seed

from utils.internal.arguments import solicit_args
from operators.preprocess import DataLoader, PreProcessor
from operators.learn import Builder, Learner
from operators.evaluate import Evaluator
from operators.system import SingleSystem, EndToEndSystem

if __name__ == "__main__":
  args = solicit_args()
  # ------ BASIC SYSTEM SETUP ------
  torch.manual_seed(args.seed)
  seed(args.seed)
  logging.basicConfig(level=logging.INFO)
  # ----- INITIALIZE OPERATORS -----
  loader = DataLoader(args)
  builder = Builder(args, loader)
  processor = PreProcessor(args, loader)
  evaluator = Evaluator(args, processor)
  # -------- RUN THE SYSTEM --------
  if args.task == "end_to_end":
    system = EndToEndSystem(args, loader, builder, processor, evaluator)
  else:
    system = SingleSystem(args, loader, builder, processor, evaluator)
  if not args.test_mode:
    system.run_main()
    logging.info(args)
  system.evaluate()