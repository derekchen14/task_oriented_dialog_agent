# -*- coding: utf-8 -*-
import torch
from random import seed

from utils.internal.arguments import solicit_args
from utils.internal import vocabulary as vocab

from model.preprocess import PreProcessor, DataLoader
from model.learn import Builder, Learner, UserSimulator
from model.evaluate import LossTracker, Evaluator, Tester

if __name__ == "__main__":
  args = solicit_args()
  torch.manual_seed(args.seed)
  seed(args.seed)
  # ---- LOAD AND PREPROCESS ------
  processor = PreProcessor(args)
  tracker = LossTracker(args)
  builder = Builder(args)
  # ---- TRAIN OR TEST MODELS  ----
  if args.test_mode:
    tester = Tester(args, processor)
    tester.test("macro_f1", "micro_f1") # "accuracy", "bleu", "just_loss"
  else:
    learner = Learner(args, processor, builder, tracker)
    learner.learn(args.task)
  # ------- MANAGE RESULTS -------
  # if not learner.tracker.completed_training: sys.exit()
  evaluator = Evaluator(args)
  evaluator.report([learner])
  if args.save_model:
    learner.save_model()
  if args.report_results:
    evaluator.quant_report(learner.tracker)
    evaluator.qual_report(system, processor.val_data)
  if args.visualize > 0:
    evaluator.visual_report(processor.val_data, system, args.task, args.visualize)