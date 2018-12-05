# -*- coding: utf-8 -*-
import torch
torch.manual_seed(14)

from utils.internal.arguments import solicit_args
from utils.internal import dual_vocab as vocab

from model.preprocess import DualProcessor, DataLoader
from model.learn import Builder, Learner, UserSimulator
from model.evaluate import LossTracker, Evaluator, Tester

if __name__ == "__main__":
  args = solicit_args()
  task = args.task_name
  kind = args.report_path
  # ---- LOAD AND PREPROCESS ------
  processor = DualProcessor(args, "intent")
  sv_processor = DualProcessor(args, "sv")
  tracker = LossTracker(args)
  builder = Builder(args)
  # ---- TRAIN OR TEST MODELS  ----
  if args.test_mode:
    tester = Tester(args, processor, kind)
    tester.test("macro_f1", "micro_f1") # "accuracy", "bleu", "just_loss"
  else:
    sv_learner = Learner(args, sv_processor, builder, tracker, "sv")
    sv_learner.learn(task)
    intent_learner = Learner(args, processor, builder, tracker, "intent")
    intent_learner.learn(task)
  # ------- MANAGE RESULTS -------
  evaluator = Evaluator(args, kind)
  evaluator.dual_report(intent_learner, sv_learner)