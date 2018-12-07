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
  slot_processor = DualProcessor(args, "slot")
  value_processor = DualProcessor(args, "value")
  tracker = LossTracker(args)
  builder = Builder(args)
  # ---- TRAIN OR TEST MODELS  ----
  if args.test_mode:
    tester = Tester(args, processor, kind)
    tester.test("macro_f1", "micro_f1") # "accuracy", "bleu", "just_loss"
  else:
    slot_learner = Learner(args, slot_processor, builder, tracker, "slot")
    slot_learner.learn(task)
    value_learner = Learner(args, value_processor, builder, tracker, "value")
    value_learner.learn(task)
  # ------- MANAGE RESULTS -------
  evaluator = Evaluator(args, kind)
  evaluator.report([slot_learner, value_learner])