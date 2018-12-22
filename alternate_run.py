# -*- coding: utf-8 -*-
import torch
torch.manual_seed(14)

from utils.internal import per_slot_vocab as vocab
from model.preprocess import PerSlotProcessor, DataLoader
# from utils.internal import dual_vocab as vocab
# from model.preprocess import DualProcessor, DataLoader

from utils.internal.arguments import solicit_args
from model.learn import Builder, Learner, UserSimulator
from model.evaluate import LossTracker, Evaluator, Tester

if __name__ == "__main__":
  args = solicit_args()
  task = args.task_name
  kind = args.report_path
  # ---- LOAD AND PREPROCESS ------
  slot_processor = DualProcessor(args, "slot")
  value_processor = DualProcessor(args, "value")
  # processor = PerSlotProcessor(args)
  tracker = LossTracker(args)
  builder = Builder(args)
  # ------- LEARNING -------
  slot_learner = Learner(args, slot_processor, builder, tracker, "slot")
  slot_learner.learn(task)
  value_learner = Learner(args, value_processor, builder, tracker, "value")
  value_learner.learn(task)
  # all_learners = []
  # for category in vocab.categories:
  #   learner = Learner(args, processor, builder, tracker, category)
  #   learner.learn(task)
  #   all_learners.append(learner)
  # ------- MANAGE RESULTS -------
  evaluator = Evaluator(args, kind)
  # evaluator.report(all_learners)
  evaluator.report([slot_learner, value_learner])
