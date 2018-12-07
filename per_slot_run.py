# -*- coding: utf-8 -*-
import torch
torch.manual_seed(14)

from utils.internal.arguments import solicit_args
from utils.internal import per_slot_vocab as vocab

from model.preprocess import PerSlotProcessor, DataLoader
from model.learn import Builder, Learner, UserSimulator
from model.evaluate import LossTracker, Evaluator, Tester

if __name__ == "__main__":
  args = solicit_args()
  task = args.task_name
  kind = args.report_path
  # ---- LOAD AND PREPROCESS ------
  tracker = LossTracker(args)
  builder = Builder(args)
  processor = PerSlotProcessor(args)

  all_learners = []
  for category in vocab.categories:
    learner = Learner(args, processor, builder, tracker, category)
    learner.learn(task)
    all_learners.append(learner)
    print("Done training {}".format(category))

  # ------- MANAGE RESULTS -------
  evaluator = Evaluator(args, kind)
  evaluator.report(all_learners)