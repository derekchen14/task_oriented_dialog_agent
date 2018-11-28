# -*- coding: utf-8 -*-
import torch
torch.manual_seed(14)

from utils.internal.arguments import solicit_args
from utils.internal import vocabulary as vocab

from model.preprocess import PreProcessor, DataLoader
from model.learn import Builder, Learner, UserSimulator
from model.evaluate import LossTracker, Evaluator, Tester

if __name__ == "__main__":
  args = solicit_args()
  task = args.task_name
  kind = args.report_path
  # ---- LOAD AND PREPROCESS ------
  processor = PreProcessor(args, kind)
  tracker = LossTracker(args)
  builder = Builder(args)
  # ---- TRAIN OR TEST MODELS  ----
  if args.test_mode:
    tester = Tester(args, processor, kind)
    tester.test("macro_f1", "accuracy") # "micro_f1", "bleu", "just_loss"
  else:
    learner = Learner(args, processor, builder, tracker, kind)
    learner.learn(task)
  # ------- MANAGE RESULTS -------
  if not learner.tracker.completed_training: sys.exit()
  evaluator = Evaluator(args, kind)
  if args.save_model:
    learner.save_model()
  if args.report_results:
    evaluator.quant_report(learner.tracker)
    evaluator.qual_report(system, processor.val_data)
  if args.visualize > 0:
    evaluator.visual_report(processor.val_data, system, task, args.visualize)