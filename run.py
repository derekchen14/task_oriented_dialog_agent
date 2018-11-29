# -*- coding: utf-8 -*-
import torch
torch.manual_seed(15)

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
  processor = PreProcessor(args, "intent")
  sv_processor = PreProcessor(args, "sv")
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
  evaluator.report(intent_learner, sv_learner)
  '''
  if not sv_learner.tracker.completed_training: sys.exit()
  evaluator = Evaluator(args, kind)
  if args.save_model:
    learner.save_model()
  if args.report_results:
    evaluator.quant_report(learner.tracker)
    evaluator.qual_report(learner.model, processor.val_data)
  if args.visualize > 0:
    evaluator.visual_report(processor.val_data, system, task, args.visualize)
  '''