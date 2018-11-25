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
  # ----- INITIALIZE MODULES -----
  processor = PreProcessor(args, kind)
  builder = Builder(args)
  tracker = LossTracker(args)
  learner = Learner(args, processor, builder, tracker, kind)
  evaluator = Evaluator(args, kind)
  # ---- LOAD AND PREPROCESS ------
  if args.debug:
    debug_data = pickle_loader("datasets/debug_data")
    train_variables, val_variables, max_length = debug_data
  if args.test_mode:
    model = torch.load("results/{0}_{1}.pt".format(args.model_path, args.suffix))
  # ---- BUILD AND TRAIN MODEL ----
  print("Running model {0}_{1}".format(args.model_path, args.suffix))
  model = builder.create_model(vocab.ulary_size(task), vocab.label_size(kind))
  learner.learn(model, task)
  # ------- MANAGE RESULTS -------
  if not learner.tracker.completed_training: sys.exit()
  if args.save_model:
    learner.save_model()
  if args.report_results:
    evaluator.quant_report(learner.tracker)
    evaluator.qual_report(model, processor.val_data)
  if args.visualize > 0:
    evaluator.visual_report(processor.val_data, model, task, args.visualize)
  if args.plot_results:
    evaluator.plot([strain, sval], [ltrain, lval])

'''
    criterion = NegLL_Loss()
rate_of_success = []
rate_of_loss = []
for iteration, data_pair in enumerate(processor.test_data):
  if iteration % 31 == 0:
    test_input, test_output = data_pair
    loss, _, success = validate(test_input, test_output, encoder, decoder, criterion, task)
    if success:
      rate_of_success.append(1)
    else:
      rate_of_success.append(0)

    rate_of_loss.append(loss)
ros = np.average(rate_of_success)
rol = np.average(rate_of_loss)
print("Loss: {} and Success: {:.3f}".format(rol, ros))
sys.exit()
'''