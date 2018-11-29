import pdb, sys, os
import torch
from tqdm import tqdm as progress_bar
from model.components import var, run_inference
import numpy as np

# Used for testing, as opposed to training or validation
class Tester(object):
  def __init__(self, args, processor, kind):
    self.test_data = processor.test_data
    print(kind)

    model_path = "results/{0}_{1}.pt".format(args.model_name, args.suffix)
    if os.path.exists(model_path):
      self.model = torch.load(model_path)
      self.model.eval()
    else:
      raise FileNotFoundError('Please train and save a dialogue model first.')

  def test(self, *metrics):
    predictions, targets = [], []
    for test_pair in self.test_data: #progress_bar(self.test_data):
      test_input, test_output = test_pair
      _, pred, _ = run_inference(self.model, test_input, \
                        test_output, criterion=None, teach_ratio=0)
      targets.append(test_output.tolist()[0])
      predictions.append(pred.item())

    classes = list(set(targets))

    for metric in metrics:
      getattr(self, metric)(classes, predictions, targets)
    sys.exit()

  def just_loss(self):
    criterion = NegLL_Loss()
    rate_of_success = []
    rate_of_loss = []
    for iteration, test_pair in enumerate(self.test_data):
      if iteration % 31 == 0:
        test_input, test_output = test_pair
        loss, _, success = validate(test_input, test_output, encoder, decoder, criterion, task)
        if success:
          rate_of_success.append(1)
        else:
          rate_of_success.append(0)

        rate_of_loss.append(loss)
    ros = np.average(rate_of_success)
    rol = np.average(rate_of_loss)
    print("Loss: {} and Success: {:.3f}".format(rol, ros))

  def accuracy(self, task):
    batch_test_loss, batch_bleu, batch_success = [], [], []
    bleu_scores, accuracy = [], []

    for test_pair in progress_bar(self.test_data):
      test_input, test_output = test_pair
      loss, predictions, visual = run_inference(self.model, test_input, \
                        test_output, criterion=NLLLoss(), teach_ratio=0)

      targets = test_output.data.tolist()
      predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
      target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

      test_loss = loss.data[0] / test_output.size()[0]
      bleu_score = BLEU.compute(predicted_tokens, target_tokens)
      turn_success = all([pred == tar[0] for pred, tar in zip(predictions, targets)])

      batch_test_loss.append(test_loss)
      batch_bleu.append(bleu_score)
      batch_success.append(turn_success)

    return batch_processing(batch_test_loss, batch_bleu, batch_success)

  def macro_f1(self, classes, predictions, targets):
    total_f1 = []
    for cls in progress_bar(classes):

      true_positive, false_positive, false_negative, true_negative = 0,0,0,0
      for pred, tar in zip(predictions, targets):
        if pred == cls and tar == cls:
          true_positive += 1
        elif pred == cls and tar != cls:
          false_positive += 1
        elif pred != cls and tar == cls:
          false_negative += 1
        elif pred != cls and tar != cls:
          true_negative += 1

      f1 = Tester.single_f1(true_positive, false_positive, false_negative)
      if f1 >= 0:
        total_f1.append(f1)

    macro = np.average(total_f1)
    print("Macro average is {:.3f}".format(macro))
    return macro

  def micro_f1(self, classes, predictions, targets):
    true_positive, false_positive, false_negative, true_negative = 0,0,0,0

    for cls in progress_bar(classes):
      for pred, tar in zip(predictions, targets):
        if pred == cls and tar == cls:
          true_positive += 1
        elif pred == cls and tar != cls:
          false_positive += 1
        elif pred != cls and tar == cls:
          false_negative += 1
        elif pred != cls and tar != cls:
          true_negative += 1

    micro = Tester.single_f1(true_positive, false_positive, false_negative)
    print("Micro average is {:.3f}".format(micro))
    return micro

  @staticmethod
  def single_f1(true_positive, false_positive, false_negative):
    precision_total = max(1, true_positive + false_positive)
    precision = true_positive / precision_total
    recall_total = max(1, true_positive + false_negative)
    recall = true_positive / recall_total
    if (true_positive + false_positive + false_negative == 0):
      return -1
    elif (precision + recall == 0):
      return 0
    else:
      return  2 * ((precision * recall) / (precision + recall))

















