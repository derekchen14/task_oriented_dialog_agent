import json
import torch
import os, pdb, sys
from random import seed
from tqdm import tqdm as progress_bar

from objects.models import NLU
from utils.internal.arguments import solicit_args
from operators.preprocess import DataLoader

if __name__ == "__main__":
  args = solicit_args()
  torch.manual_seed(args.seed)
  seed(args.seed)

  loader = DataLoader(args)
  model_path = args.prefix + args.model + args.suffix
  data = loader.json_data('clean/val')
  module = NLU(loader, 'results/track_intent/e2e/movies/')
  module.load_nlu_model(model_path)
  # module = NLU(loader, 'results/manage_policy/ddq/movies/')
  # module.load_nlu_model('nlu_1468447442')

  inform_correct = 0
  request_correct = 0
  joint_correct = 0
  i_total, r_total, j_total = 0, 0, 0

  for example in progress_bar(data):
    for turn in example['turns']:
      utt = turn['utterance']
      pred = module.classify_intent(utt)

      has_request, has_inform = False, False
      r_passed, i_passed = True, True


      for intent in turn['user_intent']:
        slot, val = intent
        if slot == 'request':
          has_request = True
          if val in pred['request_slots'].keys():
            pass
          else:
            r_passed = False
        elif slot == 'act':
          pass
        else:
          has_inform = True
          if slot in pred['inform_slots'].keys():
            if val == pred['inform_slots'][slot]:
              pass
            else:
              i_passed = False
          else:
            i_passed = False
        # print(utt, intent, pred)
        # print(f'inform: {i_passed}, request: {r_passed}')

      if has_request:
        r_total += 1
        if r_passed:
          request_correct += 1
      if has_inform:
        i_total += 1
        if i_passed:
          inform_correct += 1
      j_total += 1
      if r_passed and i_passed and (has_request or has_inform):
        joint_correct += 1

  print("inform is {0}/{1} for a accuracy of {2:.2f}%".format(
          inform_correct, i_total, 100.0 * inform_correct/float(i_total)))
  print("request is {0}/{1} for a accuracy of {2:.2f}%".format(
          request_correct, r_total, 100.0 * request_correct/float(r_total)))
  print("joint goal is {0}/{1} for a accuracy of {2:.2f}%".format(
          joint_correct, j_total, 100.0 * joint_correct/float(j_total)))



# python evaluate_old_nlu.py --task track_intent  --seed 14 \
#   --epochs 1 --dataset e2e/movies --prefix rmsprop_ --model lstm --suffix _14
