import numpy as np
import torch
from torch.autograd import Variable
import utils.internal.vocabulary as vocab
import sys

use_cuda = torch.cuda.is_available()

def token_to_vec(location):
  # 8 = match features, 1 = position
  token_vector = np.zeros(vocab_size + 8 + 1)
  token_vector[location] = 1
  return token_vector
  # sentence_length = len(encoding)
  # sentence_matrix = np.zeros((sentence_length, vector_size))
  # sentence_matrix[token_location, np.array(encoding)] = 1

def match_feature_augmentation(dialog):
  pass

def variable_from_sentence(sentence, indexes):
  indexes.extend([vocab.word_to_index(w) for w in sentence])
  indexes.append(vocab.EOS_token)
  # view.(-1,1) reshapes from vector into nx1 matrix
  result = Variable(torch.LongTensor(indexes).view(-1, 1))
  # except:
  #   rest = sentence[8]
  #   print rest
  #   print vocab.word_to_index(rest)
  #   print indexes
  #   sys.exit()
  # return result.cuda() if use_cuda else result
  return result, len(indexes)

def dialog_to_variable(dialog, maxish):
  # Dialog: list of tuples, where each tuple is utterance and response
  # [(u1, r1), (u2, r2)...]
  dialog_pairs = []
  for t_idx, turn in enumerate(dialog):
    utterance, wop = variable_from_sentence(turn[0], [t_idx+1])
    response, dop = variable_from_sentence(turn[1], [])
    # utt_vector = [token_to_vec(t) for t in utt_encoding]
    # res_vector = [token_to_vec(t) for t in res_encoding]
    dialog_pairs.append((utterance, response))
  return dialog_pairs, maxish

def collect_dialogues(dataset):
  maxish = 0
  variables = []
  for dialog in dataset:
    dialog_vars, maxish = dialog_to_variable(dialog, maxish)
    variables.extend(dialog_vars)
  return variables