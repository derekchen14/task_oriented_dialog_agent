import numpy as np
import utils.internal.vocabulary as vocab
import torch
import torch.nn as nn
from model.components import smart_variable

use_cuda = torch.cuda.is_available()

def match_embedding(vocab_size, hidden_size):
  match_tensor = torch.load('datasets/restaurants/match_features.pt')
  embed = nn.Embedding(vocab_size, hidden_size)
  # Extract just the tensor inside the Embedding
  embed_tensor = embed.weight.data
  extended_tensor = torch.cat([embed_tensor, match_tensor], dim=1)
  # Set the weight of original embedding matrix with the new Parameter
  embed.weight = nn.parameter.Parameter(extended_tensor)
  return embed

def token_to_vec(location):
  # 8 = match features, 1 = position
  token_vector = np.zeros(vocab_size + 8 + 1)
  token_vector[location] = 1
  return token_vector
  # sentence_length = len(encoding)
  # sentence_matrix = np.zeros((sentence_length, vector_size))
  # sentence_matrix[token_location, np.array(encoding)] = 1

def task_simplification(task):
  if task in ['schedule','navigate','weather']:
    return 'car'
  elif task in ['1', '2', '3', '4', '5']:
    return 'res'
  elif task == 'challenge':
    return task
  elif task == 'concierge':
    raise ValueError("Sorry, concierge task not supported at this time")
  else:
    raise ValueError("Not a valid task")

def variable_from_sentence(sentence, indexes, task):
  indexes.extend([vocab.word_to_index(w, task) for w in sentence])
  indexes.append(vocab.EOS_token)
  # view.(-1,1) reshapes from vector into nx1 matrix
  result = smart_variable(torch.LongTensor(indexes).view(-1, 1))
  return result, len(indexes)

def dialog_to_variable(dialog, task, maxish):
  # Dialog: list of tuples, where each tuple is utterance and response
  # [(u1, r1), (u2, r2)...]
  dialog_pairs = []
  for t_idx, turn in enumerate(dialog):
    utterance, wop = variable_from_sentence(turn[0], [t_idx+1], task)
    response, dop = variable_from_sentence(turn[1], [], task)
    # utt_vector = [token_to_vec(t) for t in utt_encoding]
    # res_vector = [token_to_vec(t) for t in res_encoding]
    dialog_pairs.append((utterance, response))
  return dialog_pairs, maxish

def collect_dialogues(dataset, task):
  maxish = 0
  variables = []
  for dialog in dataset:
    dialog_vars, maxish = dialog_to_variable(dialog, task, maxish)
    variables.extend(dialog_vars)
  return variables