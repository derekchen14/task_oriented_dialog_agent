import numpy as np
import utils.internal.vocabulary as vocab
import torch
from torch import LongTensor
import torch.nn as nn
import pdb, sys
from model.components import var

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

def dialog_to_variable(dialog, task):
  dialog_pairs = []
  for t_idx, turn in enumerate(dialog):
    utterance, wop = variable_from_sentence(turn[0], [t_idx+1], task)
    response, dop = variable_from_sentence(turn[1], [], task)
    # utt_vector = [token_to_vec(t) for t in utt_encoding]
    # res_vector = [token_to_vec(t) for t in res_encoding]
    dialog_pairs.append((utterance, response))
  return dialog_pairs

def prepare_input(source, use_context, task):
  tokens = []
  if "turn" in source.keys():
    # vocab is designed so that the first 14 tokens are turn indicators
    tokens.append(source["turn"])
  if use_context:
    for word in source["context"].split():
      tokens.append(var(vocab.word_to_index(word, task), "long"))
    tokens.append(var(vocab.SOS_token, "long"))

  for word in source["utterance"].split():
    tokens.append(vocab.word_to_index(word, task))
  return var(tokens, "long")

def prepare_output(target):
  kind = "ordered_values" # "full_enumeration", "possible_only"

  if len(target) == 1:
    target_index = vocab.belief_to_index(target[0], kind)
    output_var = var([target_index], "long")
    return output_var, False
  elif len(target) == 2:
    target_index = vocab.beliefs_to_index(target, kind)
    if kind == "full_enumeration":
      output_var = var([target_index], "long")
      return output_var, False
    else:
      output_vars = [var([ti], "long") for ti in target_index]
      return output_vars, True

def prepare_examples(dataset, use_context, task):
  variables = []

  for example in dataset:
    if task in ["in-car", "babi"]:
      # example is list of tuples, where each tuple is utterance and response
      # [(u1, r1), (u2, r2)...]
      input_output_vars = dialog_to_variable(example, task)
    elif task == "dstc2":
      # Example is dict with keys
      #   {"input_source": [utterance, context, id]}
      #   {"output_target": [list of labels]}
      # where each label is (high level intent, low level intent, slot, value)
      input_var = prepare_input(example["input_source"], use_context, task)
      output_var, double = prepare_output(example["output_target"])
      if double:
        variables.append((input_var, output_var[0]))  # append extra example
        output_var = output_var[1]      # set output to be the second label
    variables.append((input_var, output_var))

  return variables
