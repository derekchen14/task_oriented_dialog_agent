import numpy as np
import os, pdb, sys  # set_trace
import logging

from torch import nn

class BaseBeliefTracker(object):
  def __init__(self, data):
    self.data = data
    self.learning_method = "supervised" # or "rulebased" # or "reinforce"

  def learn(self):
    raise NotImplementedError

  def predict(self):
    '''
    a belief tracker predicts user intents:
      input - user utterance, previous embedded context, memory
      output - a binary value for each possible slot in the ontology
    '''
    raise NotImplementedError


class RuleBeliefTracker(BaseBeliefTracker):
  def __init__(self, *args):
    super().__init__(args)

  def learn(self):
    print("rule-based belief tracker has no training")

  def predict(self, examples, batch_size=1):
    if batch_size > 1:  # then examples is a list
      return [self.predict_one(exp) for exp in examples]
    else:               # examples is a single item
      self.predict_one(examples)

  def predict_one(self, example):
    input_text


class NeuralBeliefTracker(BaseBeliefTracker, nn.Module):
  def __init__(self):
    super().__init__()
