import numpy as np
import os, pdb, sys  # set_trace
import logging

from torch import nn

class BeliefTrackerTemplate(object):
  def __init__(self, data):
    self.data = data

  def learn(self):
    raise NotImplementedError

  def predict(self):
    '''
    a belief tracker predicts user intents:
      input - user utterance, previous embedded context, memory
      output - a binary value for each possible slot in the ontology
    '''
    raise NotImplementedError


class RuleBeliefTracker(BeliefTrackerTemplate):
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


class NeuralBeliefTracker(BeliefTrackerTemplate, nn.module)
  def __init__(self):
    super().__init__()
