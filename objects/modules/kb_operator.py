import numpy as np
import os, pdb, sys  # set_trace
import logging

from torch import nn

class KBOperatorTemplate(object):
  def __init__(self, data):
    self.data = data

  def learn(self):
    raise NotImplementedError

  def predict(self):
    '''
    a knowledge base operator returns system queries:
      input - current user intent in parsed format
      output - list of matching items in the knowledge base
    '''
    raise NotImplementedError


class RuleBeliefTracker(KBOperatorTemplate):
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


class NeuralBeliefTracker(KBOperatorTemplate, nn.module)
  def __init__(self):
    super().__init__()
