import numpy as np
import os, pdb, sys  # set_trace
import logging

from torch import nn

class PolicyManagerTemplate(object):
  def __init__(self, data):
    self.data = data
    self.experiences = []  # tuples of state, action, reward, next state

  def learn(self):
    raise NotImplementedError

  def predict(self):
    '''
    a POMDP takes in the dialogue state with latent intent
      input - dialogue state consisting of:
        1) current user intent --> act(slot-relation-value) + confidence score
        2) previous agent action
        3) knowledge base query results
        4) turn count
        5) complete semantic frame
      output - next agent action
    '''
    raise NotImplementedError


class RulePolicyManager(PolicyManagerTemplate):
  def __init__(self, *args):
    super().__init__(args)

  def learn(self):
    print("rule-based policy manager has no training")

  def predict(self, examples, batch_size=1):
    if batch_size > 1:  # then examples is a list
      return [self.predict_one(exp) for exp in examples]
    else:               # examples is a single item
      self.predict_one(examples)

  def predict_one(self, example):
    input_text




class NeuralPolicyManager(PolicyManagerTemplate, nn.Module)
  def __init__(self):
    super().__init__()
