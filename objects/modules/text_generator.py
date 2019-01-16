import numpy as np
import os, pdb, sys  # set_trace
import logging

from torch import nn

class TextGeneratorTemplate(object):
  def __init__(self, data):
    self.data = data

  def learn(self):
    raise NotImplementedError

  def predict(self):
    '''
    a text generator predicts words until reaching <EOS> token:
      input - agent action and context of previous sentence embedding
      output - natural lanaguage response
    '''
    raise NotImplementedError


class RuleTextGenerator(TextGeneratorTemplate):
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


class NeuralTextGenerator(TextGeneratorTemplate, nn.module)
  def __init__(self):
    super().__init__()
