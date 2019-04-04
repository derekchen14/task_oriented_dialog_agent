import numpy as np
import os, pdb, sys  # set_trace

from torch import nn
from objects.blocks.base import BaseBeliefTracker

'''
class BaseBeliefTracker(object):
  def __init__(self, data):
    self.data = data
    self.learning_method = "supervised" # or "rulebased" # or "reinforce"

  def learn(self):
    raise NotImplementedError

  def predict(self):
    a belief tracker predicts user intents:
      input - user utterance, previous embedded context, memory
      output - a binary value for each possible slot in the ontology
    raise NotImplementedError
'''


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


class NeuralBeliefTracker(BaseBeliefTracker):
  def __init__(self, args, model):
    super().__init__(args, model)

  def extract_predictions(self, scores, threshold=0.5):
    batch_size = len(list(scores.values())[0])
    predictions = [set() for i in range(batch_size)]
    for s in self.ontology.slots:
      for i, p in enumerate(scores[s]):
        triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > threshold]
        if s == 'request':
          # we can have multiple requests predictions
          predictions[i] |= set([(s, v) for s, v, p_v in triggered])
        elif triggered:
          # only extract the top inform prediction
          sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
          predictions[i].add((sort[0][0], sort[0][1]))
    return predictions

  def run_glad_inference(self, data):
    self.eval()
    predictions = []
    for batch in data.batch(self.batch_size):
      loss, scores = self.model(batch)
      predictions += self.extract_predictions(scores)
    return predictions

