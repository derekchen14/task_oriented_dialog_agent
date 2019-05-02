import numpy as np
import os, pdb, sys  # set_trace

from torch import nn
from nltk import word_tokenize
from objects.blocks.base import BaseBeliefTracker
from objects.blocks import Turn

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
    self.vocab = None

  def extract_beliefs(self, scores, threshold=0.01):
    batch_size = len(list(scores.values())[0])
    predictions = [set() for i in range(batch_size)]
    ont = self.model.ontology
    for s in ont.slots:
      # idx is an index of the batch size
      for idx, p in enumerate(scores[s]):
        triggered = [(s, v, p_v) for v, p_v in zip(ont.values[s], p) if p_v > threshold]
        if batch_size == 1: # grab all triggered values, along with the score
          for belief in triggered:
            predictions[idx].add(belief)
        elif s == 'request':
          # we can have multiple requests predictions
          predictions[idx] |= set([(s, v) for s, v, p_v in triggered])
        elif triggered:
          # only extract the top inform prediction
          sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
          predictions[idx].add((sort[0][0], sort[0][1]))
    return predictions

  def extract_predictions(self, scores, threshold=0.5):
    batch_size = len(list(scores.values())[0])
    predictions = [set() for i in range(batch_size)]
    ont = self.model.ontology
    for s in ont.slots:
      # idx is an index of the batch size
      for idx, p in enumerate(scores[s]):
        triggered = [(s, v, p_v) for v, p_v in zip(ont.values[s], p) if p_v > threshold]
        if batch_size == 1: # grab all triggered values, along with the score
          for belief in triggered:
            predictions[idx].add(belief)
        elif s == 'request':
          # we can have multiple requests predictions
          predictions[idx] |= set([(s, v) for s, v, p_v in triggered])
        elif triggered:
          # only extract the top inform prediction
          sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
          predictions[idx].add((sort[0][0], sort[0][1]))
    return predictions

  def run_glad_inference(self, data):
    self.model.eval()
    predictions = []
    for batch in data.batch(self.batch_size):
      loss, scores = self.model(batch)
      predictions += self.extract_predictions(scores)
    return predictions

  def classify_intent(self, utterance, agent_action=None):
    exp = {'turn_id': 14, 'utterance': '', 'user_intent': {}, 'num': {},
      'belief_state': {}, 'agent_actions': {}, 'agent_utterance': {}}

    tokens = utterance.replace(':', ' ').split()
    cleaned = [token.rstrip(',').rstrip('.') for token in tokens]
    exp['utterance'] = ['<sos>'] + cleaned + ['<eos>']

    exp['num']['utterance'] = [self.w2i(word) for word in exp['utterance']]
    exp['num']['agent_actions'] = self.scrub(agent_action)

    turn = Turn.from_dict(exp)
    _, scores = self.model([turn])
    predictions = self.extract_beliefs(scores)

    user_beliefs = {'dialogue_act': 'inform'}
    assert len(self.model.ontology.slots) == 10

    for slot_type in self.model.ontology.slots:
      user_beliefs[f'{slot_type}_slots'] = []
    for slot, value, score in predictions[0]:
      user_beliefs[f'{slot}_slots'].append((value,score))
      if slot == 'request' and score > 0.5:
        user_beliefs['dialogue_act'] = 'request'

    return user_beliefs


  def scrub(self, agent_action):

    def vectorize(text):
      return [self.w2i(word) for word in text.split()]
    extracted = []

    if agent_action is None:
      text = f'<sos> <special2> = <special2> <eos>'
      extracted.append(vectorize(text))
      return extracted

    for slot, val in agent_action['slot_action']['inform_slots'].items():
      text = f'<sos> {slot} = {val} <eos>'
      extracted.append(vectorize(text))
    for slot, val in agent_action['slot_action']['request_slots'].items():
      text = f'<sos> request = {slot} <eos>'
      extracted.append(vectorize(text))

    dialogue_act = agent_action['slot_action']['dialogue_act']
    if dialogue_act != 'request' and dialogue_act != 'inform':
      text = f'<sos> act = {dialogue_act} <eos>'
      extracted.append(vectorize(text))

    return extracted

"""
{
  "current_slots": {
    "inform_slots": {
      "numberofpeople": "3",
      "moviename": "risen"
    },
    "request_slots": {
      "starttime": "UNK"
    },
    "act_slots": {},
    "proposed_slots": {},
    "agent_request_slots": {}
  },
  "kb_results_dict": {
    "numberofpeople": 0,
    "moviename": 12,
    "matching_all_constraints": 0
  },
  "turn_count": 1,
  "history": [
    {
      "turn_count": 0,
      "speaker": "user",
      "request_slots": {
        "starttime": "UNK"
      },
      "inform_slots": {
        "numberofpeople": "3",
        "moviename": "risen"
      },
      "dialogue_act": "request"
    }
  ],
  "user_action": {
    "turn_count": 0,
    "speaker": "user",
    "request_slots": {
      "starttime": "UNK"
    },
    "inform_slots": {
      "numberofpeople": "3",
      "moviename": "risen"
    },
    "dialogue_act": "request"
  },
  "agent_action": null
}


{
  "current_slots": {
    "inform_slots": {
      "numberofpeople": "3",
      "moviename": "risen"
    },
    "request_slots": {
      "starttime": "UNK"
    },
    "act_slots": {},
    "proposed_slots": {},
    "agent_request_slots": {}
  },
  "kb_results_dict": {
    "numberofpeople": 0,
    "moviename": 12,
    "matching_all_constraints": 0
  },
  "turn_count": 1,
  "history": [
    {
      "turn_count": 0,
      "speaker": "user",
      "request_slots": {
        "starttime": "UNK"
      },
      "inform_slots": {
        "numberofpeople": "3",
        "moviename": "risen"
      },
      "dialogue_act": "request"
    }
  ],
  "user_action": {
    "turn_count": 0,
    "speaker": "user",
    "request_slots": {
      "starttime": "UNK"
    },
    "inform_slots": {
      "numberofpeople": "3",
      "moviename": "risen"
    },
    "dialogue_act": "request"
  },
  "agent_action": null
}

{
  "dialogue_act": "inform",
  "inform_slots": {
    "moviename": "london has fallen"
  },
  "request_slots": {},
  "turn_count": 2,
  "nl": "I want to watch london has fallen."
}
"""