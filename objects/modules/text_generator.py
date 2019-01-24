import numpy as np
import os, pdb, sys  # set_trace
import logging
import copy
import json

from torch import nn
from objects.models.external import DeepDialogDecoder

class BaseTextGenerator(object):
  def __init__(self, model, dictionaries, acts, slots, params):
    self.model = model
    self.word_dict = dictionaries['word']
    self.template_word_dict = dictionaries['template']
    self.inverse_word_dict = dictionaries['inverse']

    self.act_dict = acts
    self.slot_dict = slots
    self.params = params
    # self.data = data
    self.learning_method = "rulebased" # or "reinforce" or "supervised"

  def learn(self):
    raise NotImplementedError

  def predict(self):
    '''
    a text generator predicts words until reaching <EOS> token:
      input - agent action and context of previous sentence embedding
      output - natural lanaguage response
    '''
    raise NotImplementedError

  @classmethod
  def from_params(cls, model_params):
    hidden_size = model_params['model']['Wd'].shape[0]
    output_size = model_params['model']['Wd'].shape[1]
    # lstm_decoder_tanh
    if model_params['params']['model'] == 'lstm_tanh':
      diaact_input_size = model_params['model']['Wah'].shape[0]
      input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
      rnnmodel = DeepDialogDecoder(diaact_input_size, input_size, hidden_size, output_size)
    rnn_model.model = copy.deepcopy(model_params['model'])

    word = copy.deepcopy(model_params['word_dict'])
    template = copy.deepcopy(model_params['template_word_dict'])
    inverse = {self.template_word_dict[k]:k for k in self.template_word_dict.keys()}
    dictionaries = { "word": word, "inverse": inverse, "template": template }

    acts = copy.deepcopy(model_params['act_dict'])
    slots = copy.deepcopy(model_params['slot_dict'])
    model_params['params']['beam_size'] = 10 # dialog_config.nlg_beam_size

    return cls(rnn_model, dictionaries, acts, slots, model_params)


  def set_templates(self, da_nl_pairs):
    # Load some pre-defined Dialogue Act & Natural Language Pairs from file
    self.diaact_nl_pairs = da_nl_pairs

    for key in self.diaact_nl_pairs['dia_acts'].keys():
      for ele in self.diaact_nl_pairs['dia_acts'][key]:
        ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
        ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue



class RuleTextGenerator(BaseTextGenerator):
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


class NeuralTextGenerator(BaseTextGenerator, nn.Module):
  def __init__(self):
    super().__init__()
