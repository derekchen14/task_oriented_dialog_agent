'''
Created on Oct 30, 2017, Updated Mar 11, 2019

An DQN Agent modified for DDQ Agent
@author: Baolin Peng, Xiujun Li, Derek Chen
'''

import random, copy, json
import pickle
import numpy as np
from collections import namedtuple

from utils.external import dialog_constants
from utils.external.dqn import *
from objects.blocks.base import BasePolicyManager
from objects.models.external import DeepDialogDecoder, lstm, biLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cpu')
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))

class DQN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(DQN, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.linear_i2h = nn.Linear(self.input_size, self.hidden_size)
    self.linear_h2o = nn.Linear(self.hidden_size, self.output_size)

  def forward(self, x):
    x = torch.tanh(self.linear_i2h(x))
    x = self.linear_h2o(x)
    return x

  def predict(self, x):
    y = self.forward(x)
    return torch.argmax(y, 1)


class NLG(object):
  def __init__(self, loader, results_dir):
    self.loader = loader
    self.results_dir = results_dir

  def post_process(self, pred_template, slot_val_dict, slot_dict):
    """ post_process to fill the slot in the template sentence """

    sentence = pred_template
    suffix = "_PLACEHOLDER"

    for slot in slot_val_dict.keys():
      slot_vals = slot_val_dict[slot]
      slot_placeholder = slot + suffix
      if slot == 'result' or slot == 'numberofpeople': continue
      if slot_vals == dialog_constants.NO_VALUE_MATCH: continue
      tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
      sentence = tmp_sentence

    if 'numberofpeople' in slot_val_dict.keys():
      slot_vals = slot_val_dict['numberofpeople']
      slot_placeholder = 'numberofpeople' + suffix
      tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
      sentence = tmp_sentence

    for slot in slot_dict.keys():
      slot_placeholder = slot + suffix
      tmp_sentence = sentence.replace(slot_placeholder, '')
      sentence = tmp_sentence

    return sentence


  def generate(self, dia_act, turn_msg):
    """ Convert Dia_Act into NL: Rule + Model """

    sentence = ""
    boolean_in = False

    # remove I do not care slot in task(complete)
    if dia_act['dialogue_act'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] != dialog_constants.NO_VALUE_MATCH:
      inform_slot_set = list(dia_act['inform_slots'].keys()).copy()
      for slot in inform_slot_set:
        if dia_act['inform_slots'][slot] == dialog_constants.I_DO_NOT_CARE:
          del dia_act['inform_slots'][slot]

    if dia_act['dialogue_act'] in self.diaact_nl_pairs['dia_acts'].keys():
      for ele in self.diaact_nl_pairs['dia_acts'][dia_act['dialogue_act']]:
        if set(ele['inform_slots']) == set(dia_act['inform_slots'].keys()) and set(ele['request_slots']) == set(dia_act['request_slots'].keys()):
          sentence = self.diaact_to_nl_slot_filling(dia_act, ele['nl'][turn_msg])
          boolean_in = True
          break

    if dia_act['dialogue_act'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] == dialog_constants.NO_VALUE_MATCH:
      sentence = "Oh sorry, there is no ticket available."

    if boolean_in == False: sentence = self.translate_diaact(dia_act)
    return sentence


  def translate_diaact(self, dia_act):
    """ prepare the diaact into vector representation, and generate the sentence by Model """

    word_dict = self.word_dict
    template_word_dict = self.template_word_dict
    act_dict = self.act_dict
    slot_dict = self.slot_dict
    inverse_word_dict = self.inverse_word_dict

    act_rep = np.zeros((1, len(act_dict)))
    act_rep[0, act_dict[dia_act['dialogue_act']]] = 1.0

    slot_rep_bit = 2
    slot_rep = np.zeros((1, len(slot_dict)*slot_rep_bit))

    suffix = "_PLACEHOLDER"
    if self.params['dia_slot_val'] == 2 or self.params['dia_slot_val'] == 3:
      word_rep = np.zeros((1, len(template_word_dict)))
      words = np.zeros((1, len(template_word_dict)))
      words[0, template_word_dict['s_o_s']] = 1.0
    else:
      word_rep = np.zeros((1, len(word_dict)))
      words = np.zeros((1, len(word_dict)))
      words[0, word_dict['s_o_s']] = 1.0

    for slot in dia_act['inform_slots'].keys():
      slot_index = slot_dict[slot]
      slot_rep[0, slot_index*slot_rep_bit] = 1.0

      for slot_val in dia_act['inform_slots'][slot]:
        if self.params['dia_slot_val'] == 2:
          slot_placeholder = slot + suffix
          if slot_placeholder in template_word_dict.keys():
            word_rep[0, template_word_dict[slot_placeholder]] = 1.0
        elif self.params['dia_slot_val'] == 1:
          if slot_val in word_dict.keys():
            word_rep[0, word_dict[slot_val]] = 1.0

    for slot in dia_act['request_slots'].keys():
      slot_index = slot_dict[slot]
      slot_rep[0, slot_index*slot_rep_bit + 1] = 1.0

    if self.params['dia_slot_val'] == 0 or self.params['dia_slot_val'] == 3:
      final_representation = np.hstack([act_rep, slot_rep])
    else: # dia_slot_val = 1, 2
      final_representation = np.hstack([act_rep, slot_rep, word_rep])

    dia_act_rep = {}
    dia_act_rep['dialogue_act'] = final_representation
    dia_act_rep['words'] = words

    #pred_ys, pred_words = nlg_model['model'].forward(inverse_word_dict, dia_act_rep, nlg_model['params'], predict_model=True)
    pred_ys, pred_words = self.model.beam_forward(inverse_word_dict, dia_act_rep, self.params, predict_model=True)
    pred_sentence = ' '.join(pred_words[:-1])
    sentence = self.post_process(pred_sentence, dia_act['inform_slots'], slot_dict)

    return sentence


  def load_nlg_model(self, model_name):
    """ load the trained NLG model """
    model_params = self.loader.pickle_data(model_name, self.results_dir)

    hidden_size = model_params['model']['Wd'].shape[0]
    output_size = model_params['model']['Wd'].shape[1]

    if model_params['params']['model'] == 'lstm_tanh': # lstm_tanh
      diaact_input_size = model_params['model']['Wah'].shape[0]
      input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
      rnnmodel = DeepDialogDecoder(diaact_input_size, input_size, hidden_size, output_size)

    rnnmodel.model = copy.deepcopy(model_params['model'])
    model_params['params']['beam_size'] = dialog_constants.nlg_beam_size

    self.model = rnnmodel
    self.word_dict = copy.deepcopy(model_params['word_dict'])
    self.template_word_dict = copy.deepcopy(model_params['template_word_dict'])
    self.slot_dict = copy.deepcopy(model_params['slot_dict'])
    self.act_dict = copy.deepcopy(model_params['act_dict'])
    self.inverse_word_dict = {self.template_word_dict[k]:k for k in self.template_word_dict.keys()}
    self.params = copy.deepcopy(model_params['params'])


  def diaact_to_nl_slot_filling(self, dia_act, template_sentence):
    """ Replace the slots with its values """

    sentence = template_sentence.decode("utf-8")
    counter = 0
    for slot in dia_act['inform_slots'].keys():
      slot_val = dia_act['inform_slots'][slot]
      if slot_val == dialog_constants.NO_VALUE_MATCH:
        sentence = slot + " is not available!"
        break
      elif slot_val == dialog_constants.I_DO_NOT_CARE:
        counter += 1
        sentence = sentence.replace('$'+slot+'$', '', 1)
        continue

      sentence = sentence.replace('$'+slot+'$', slot_val, 1)

    if counter > 0 and counter == len(dia_act['inform_slots']):
      sentence = dialog_constants.I_DO_NOT_CARE

    return sentence


  def load_natural_langauge_templates(self, template_path):
    """ Load some pre-defined Dia_Act&NL Pairs from file """
    self.diaact_nl_pairs = self.loader.json_data(template_path)

    for key in self.diaact_nl_pairs['dia_acts'].keys():
      for ele in self.diaact_nl_pairs['dia_acts'][key]:
        ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
        ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue


class NLU(object):
  def __init__(self, loader, results_dir):
    self.loader = loader
    self.results_dir = results_dir

  def load_nlu_model(self, model_name):
    """ load the trained NLU model """
    model_params = self.loader.pickle_data(model_name, self.results_dir)

    hidden_size = model_params['model']['Wd'].shape[0]
    output_size = model_params['model']['Wd'].shape[1]

    if model_params['params']['model'] == 'lstm': # lstm_
      input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
      rnnmodel = lstm(input_size, hidden_size, output_size)
    elif model_params['params']['model'] == 'bi_lstm': # bi_lstm
      input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
      rnnmodel = biLSTM(input_size, hidden_size, output_size)

    rnnmodel.model = copy.deepcopy(model_params['model'])

    self.model = rnnmodel
    self.word_dict = copy.deepcopy(model_params['word_dict'])
    self.slot_dict = copy.deepcopy(model_params['slot_dict'])
    self.act_dict = copy.deepcopy(model_params['act_dict'])
    self.tag_set = copy.deepcopy(model_params['tag_set'])
    self.params = copy.deepcopy(model_params['params'])
    self.inverse_tag_dict = {self.tag_set[k]:k for k in self.tag_set.keys()}

  def classify_intent(self, utterance, agent_action=None):
    """ input utterance is a string and  agent action is a dict with
        'slot_action' as key, the agent action is not used by this model.
        User intent output is dialogue_act, inform_slots and request_slots
    """

    if len(utterance) == 0: return None

    stripped = utterance.strip('.').strip('?').strip(',').strip('!')
    rep = self.embed_to_one_hot(stripped)
    Ys, cache = self.model.fwdPass(rep, self.params, predict_model=True) # default: True

    # manual softmax
    maxes = np.amax(Ys, axis=1, keepdims=True)
    e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
    probs = e/np.sum(e, axis=1, keepdims=True)
    if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)

    # special handling with intent label
    for tag_id in self.inverse_tag_dict.keys():
      if self.inverse_tag_dict[tag_id].startswith('B-') or self.inverse_tag_dict[tag_id].startswith('I-') or self.inverse_tag_dict[tag_id] == 'O':
        probs[-1][tag_id] = 0

    pred_words_indices = np.nanargmax(probs, axis=1)
    pred_tags = [self.inverse_tag_dict[index] for index in pred_words_indices]

    user_intent = self.prepare_dialogue_act(pred_tags, stripped)
    return user_intent

  def embed_to_one_hot(self, utterance):
    """ Parse utterance into one-hot vector representations
        returns a dict containing both the raw sequence
        and the vectorized sentence """
    bookends = 'BOS ' + utterance + ' EOS'
    tokens = bookends.lower().split(' ')

    vecs = np.zeros((len(tokens), len(self.word_dict)))
    for w_index, w in enumerate(tokens):
      if w.endswith(',') or w.endswith('?'): w = w[0:-1]
      if w in self.word_dict.keys():
        vecs[w_index][self.word_dict[w]] = 1
      else: vecs[w_index][self.word_dict['unk']] = 1

    rep = {}
    rep['word_vectors'] = vecs
    rep['raw_seq'] = utterance
    return rep

  def prepare_dialogue_act(self, nlu_vector, utterance):
    """ Parse BIO and Intent into Dia-Act """

    bookends = 'BOS ' + utterance + ' EOS'
    words = bookends.lower().split(' ')

    diaact = {}
    diaact['dialogue_act'] = "inform"
    diaact['request_slots'] = {}
    diaact['inform_slots'] = {}

    intent = nlu_vector[-1]
    index = 1
    pre_tag = nlu_vector[0]
    pre_tag_index = 0

    slot_val_dict = {}

    while index<(len(nlu_vector)-1): # except last Intent tag
      cur_tag = nlu_vector[index]
      if cur_tag == 'O' and pre_tag.startswith('B-'):
        slot = pre_tag.split('-')[1]
        slot_val_str = ' '.join(words[pre_tag_index:index])
        slot_val_dict[slot] = slot_val_str
      elif cur_tag.startswith('B-') and pre_tag.startswith('B-'):
        slot = pre_tag.split('-')[1]
        slot_val_str = ' '.join(words[pre_tag_index:index])
        slot_val_dict[slot] = slot_val_str
      elif cur_tag.startswith('B-') and pre_tag.startswith('I-'):
        if cur_tag.split('-')[1] != pre_tag.split('-')[1]:
          slot = pre_tag.split('-')[1]
          slot_val_str = ' '.join(words[pre_tag_index:index])
          slot_val_dict[slot] = slot_val_str
      elif cur_tag == 'O' and pre_tag.startswith('I-'):
        slot = pre_tag.split('-')[1]
        slot_val_str = ' '.join(words[pre_tag_index:index])
        slot_val_dict[slot] = slot_val_str

      if cur_tag.startswith('B-'): pre_tag_index = index

      pre_tag = cur_tag
      index += 1

    if cur_tag.startswith('B-') or cur_tag.startswith('I-'):
      slot = cur_tag.split('-')[1]
      slot_val_str = ' '.join(words[pre_tag_index:-1])
      slot_val_dict[slot] = slot_val_str

    if intent != 'null':
      arr = intent.split('+')
      diaact['dialogue_act'] = arr[0]
      diaact['request_slots'] = {}
      for ele in arr[1:]:
        #request_slots.append(ele)
        diaact['request_slots'][ele] = 'UNK'

    diaact['inform_slots'] = slot_val_dict

    # add rule here
    for slot in diaact['inform_slots'].keys():
      slot_val = diaact['inform_slots'][slot]
      if slot_val.startswith('bos'):
        slot_val = slot_val.replace('bos', '', 1)
        diaact['inform_slots'][slot] = slot_val.strip(' ')

    self.refine_diaact_by_rules(diaact)
    return diaact

  def refine_diaact_by_rules(self, diaact):
    """ refine the dia_act by rules """

    # rule for taskcomplete
    if 'request_slots' in diaact.keys():
      if 'taskcomplete' in diaact['request_slots'].keys():
        del diaact['request_slots']['taskcomplete']
        diaact['inform_slots']['taskcomplete'] = 'PLACEHOLDER'

      # rule for request
      if len(diaact['request_slots'])>0: diaact['dialogue_act'] = 'request'

      if len(diaact['request_slots'])==0 and diaact['dialogue_act'] == 'request': diaact['dialogue_act'] = 'inform'


  def diaact_penny_string(self, dia_act):
    """ Convert the Dia-Act into penny string """

    penny_str = ""
    penny_str = dia_act['dialogue_act'] + "("
    for slot in dia_act['request_slots'].keys():
      penny_str += slot + ";"

    for slot in dia_act['inform_slots'].keys():
      slot_val_str = slot + "="
      if len(dia_act['inform_slots'][slot]) == 1:
        slot_val_str += dia_act['inform_slots'][slot][0]
      else:
        slot_val_str += "{"
        for slot_val in dia_act['inform_slots'][slot]:
          slot_val_str += slot_val + "#"
        slot_val_str = slot_val_str[:-1]
        slot_val_str += "}"
      penny_str += slot_val_str + ";"

    if penny_str[-1] == ";": penny_str = penny_str[:-1]
    penny_str += ")"
    return penny_str

