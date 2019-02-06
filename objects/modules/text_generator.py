import numpy as np
import os, pdb, sys  # set_trace
import logging
import copy
import json
import pickle as pkl

from torch import nn
from objects.models.external import DeepDialogDecoder
from datasets.ddq import constants as dialog_config

class TextGenerator(object):

  def __init__(self, model, dictionaries, acts, slots, parameters):
    self.model = model
    self.word_dict = dictionaries['word']
    self.template_word_dict = dictionaries['template']
    self.inverse_word_dict = dictionaries['inverse']

    self.act_dict = acts
    self.slot_dict = slots
    self.params = parameters
    self.learning_method = "rulebased" # or "reinforce" or "supervised"

  def post_process(self, pred_template, slot_val_dict, slot_dict):
    """ post_process to fill the slot in the template sentence """

    sentence = pred_template
    suffix = "_PLACEHOLDER"

    for slot in slot_val_dict.keys():
      slot_vals = slot_val_dict[slot]
      slot_placeholder = slot + suffix
      if slot == 'result' or slot == 'numberofpeople': continue
      if slot_vals == dialog_config.NO_VALUE_MATCH: continue
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
    """ convert_diaact_to_nl(self, dia_act, turn_msg):
    Convert Dia_Act into NL: Rule + Model """

    sentence = ""
    boolean_in = False

    # remove I do not care slot in task(complete)
    if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] != dialog_config.NO_VALUE_MATCH:
      inform_slot_set = dia_act['inform_slots'].keys()
      for slot in inform_slot_set:
        if dia_act['inform_slots'][slot] == dialog_config.I_DO_NOT_CARE: del dia_act['inform_slots'][slot]

    if dia_act['diaact'] in self.diaact_nl_pairs['dia_acts'].keys():
      for ele in self.diaact_nl_pairs['dia_acts'][dia_act['diaact']]:
        if set(ele['inform_slots']) == set(dia_act['inform_slots'].keys()) and set(ele['request_slots']) == set(dia_act['request_slots'].keys()):
          sentence = self.diaact_to_nl_slot_filling(dia_act, ele['nl'][turn_msg])
          boolean_in = True
          break

    if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
      sentence = "Oh sorry, there is no ticket available."

      if 'restaurantname' in dia_act['inform_slots'].keys(): sentence = "Oh sorry, there is no restaurant available."
      if 'pickup_location' in dia_act['inform_slots'].keys(): sentence = "Oh sorry, there is no taxi available."

    if boolean_in == False: sentence = self.translate_diaact(dia_act)
    return sentence


  def translate_diaact(self, dia_act):
    """ prepare the diaact into vector representation, and generate the sentence by Model """
    if self.params['dia_slot_val'] != 1:
      if not hasattr(self, "nlg_cache"):
        self.nlg_cache = {}
      tmp_dia_act = copy.deepcopy(dia_act)
      tmp_dia_act['inform_slots'] = {inform_slot_name: "" for inform_slot_name in tmp_dia_act['inform_slots'].keys()}
      dia_act_key = repr(to_consistent_data_structure(tmp_dia_act))
      pred_sentence = self.nlg_cache.get(dia_act_key, None)
      if pred_sentence is not None:
        sentence = self.post_process(pred_sentence, dia_act['inform_slots'], self.slot_dict)
        return sentence

    word_dict = self.word_dict
    template_word_dict = self.template_word_dict
    act_dict = self.act_dict
    slot_dict = self.slot_dict
    inverse_word_dict = self.inverse_word_dict

    act_rep = np.zeros((1, len(act_dict)))
    act_rep[0, act_dict[dia_act['diaact']]] = 1.0

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
    dia_act_rep['diaact'] = final_representation
    dia_act_rep['words'] = words

    #pred_ys, pred_words = nlg_model['model'].forward(inverse_word_dict, dia_act_rep, nlg_model['params'], predict_model=True)
    pred_ys, pred_words = self.model.beam_forward(inverse_word_dict, dia_act_rep, self.params, predict_model=True)
    pred_sentence = ' '.join(pred_words[:-1])

    if self.params['dia_slot_val'] != 1:
      self.nlg_cache[dia_act_key] = pred_sentence

    sentence = self.post_process(pred_sentence, dia_act['inform_slots'], slot_dict)
    return sentence

  def diaact_to_nl_slot_filling(self, dia_act, template_sentence):
    """ Replace the slots with its values """

    sentence = template_sentence.decode("utf-8")
    counter = 0
    for slot in dia_act['inform_slots'].keys():
      slot_val = dia_act['inform_slots'][slot]
      if slot_val == dialog_config.NO_VALUE_MATCH:
        sentence = slot + " is not available!"
        break
      elif slot_val == dialog_config.I_DO_NOT_CARE:
        counter += 1
        sentence = sentence.replace('$'+slot+'$', '', 1)
        continue

      sentence = sentence.replace('$'+slot+'$', slot_val, 1)

    if counter > 0 and counter == len(dia_act['inform_slots']):
      sentence = dialog_config.I_DO_NOT_CARE

    return sentence

  def set_templates(self, dataset_type):
    """ Set some pre-defined Dia_Act&NL Pairs from loaded file """
    data_dir = os.path.join('datasets', dataset_type)
    template_path =  os.path.join(data_dir, 'dia_act_nl_pairs.v6.json')
    self.diaact_nl_pairs = json.load(open(template_path, 'rb'))

    for key in self.diaact_nl_pairs['dia_acts'].keys():
      for ele in self.diaact_nl_pairs['dia_acts'][key]:
        ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
        ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue

  @classmethod
  def from_pretrained(cls, args):
    model_name = 'lstm_tanh_relu_[1468202263.38]_2_0.610.p'
    results_dir = os.path.join('results', args.task, args.dataset, 'models')
    model_path =  os.path.join(results_dir, model_name)
    model_params = pkl.load(open(model_path, 'rb'), encoding='latin1')

    hidden_size = model_params['model']['Wd'].shape[0]
    output_size = model_params['model']['Wd'].shape[1]
    # lstm_decoder_tanh
    if model_params['params']['model'] == 'lstm_tanh':
      diaact_input_size = model_params['model']['Wah'].shape[0]
      input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
      rnn_model = DeepDialogDecoder(diaact_input_size, input_size, hidden_size, output_size)
    rnn_model.model = copy.deepcopy(model_params['model'])

    word = copy.deepcopy(model_params['word_dict'])
    template = copy.deepcopy(model_params['template_word_dict'])
    inverse = {template[k]:k for k in template.keys()}
    dictionaries = { "word": word, "inverse": inverse, "template": template }

    acts = copy.deepcopy(model_params['act_dict'])
    slots = copy.deepcopy(model_params['slot_dict'])
    model_params['params']['beam_size'] = 10 # dialog_config.nlg_beam_size
    parameters = copy.deepcopy(model_params['params'])

    return cls(rnn_model, dictionaries, acts, slots, parameters)

  def predict(self):
    ''' a text generator predicts words until reaching <EOS> token:
      input - agent action and context of previous sentence embedding
      output - natural lanaguage response '''
    pass


class NeuralTextGenerator(nn.Module):
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


def to_consistent_data_structure(obj):
  """obj could be set, dictionary, list, tuple or nested of them.
  This function will convert all dictionaries inside the obj to be list of tuples (sorted by key),
  will convert all set inside the obj to be list (sorted by to_consistent_data_structure(value))

  >>> to_consistent_data_structure([
    {"a" : 3, "b": 4},
    ( {"e" : 5}, (6, 7)
    ),
    set([10, ]),
    11
  ])

  Out[2]: [[('a', 3), ('b', 4)], ([('e', 5)], (6, 7)), [10], 11]
  """

  if isinstance(obj, dict):
    return [(k, to_consistent_data_structure(v)) for k, v in sorted(list(obj.items()), key=lambda x: x[0])]
  elif isinstance(obj, set):
    return sorted([to_consistent_data_structure(v) for v in obj])
  elif isinstance(obj, list):
    return [to_consistent_data_structure(v) for v in obj]
  elif isinstance(obj, tuple):
    return tuple([to_consistent_data_structure(v) for v in obj])
  return obj