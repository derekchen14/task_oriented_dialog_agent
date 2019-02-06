import numpy as np
import os, pdb, sys  # set_trace
import logging

from torch import nn

import copy
from collections import defaultdict
import datasets.ddq.constants as dialog_config

class KBHelper(object):
  """ An assistant to fill in values for the agent (which knows about slots of values) """

  def __init__(self, knowledge_base):
    """ Constructor for a KBHelper """

    self.knowledge_base = knowledge_base
    self.cached_kb = defaultdict(list)
    self.cached_kb_slot = defaultdict(list)


  def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
    """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)

    Arguments:
    inform_slots_to_be_filled   --  Something that looks like {starttime:None, theater:None} where starttime and theater are slots that the agent needs filled
    current_slots               --  Contains a record of all filled slots in the conversation so far - for now, just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots

    Returns:
    filled_in_slots             --  A dictionary of form {slot1:value1, slot2:value2} for each sloti in inform_slots_to_be_filled
    """

    kb_results = self.available_results_from_kb(current_slots)
    #if dialog_config.auto_suggest == 1:
    #    print 'Number of entries in KB satisfying current constraints: ', len(kb_results)

    filled_in_slots = {}
    if 'taskcomplete' in inform_slots_to_be_filled.keys():
      filled_in_slots.update(current_slots['inform_slots'])

    for slot in inform_slots_to_be_filled.keys():
      if slot == 'numberofpeople':
        if slot in current_slots['inform_slots'].keys():
          filled_in_slots[slot] = current_slots['inform_slots'][slot]
        elif slot in inform_slots_to_be_filled.keys():
          filled_in_slots[slot] = inform_slots_to_be_filled[slot]
        continue

      if slot == 'ticket' or slot == 'reservation' or slot == 'taxi' or slot == 'taskcomplete':
        filled_in_slots[slot] = dialog_config.TICKET_AVAILABLE if len(kb_results)>0 else dialog_config.NO_VALUE_MATCH
        continue

      if slot == 'closing': continue

      ####################################################################
      #   Grab the value for the slot with the highest count and fill it
      ####################################################################
      values_dict = self.available_slot_values(slot, kb_results)

      values_counts = [(v, values_dict[v]) for v in values_dict.keys()]
      if len(values_counts) > 0:
        if inform_slots_to_be_filled[slot] == "PLACEHOLDER":
          filled_in_slots[slot] = sorted(values_counts, key = lambda x: -x[1])[0][0] # something wrong here
        else:
          filled_in_slots[slot] = inform_slots_to_be_filled[slot]
      else:
        filled_in_slots[slot] = dialog_config.NO_VALUE_MATCH #"NO VALUE MATCHES SNAFU!!!"

    return filled_in_slots


  def available_slot_values(self, slot, kb_results):
    """ Return the set of values available for the slot based on the current constraints """

    slot_values = {}
    for movie_id in kb_results.keys():
      if slot in kb_results[movie_id].keys():
        slot_val = kb_results[movie_id][slot]
        if slot_val in slot_values.keys():
          slot_values[slot_val] += 1
        else: slot_values[slot_val] = 1
    return slot_values

  def available_results_from_kb(self, current_slots):
    """ Return the available movies in the movie_kb based on the current constraints """

    ret_result = []
    current_slots = current_slots['inform_slots']
    constrain_keys = current_slots.keys()

    constrain_keys = filter(lambda k : k != 'reservation' and \
                       k != 'numberofpeople' and \
                       k != 'taskcomplete' and \
                       k != 'occasion' and \
                       k != 'personfullname' and \
                       k != 'mealtype' and \
                       k != 'restauranttype' and \
                       k != 'greeting' and \
                       k != 'distanceconstraints' and \
                       k != 'other' and \
                       k != 'name' and \
                       k != 'taxi' and \
                       k != 'closing', constrain_keys)
    constrain_keys = [k for k in constrain_keys if current_slots[k] != dialog_config.I_DO_NOT_CARE]

    query_idx_keys = frozenset(current_slots.items())
    cached_kb_ret = self.cached_kb[query_idx_keys]

    cached_kb_length = len(cached_kb_ret) if cached_kb_ret != None else -1
    if cached_kb_length > 0:
      return dict(cached_kb_ret)
    elif cached_kb_length == -1:
      return dict([])

    # kb_results = copy.deepcopy(self.knowledge_base)
    for id in self.knowledge_base.keys():
      kb_keys = self.knowledge_base[id].keys()
      if len(set(constrain_keys).union(set(kb_keys)) ^ (set(constrain_keys) ^ set(kb_keys))) == len(constrain_keys):
        match = True
        for idx, k in enumerate(constrain_keys):
          if str(current_slots[k]).lower() == str(self.knowledge_base[id][k]).lower():
            continue
          else:
            match = False

        if match:
          self.cached_kb[query_idx_keys].append((id, self.knowledge_base[id]))
          ret_result.append((id, self.knowledge_base[id]))
          #print 'res', id

      # for slot in current_slots['inform_slots'].keys():
      #     if slot == 'ticket' or slot == 'numberofpeople' or slot == 'taskcomplete' or slot == 'closing': continue
      #     if current_slots['inform_slots'][slot] == dialog_config.I_DO_NOT_CARE: continue
      #
      #     if slot not in self.knowledge_base[movie_id].keys():
      #         if movie_id in kb_results.keys():
      #             del kb_results[movie_id]
      #     else:
      #         if current_slots['inform_slots'][slot].lower() != self.knowledge_base[movie_id][slot].lower():
      #             if movie_id in kb_results.keys():
      #                 del kb_results[movie_id]

    if len(ret_result) == 0:
      self.cached_kb[query_idx_keys] = None

    ret_result = dict(ret_result)
    return ret_result

  def available_results_from_kb_for_slots(self, inform_slots):
    """ Return the count statistics for each constraint in inform_slots """

    kb_results = {key:0 for key in inform_slots.keys()}
    kb_results['matching_all_constraints'] = 0

    query_idx_keys = frozenset(inform_slots.items())
    cached_kb_slot_ret = self.cached_kb_slot[query_idx_keys]

    if len(cached_kb_slot_ret) > 0:
      return cached_kb_slot_ret[0]

    for movie_id in self.knowledge_base.keys():
      all_slots_match = True
      for slot in inform_slots.keys():
        desired_value = inform_slots[slot]
        movie = self.knowledge_base[movie_id]
        if slot == 'taxi' or desired_value == dialog_config.I_DO_NOT_CARE:
          continue

        if slot in movie.keys():
          if desired_value.lower() == movie[slot].lower():
            kb_results[slot] += 1
          else:
            all_slots_match = False
        else:
          all_slots_match = False

      if all_slots_match:
        kb_results['matching_all_constraints'] += 1

    self.cached_kb_slot[query_idx_keys].append(kb_results)
    return kb_results


  def database_results_for_agent(self, current_slots):
    """ A dictionary of the number of results matching each current constraint. The agent needs this to decide what to do next. """

    database_results ={} # { date:100, distanceconstraints:60, theater:30,  matching_all_constraints: 5}
    database_results = self.available_results_from_kb_for_slots(current_slots['inform_slots'])
    return database_results

  def suggest_slot_values(self, request_slots, current_slots):
    """ Return the suggest slot values """

    avail_kb_results = self.available_results_from_kb(current_slots)

    #print 'request_slot', len(avail_kb_results)
    #print avail_kb_results

    return_suggest_slot_vals = {}
    for slot in request_slots.keys():
      avail_values_dict = self.available_slot_values(slot, avail_kb_results)
      values_counts = [(v, avail_values_dict[v]) for v in avail_values_dict.keys()]

      if len(values_counts) > 0:
        return_suggest_slot_vals[slot] = []
        sorted_dict = sorted(values_counts, key = lambda x: -x[1])
        for k in sorted_dict: return_suggest_slot_vals[slot].append(k[0])
      else:
        return_suggest_slot_vals[slot] = []

    return return_suggest_slot_vals

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


class NeuralBeliefTracker(KBOperatorTemplate, nn.Module):
  def __init__(self):
    super().__init__()
