import os, pdb, sys  # set_trace
import logging
import copy
import random

class BaseAgent(object):
  """ Prototype for all agent policy models """
  def __init__(self, ontology):
    self.act_set = ontology.acts
    self.slot_set = ontology.slots
    self.relation_set = ontology.relations
    self.value_set = ontology.values
    self.act_cardinality = len(self.act_set)
    self.slot_cardinality = len(self.slot_set)
    # self.agent_run_mode = 0 or 1 and 2
    # self.agent_act_level = 1 or 0

  def initialize_episode(self):
    self.current_slot_id = 0
    self.agent_turn_count = -1

  def state_to_action(self, state, available):
    """ Take the current state and return an action according to the current
    exploration/exploitation policy. We define the agents flexibly so that
    they can either operate on act_slot representations or act_slot_value
    representations. We also define the responses flexibly, returning a
    dictionary with keys [slot_action, slot_value_action].
    This way the command-line agent can continue to operate with values

    Arguments:
    dialogue state -- A dict, see DialogueState class for more information
          1) user_belief or user_intent - a vector with size equal to the
            number of slot-values, where values are one-hot or a distribution
          2) context - vector embedding of previous user utterancees
          3) history - a sequence of previous agent actions
          3) turn_count - scalar value of the turn count
          4) kb_results - number of KB results matching all user constraints
          5) frame - dialogue frame of all previous intents
    available -- a masking vector with 1 for allowed slot-values and 0 otherwize

    Returns:
    slot_action --  An action consisting of one act and >= 0 slots
          as well as which slots are informed vs requested.
    slot_value_action  --  An action consisting of acts slots and values in
          the legacy format. This can be used in the future for training agents
          that take value into account and interact directly with the database
    """
    return {"slot_action": None, "slot_value_action": None}

class InformPolicy(BaseAgent):
  """ A simple agent to test the system. This agent should simply inform
  all the slots and then issue: taskcomplete. """
  def __init__(self, ontology):
    super().__init__(ontology)
    self.model_type = "rulebased"

  def state_to_action(self, state, available=None):
    self.agent_turn_count += 2   # state['turn_count'] += 2
    if self.current_slot_id < self.slot_cardinality:
      slot = self.slot_set[self.current_slot_id]
      self.current_slot_id += 1
      slot_action = {'diaact': "inform",
                          'inform_slots': {slot: "PLACEHOLDER"},
                          'request_slots':  {},
                          'turn_count': self.agent_turn_count }
    else:
      slot_action = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.agent_turn_count }

    return {'slot_action': slot_action, 'slot_value_action': None}


class RequestPolicy(BaseAgent):
  """ A simple agent to test the system. This agent should simply
        request all the slots and then issue: thanks(). """
  def state_to_action(self, state):
    self.state['turn_count'] += 2
    if self.current_slot_id < len(dialog_config.sys_request_slots):
      slot = dialog_config.sys_request_slots[self.current_slot_id]
      self.current_slot_id += 1

      slot_action = {'diaact': "request",
                          'inform_slots': {},
                          'request_slots':  {slot: "PLACEHOLDER"},
                          'turn_count': self.state['turn_count'] }
    else:
      slot_action = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.state['turn_count'] }

    return {'slot_action': slot_action, 'slot_value_action': None}


class RandomPolicy(BaseAgent):
  """ A simple agent to test the interface which chooses actions randomly. """
  def state_to_action(self, state):
    self.state['turn_count'] += 2
    random_action = random.choice(dialog_config.feasible_actions)
    slot_action = copy.deepcopy(random_action)
    slot_action['turn_count'] = self.state['turn_count']

    return {'slot_action': slot_action, 'slot_value_action': None}


class EchoPolicy(BaseAgent):
  """ A simple agent that informs all requested slots,
  then issues inform(taskcomplete) when the user stops making requests. """
  def state_to_action(self, state):
    user_action = state['user_action']
    self.state['turn_count'] += 2
    slot_action = {'diaact': 'thanks',
                        'inform_slots': {},
                        'request_slots':  {},
                        'turn_count': self.state['turn_count'] }
    # find out if the user is requesting anything.  if so, inform it
    if user_action['diaact'] == 'request':
      slot_action['diaact'] = "inform"
      requested_slot = user_action['request_slots'].keys()[0]
      slot_action['inform_slots'][requested_slot] = "PLACEHOLDER"

    return {'slot_action': slot_action, 'slot_value_action': None}


class RequestBasicsPolicy(BaseAgent):
  """ A simple agent to test the system. This agent should simply
      request all the basic slots and then issue: thanks().

      Now there are "phases"???
      """

  def __init__(self, movie_dict=None, act_set=None, slot_set=None,
              params=None, request_set=None):
    self.request_set = request_set
    #self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
    #self.request_set = ["restaurantname", "date", "numberofpeople", "starttime", "address"]

  def initialize_episode(self):
    self.state = {'diaact': 'UNK',
                  'inform_slots': {},
                  'request_slots': {},
                  'turn_count': -1  }
    self.current_slot_id = 0
    self.phase = 0

  def state_to_action(self, state):
    self.state['turn_count'] += 2
    if self.current_slot_id < len(self.request_set):
      slot = self.request_set[self.current_slot_id]
      self.current_slot_id += 1

      slot_action = {'diaact': "request",
                          'inform_slots': {},
                          'request_slots': {slot: "UNK"},
                          'turn_count': self.state['turn_count'] }
    elif self.phase == 0:
      slot_action = {'diaact': "inform",
                          'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                          'request_slots': {},
                          'turn_count':self.state['turn_count'] }
      self.phase += 1
    elif self.phase == 1:
      slot_action = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.state['turn_count'] }
    else:
      raise Exception("IS NOT POSSIBLE! (agent called in unexpected way)")

    return {'slot_action': slot_action, 'slot_value_action': None}

class BasicsPolicy(BaseAgent):
  """ This agent should simply request and inform all the basic slots
  and then issue: thanks(). """

  def __init__(self, movie_dict=None, act_set=None, slot_set=None, 
              params=None, request_set=None, inform_set=None):
    self.request_set = request_set
    self.inform_set = inform_set
    #self.request_set = ['or_city', 'dst_city', 'seat', 'depart_date_dep', 'depart_time_dep', 'return_date_dep', 'return_time_dep', 'numberofpeople','hotel_name', 'hotel_city', 'hotel_numberofpeople', 'hotel_date_checkin', 'hotel_date_checkout']
    #self.inform_set = ['or_city', 'dst_city', 'seat', 'depart_date_dep', 'depart_time_dep', 'return_date_dep', 'return_time_dep','price', 'hotel_name', 'hotel_city', 'hotel_date_checkin', 'hotel_date_checkout', 'hotel_price']

  def initialize_episode(self):
    self.state = {'diaact': 'UNK',
                  'inform_slots': {},
                  'request_slots': {},
                  'turn_count': -1  }
    self.current_request_slot_id = 0
    self.current_inform_slot_id = 0
    self.phase = 0

  def state_to_action(self, state):
    self.state['turn_count'] += 2
    if self.current_request_slot_id < len(self.request_set):
      slot = self.request_set[self.current_request_slot_id]
      self.current_request_slot_id += 1
      slot_action = {'diaact': "request",
                          'inform_slots': {},
                          'request_slots': {slot: "PLACEHOLDER"},
                          'turn_count': self.state['turn_count'] }
    elif self.current_inform_slot_id < len(self.inform_set):
      slot = self.inform_set[self.current_inform_slot_id]
      self.current_inform_slot_id += 1
      slot_action = {'diaact': "inform",
                          'inform_slots': {slot: "PLACEHOLDER"},
                          'request_slots': {},
                          'turn_count': self.state['turn_count'] }
    elif self.phase == 0:
      slot_action = {'diaact': "inform",
                          'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                          'request_slots': {},
                          'turn_count':self.state['turn_count'] }
      self.phase += 1
    elif self.phase == 1:
      slot_action = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.state['turn_count'] }
    else:
      raise Exception("IS NOT POSSIBLE! (agent called in unexpected way)")

    return {'slot_action': slot_action, 'slot_value_action': None}
