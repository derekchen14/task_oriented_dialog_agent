import os, pdb, sys  # set_trace
import logging
import json
import copy
import random
import datasets.ddq.constants as dialog_config

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
  def state_to_action(self, state, available=None):
    self.agent_turn_count += 2
    allowed_slots = dialog_config.movie_agent_inform_slots
    if self.current_slot_id < len(allowed_slots):
      slot = allowed_slots[self.current_slot_id]
      self.current_slot_id += 1
      slot_action = {'dialogue_act': "inform",
                          'inform_slots': {slot: "PLACEHOLDER"},
                          'request_slots':  {},
                          'turn_count': self.agent_turn_count }
    else:
      slot_action = {'dialogue_act': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.agent_turn_count }

    return {'slot_action': slot_action, 'slot_value_action': None}

class RequestPolicy(BaseAgent):
  """ A simple agent to test the system. This agent should simply
        request all the slots and then issue: thanks(). """
  def state_to_action(self, state):
    self.agent_turn_count += 2
    allowed_slots = dialog_config.movie_agent_request_slots
    if self.current_slot_id < len(allowed_slots):
      slot = allowed_slots[self.current_slot_id]
      self.current_slot_id += 1

      slot_action = {'dialogue_act': "request",
                          'inform_slots': {},
                          'request_slots':  {slot: "PLACEHOLDER"},
                          'turn_count': self.agent_turn_count }
    else:
      slot_action = {'dialogue_act': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.agent_turn_count }

    return {'slot_action': slot_action, 'slot_value_action': None}

class RandomPolicy(BaseAgent):
  """ A simple agent to test the interface which chooses actions randomly. """
  def state_to_action(self, state):
    self.agent_turn_count += 2
    random_action = random.choice(dialog_config.feasible_actions)
    slot_action = copy.deepcopy(random_action)
    slot_action['turn_count'] = self.agent_turn_count

    return {'slot_action': slot_action, 'slot_value_action': None}

class EchoPolicy(BaseAgent):
  """ A simple agent that informs all requested slots,
  then issues inform(taskcomplete) when the user stops making requests. """
  def state_to_action(self, state):
    user_action = state['user_action']
    self.agent_turn_count += 2
    slot_action = {'dialogue_act': 'thanks',
                        'inform_slots': {},
                        'request_slots':  {},
                        'turn_count': self.agent_turn_count }
    # find out if the user is requesting anything.  if so, inform it
    if user_action['dialogue_act'] == 'request':
      slot_action['dialogue_act'] = "inform"
      requested_slot = list(user_action['request_slots'].keys())[0]
      slot_action['inform_slots'][requested_slot] = "PLACEHOLDER"

    return {'slot_action': slot_action, 'slot_value_action': None}

class BasicsPolicy(BaseAgent):
  """ A simple agent to test the system. This agent should simply
      request all the basic slots and then issue: thanks(). """

  def __init__(self, ontology):
    super().__init__(ontology)
    self.request_set = dialog_config.movie_agent_request_slots
    self.inform_set = dialog_config.movie_agent_inform_slots
    self.complete = False

  def state_to_action(self, state):
    self.agent_turn_count += 2
    if self.current_slot_id < len(self.request_set) -14:
      slot = self.request_set[self.current_slot_id]
      self.current_slot_id += 1
      slot_action = {'dialogue_act': "request",
                          'inform_slots': {},
                          'request_slots': {slot: "UNK"},
                          'turn_count': self.agent_turn_count }
    elif not self.complete:
      self.complete = True
      slot_action = {'dialogue_act': "inform",
                          'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                          'request_slots': {},
                          'turn_count':self.agent_turn_count }
    elif self.complete:
      slot_action = {'dialogue_act': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.agent_turn_count }
    else:
      raise Exception("IS NOT POSSIBLE! (agent called in unexpected way)")

    return {'slot_action': slot_action, 'slot_value_action': None}

class RequestThenInformPolicy(BaseAgent):
  """ This agent requests and informs basic slots and then concludes. """

  def __init__(self, ontology):
    super().__init__(ontology)
    self.request_set = dialog_config.movie_agent_request_slots
    self.inform_set = dialog_config.movie_agent_inform_slots
    self.complete = False

  def initialize_episode(self):
    self.request_slot_id = 0
    self.inform_slot_id = 0
    self.agent_turn_count = -1

  def state_to_action(self, state):
    self.agent_turn_count += 2
    if self.request_slot_id < len(self.request_set) -14:
      slot = self.request_set[self.request_slot_id]
      self.request_slot_id += 1
      slot_action = {'dialogue_act': "request",
                          'inform_slots': {},
                          'request_slots': {slot: "PLACEHOLDER"},
                          'turn_count': self.agent_turn_count }
    elif self.inform_slot_id < len(self.inform_set) -14:
      slot = self.inform_set[self.inform_slot_id]
      self.inform_slot_id += 1
      slot_action = {'dialogue_act': "inform",
                          'inform_slots': {slot: "PLACEHOLDER"},
                          'request_slots': {},
                          'turn_count': self.agent_turn_count }
    elif not self.complete:
      self.complete = True
      slot_action = {'dialogue_act': "inform",
                          'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                          'request_slots': {},
                          'turn_count':self.agent_turn_count }
    elif self.complete:
      slot_action = {'dialogue_act': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn_count': self.agent_turn_count }
    else:
      raise Exception("Conversation should have ended!")

    return {'slot_action': slot_action, 'slot_value_action': None}


class HackPolicy(BaseAgent):
  """ This agent requests and informs basic slots and then concludes. """
  def initialize_episode(self):
    self.request_slot_id = 0
    self.inform_slot_id = 0
    self.agent_turn_count = -1
    self.unknown_set = dialog_config.start_dia_acts["request"].copy()
    self.known_set = []
    self.complete = False

  def state_to_action(self, state):
    self.agent_turn_count += 2
    slot_action = {'dialogue_act': None, 'inform_slots': {}, 'request_slots': {},
                                          'turn_count': self.agent_turn_count}
    if state['user_action']['dialogue_act'] == 'thanks':
      # print("unknown_set", self.unknown_set)
      # print("compelte", self.complete)
      if len(self.unknown_set) > 0:
        chosen_slot = random.choice(self.unknown_set)
        slot_action['dialogue_act'] = 'request'
        slot_action['request_slots'] = {chosen_slot: "PLACEHOLDER"}
      elif self.complete:
        slot_action['dialogue_act'] = 'thanks'
      else:
        slot_action['dialogue_act'] = 'inform'
        slot_action['inform_slots']['taskcomplete'] = True
        self.complete = True

    else:
      self.remember_user_action(state["user_action"])
      # user made a request
      if len(state["user_action"]["request_slots"]) > 0:
        chosen_slot = list(state["user_action"]["request_slots"].keys())[0]
        slot_action['dialogue_act'] = "inform"
        slot_action['inform_slots'] = {chosen_slot: "PLACEHOLDER"}
      # user informed us of a constraint
      elif len(state["user_action"]["inform_slots"]) > 0:
        if len(self.unknown_set) > 0:
          chosen_slot = random.choice(self.unknown_set)
          slot_action['dialogue_act'] = "request"
          slot_action['request_slots'] = {chosen_slot: "PLACEHOLDER"}
        else:
          slot_action['dialogue_act'] = "inform"
          slot_action['inform_slots']['taskcomplete'] = True
          self.complete = True

    return {'slot_action': slot_action, 'slot_value_action': None}

  def remember_user_action(self, action):
    slots = {**action["request_slots"], **action["inform_slots"]}
    for slot in slots.keys():
      try:
        self.unknown_set.remove(slot)
        self.known_set.append(slot)
      except(ValueError): pass
