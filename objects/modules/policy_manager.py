import numpy as np
import os, pdb, sys  # set_trace
import logging
import copy
import random

from objects.blocks.base import BasePolicy
from objects.modules.user import UserSimulator, RealUser
from objects.modules.dialogue_state import StateTracker
from torch import nn

class BasePolicyManager(object):
  def __init__(self, args):
    self.data = data
    self.verbose = args.verbose
    self.experiences = []  # tuples of state, action, reward, next state
    self.learning_method = "reinforce" # or "rulebased" or "supervised"

    self.agent = model
    self.state_tracker = StateTracker(act_set, slot_set, value_set)
    self.user_action = None

    self.max_turns = args.max_turns
    self.num_episodes = args.num_iters
    self.batch_size = args.batch_size

    if args.user == "simulate":
      self.user = UserSimulator(act_set, slot_set, value_set, goal_set, args)
      # self.user = MovieUser(act_set, slot_set, value_set, goal_set, args)
      # self.user = RestaurantUser(act_set, slot_set, value_set, goal_set, args)
      # self.user = TaxiUser(act_set, slot_set, value_set, goal_set, args)
    else:
      self.user = RealUser(act_set, slot_set, value_set, goal_set, args)

  def initialize_episode(self):
    self.reward = 0
    self.episode_over = False

    self.state_tracker.initialize_episode()
    self.user_action = self.user.initialize_episode()
    # self.agent.initialize_episode()
    self.state_tracker.update(user_action = self.user_action)

    if args.verbose:
      print("New episode, user goal:")
      print(json.dumps(self.user.goal, indent=2))
      self.print_function()

  def print_function(self):
    print("go back to d3q to find this method for debugging")

  def next_turn(self, record_training_data=True):
    """ Initiates exchange between agent and user (agent first) """

    #   CALL AGENT TO TAKE HER TURN
    self.state = self.state_tracker.get_state_for_agent()
    self.agent_action = self.agent.state_to_action(self.state)
    #   Register AGENT action with the state_tracker
    self.state_tracker.update(agent_action=self.agent_action)
    self.agent.add_nl_to_action(self.agent_action) # add NL to BasePolicy Dia_Act
    if self.verbose:
      self.print_function(agent_action = self.agent_action['act_slot_response'])

    #   CALL USER TO TAKE HER TURN
    self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
    self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
    self.reward = self.reward_function(dialog_status)
    #   Update state tracker with latest user action
    if self.episode_over != True:
      self.state_tracker.update(user_action=self.user_action)
      # self.print_function(user_action=self.user_action)

    #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
    if record_training_data:
      self.agent.store_experience(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over)

    return (self.episode_over, self.reward)

  def reward_function(self, dialog_status, penalty=True):
    """ Reward Function 1: a reward function based on the dialog_status
    if penalty is True, then there is also a negative reward for each turn
    """
    if dialog_status == dialog_config.FAILED_DIALOG:
      reward = -self.user.max_turn if penalty else 0 # 10
    elif dialog_status == dialog_config.SUCCESS_DIALOG:
      reward = 2 * self.user.max_turn #20
    else:  # for per turn
      reward = -1 if penalty else 0
    return reward

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



class InformPolicy(BasePolicy):
  """ A simple agent to test the system. This agent should simply inform
  all the slots and then issue: taskcomplete. """
  def state_to_action(self, state):
    self.state['turn'] += 2
    if self.current_slot_id < len(self.slot_set.keys()):
      slot = self.slot_set.keys()[self.current_slot_id]
      self.current_slot_id += 1
      action_slot_only = {'diaact': "inform",
                          'inform_slots': {slot: "PLACEHOLDER"},
                          'request_slots':  {},
                          'turn': self.state['turn'] }
    else:
      action_slot_only = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn': self.state['turn'] }

    return {'action_slot_only': action_slot_only, 'agent_action': None}


class RequestPolicy(BasePolicy):
  """ A simple agent to test the system. This agent should simply
        request all the slots and then issue: thanks(). """
  def state_to_action(self, state):
    self.state['turn'] += 2
    if self.current_slot_id < len(dialog_config.sys_request_slots):
      slot = dialog_config.sys_request_slots[self.current_slot_id]
      self.current_slot_id += 1

      action_slot_only = {'diaact': "request",
                          'inform_slots': {},
                          'request_slots':  {slot: "PLACEHOLDER"},
                          'turn': self.state['turn'] }
    else:
      action_slot_only = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn': self.state['turn'] }

    return {'action_slot_only': action_slot_only, 'agent_action': None}


class RandomPolicy(BasePolicy):
  """ A simple agent to test the interface which chooses actions randomly. """
  def state_to_action(self, state):
    self.state['turn'] += 2
    random_action = random.choice(dialog_config.feasible_actions)
    action_slot_only = copy.deepcopy(random_action)
    action_slot_only['turn'] = self.state['turn']

    return {'action_slot_only': action_slot_only, 'agent_action': None}


class EchoPolicy(BasePolicy):
  """ A simple agent that informs all requested slots,
  then issues inform(taskcomplete) when the user stops making requests. """
  def state_to_action(self, state):
    user_action = state['user_action']
    self.state['turn'] += 2
    action_slot_only = {'diaact': 'thanks',
                        'inform_slots': {},
                        'request_slots':  {},
                        'turn': self.state['turn'] }
    # find out if the user is requesting anything.  if so, inform it
    if user_action['diaact'] == 'request':
      action_slot_only['diaact'] = "inform"
      requested_slot = user_action['request_slots'].keys()[0]
      action_slot_only['inform_slots'][requested_slot] = "PLACEHOLDER"

    return {'action_slot_only': action_slot_only, 'agent_action': None}


class RequestBasicsPolicy(BasePolicy):
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
                  'turn': -1  }
    self.current_slot_id = 0
    self.phase = 0

  def state_to_action(self, state):
    self.state['turn'] += 2
    if self.current_slot_id < len(self.request_set):
      slot = self.request_set[self.current_slot_id]
      self.current_slot_id += 1

      action_slot_only = {'diaact': "request",
                          'inform_slots': {},
                          'request_slots': {slot: "UNK"},
                          'turn': self.state['turn'] }
    elif self.phase == 0:
      action_slot_only = {'diaact': "inform",
                          'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                          'request_slots': {},
                          'turn':self.state['turn'] }
      self.phase += 1
    elif self.phase == 1:
      action_slot_only = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn': self.state['turn'] }
    else:
      raise Exception("IS NOT POSSIBLE! (agent called in unexpected way)")

    return {'action_slot_only': action_slot_only, 'agent_action': None}


class BasicsPolicy(BasePolicy):
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
                  'turn': -1  }
    self.current_request_slot_id = 0
    self.current_inform_slot_id = 0
    self.phase = 0

  def state_to_action(self, state):
    self.state['turn'] += 2
    if self.current_request_slot_id < len(self.request_set):
      slot = self.request_set[self.current_request_slot_id]
      self.current_request_slot_id += 1
      action_slot_only = {'diaact': "request",
                          'inform_slots': {},
                          'request_slots': {slot: "PLACEHOLDER"},
                          'turn': self.state['turn'] }
    elif self.current_inform_slot_id < len(self.inform_set):
      slot = self.inform_set[self.current_inform_slot_id]
      self.current_inform_slot_id += 1
      action_slot_only = {'diaact': "inform",
                          'inform_slots': {slot: "PLACEHOLDER"},
                          'request_slots': {},
                          'turn': self.state['turn'] }
    elif self.phase == 0:
      action_slot_only = {'diaact': "inform",
                          'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                          'request_slots': {},
                          'turn':self.state['turn'] }
      self.phase += 1
    elif self.phase == 1:
      action_slot_only = {'diaact': "thanks",
                          'inform_slots': {},
                          'request_slots': {},
                          'turn': self.state['turn'] }
    else:
      raise Exception("IS NOT POSSIBLE! (agent called in unexpected way)")

    return {'action_slot_only': action_slot_only, 'agent_action': None}


class RulePolicyManager(BasePolicyManager):
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
    pass


class NeuralPolicyManager(BasePolicyManager, nn.Module):
  def __init__(self, args):
    super().__init__(args)
    self.hidden_dim = args.hidden_dim
    self.gamma = args.discount_rate
    self.warm_start = args.warm_start
    self.experience_replay_pool_size = args.pool_size
