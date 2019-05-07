"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
import copy
import torch
import os, pdb, sys
import numpy as np
from objects.modules.dialog_state import DialogState
from utils.external import dialog_constants

class DialogManager:
  """ A dialog manager to mediate the interaction between an agent and a customer """

  def __init__(self, args, sub_module, users, ontology, movie_dictionary):
    self.model = sub_module
    self.debug = args.debug
    self.verbose = args.verbose
    self.task = args.task
    self.use_old_nlu = args.use_old_nlu

    if len(users) == 4:
      user_sim, world_sim, real_user, turk_user = users
      self.real_user = real_user
      self.turk_user = turk_user
    elif len(users) == 2:
      user_sim, world_sim = users
    self.user_sim = user_sim
    self.world_model = world_sim

    self.act_set = ontology.acts
    self.slot_set = ontology.slots
    self.state_tracker = DialogState(ontology, movie_dictionary)
    self.reward = 0
    self.episode_over = False

    self.save_dir = sub_module.model.save_dir
    self.use_world_model = False
    self.running_user = self.user_sim

    self.holder = {'examples': [], 'counter': 0}

  def initialize_episode(self, user_type):
    """ Refresh state for new dialog """
    self.reward = 0
    self.episode_over = False
    self.state_tracker.initialize_episode()
    self.run_mode = user_type
    self.use_world_model = False

    if user_type == 'rule':
      self.running_user = self.user_sim
    elif user_type == 'neural':
      self.running_user = self.world_model
      self.use_world_model = True
    elif user_type == 'command':
      self.running_user = self.real_user
    elif user_type == 'turk':
      self.running_user = self.turk_user

    self.running_user.initialize_episode()
    self.model.initialize_episode()

  def start_conversation(self, user_type):
    """ User takes the first turn and updates the dialog state """
    user_intent = self.running_user.take_first_turn()
    if user_type == 'rule':
      self.world_model.goal = self.user_sim.goal

    if self.task == 'end_to_end' and self.use_world_model:
      utterance = self.running_user.nlg_model.generate(user_intent, "usr")
      user_belief = self.model.belief_tracker.classify_intent(utterance)
      if self.use_old_nlu:
        user_intent = user_belief
      else:
        user_intent['belief'] = user_belief

    self.state_tracker.update_user_state(user_intent)
    self.print_function(user_intent, 'user')

  def next(self, record_agent_data=True, record_user_data=True):
    """ Initiates exchange between agent and user (agent first)
    a POMDP takes in the dialogue state with latent intent
      input - dialogue state consisting of:
        1) current user intent --> act(slot-relation-value) + confidence score
        2) previous agent action
        3) complete semantic frame
        4) turn count
        5) knowledge base query result count
      output - next agent action

      > model_action keys: slot_action, action_id
        slot action: nl, dialogue_act, inform_slots, request_slots, turn_count
      > sys_action keys: speaker, dialogue_act, inform_slots, request_slots, turn_count
      > user_intent keys: dialogue_act, inform_slots, request_slots

      {'dialogue_act': 'inform',
      'inform_slots': {'moviename': 'risen'},
      'request_slots': {}}

    """
    #   CURRENT STATE (s)
    self.agent_state = self.state_tracker.get_state('agent')
    #   ACTION (a)
    model_action = self.model.state_to_action(self.agent_state)

    #   Register agent action with the state_tracker
    self.state_tracker.update_agent_state(model_action)
    self.user_state = self.state_tracker.get_state('user')
    self.model.action_to_nl(model_action)  # add NL to Agent Dia_Act
    self.print_function(model_action['slot_action'], 'agent')
    self.sys_action = self.state_tracker.history_dictionaries[-1]

    """
    Have the user take their turn, note that the environment takes
      the form of a user simulator.  Thus, this is how we get the reward.
    Note also, that an environment should return a next agent state.
      In this case, the user intent *is* the next agent state.
    """

    # REWARD (r)
    if self.use_world_model:
      user_output = self.running_user.next(self.user_state, model_action)
      user_intent, self.episode_over, self.reward = user_output
    else:
      user_output = self.running_user.next(self.sys_action)
      user_intent, self.episode_over, dialog_status = user_output
      self.reward = self.model.reward_function(dialog_status)

    #   Register user action with the state_tracker
    if self.episode_over != True:
      # minimal change of adding belief as an extra portion
      if self.task == 'end_to_end' and self.use_world_model:
        datapoint = []
        datapoint.append(user_intent.copy())
        utterance = self.running_user.nlg_model.generate(user_intent, 'usr')
        datapoint.append(utterance)
        # utterance is a string, not a list of tokens
        user_belief = self.model.belief_tracker.classify_intent(utterance, model_action)
        datapoint.append(user_belief)
        if self.use_old_nlu:
          user_intent = user_belief
        else:
          user_intent['belief'] = user_belief

          self.holder['examples'].append(datapoint)
          self.holder['counter'] += 1
          if self.holder['counter'] > 100:
            for utt, pred, actual in self.holder['examples']:
              print(f"Original: {actual}")
              print(utt)
              print(f"Belief: {pred}")
              print("-----------------")
            sys.exit()

      self.state_tracker.update_user_state(user_intent)
      self.print_function(user_intent, 'user')

    # NEXT STATE (s')
    next_agent_state = self.state_tracker.get_state('agent')
    next_user_state = self.state_tracker.get_state('user')

    # Record data in experience replay pools
    if record_agent_data:
      self.model.use_world_model = self.use_world_model
      self.model.store_experience(self.agent_state, model_action['action_id'],
        self.reward, next_agent_state, self.episode_over)
    if record_user_data and not self.use_world_model:
      self.world_model.store_experience(self.user_state, model_action['action_id'],
        self.reward, next_agent_state, self.episode_over, user_intent)

    return self.episode_over, self.reward

  def respond_to_turker(self, raw_user_input):
    # intent classification
    if self.running_user.agent_input_mode == 'natural_language':
      user_input = self.model.belief_tracker.classify_intent(raw_user_input)
    elif self.running_user.agent_input_mode == 'dialogue_act':
      user_input = self.parse_raw_input(raw_user_input)
    print(json.dumps(user_input, indent=2))
    self.state_tracker.update_user_state(user_input)
    # policy management
    self.agent_state = self.state_tracker.get_state('agent')
    model_action = self.model.state_to_action(self.agent_state)
    self.state_tracker.update_agent_state(model_action)
    # text generation
    self.model.action_to_nl(model_action)  # add NL to Agent Dia_Act
    agent_response = model_action['slot_action']['nl']
    return agent_response

  def parse_raw_input(self, raw):
    parsed = {'inform_slots':{}, 'request_slots':{}}
    cleaned = raw.strip(' ').strip('\n').strip('\r')
    intents = cleaned.lower().split(',')
    for intent in intents:
      idx = intent.find('(')
      act = intent[0:idx]
      if re.search(r'thanks?', act):
        self.finish_episode = True
      else:
        slot, value = intent[idx+1:-1].split("=") # -1 is to skip the closing ')'
        parsed["{}_slots".format(act)][slot] = value

      parsed["dialogue_act"] = act
      parsed["nl"] = cleaned

    parsed['turn_count'] = 2
    return parsed

  def save_performance_records(self, monitor):
    episode = monitor.num_episodes
    filepath = os.path.join(self.save_dir, f'results_{episode}.json')
    records = {'turns': monitor.turns, 'avg_turn': monitor.avg_turn,
      'rewards': monitor.rewards, 'avg_reward': monitor.avg_reward,
      'successes': monitor.simulation_successes, 'episode': episode,
      'avg_sim_success': np.average(monitor.simulation_successes),
      'avg_true_success': monitor.success_rate }
    json.dump(records, open(filepath, "w"))
    print('Saved performance records at {}'.format(filepath))

  def save_config(self, args, directory):
    # just pass the data onto the policy module for the real heavy lifting
    self.model.save_config(args, directory)

  def print_function(self, action_dict, kind):
    if self.run_mode == "command" and kind == "agent":
      print ("{}) {}: {}".format(action_dict['turn_count'], kind, action_dict['nl']))
    # kind should be "agent" or "user"
    if not self.debug: return
    if self.verbose:
      for k, v in action_dict.items(): print(kind, k, v)
    else:
      print ("{}) {}: {}".format(action_dict['turn_count'], kind, action_dict['nl']))
    if dialog_constants.auto_suggest and kind == "agent":
      output = self.state_tracker.make_suggestion(action_dict['request_slots'])
      print(f'(Suggested Values: {output})')
