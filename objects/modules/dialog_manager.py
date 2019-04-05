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
from utils.external import dialog_config

class DialogManager:
  """ A dialog manager to mediate the interaction between an agent and a customer """

  def __init__(self, args, sub_module, user_sim, world_model, real_user,
          turk_user, ontology, movie_dictionary):
    self.model = sub_module
    self.debug = args.debug
    self.verbose = args.verbose

    self.user_sim = user_sim
    self.world_model = world_model
    self.real_user = real_user
    self.turk_user = turk_user

    self.act_set = ontology.acts
    self.slot_set = ontology.slots
    self.state_tracker = DialogState(ontology, movie_dictionary)
    self.user_action = None
    self.reward = 0
    self.episode_over = False

    self.save_dir = sub_module.model.save_dir
    self.use_world_model = False
    self.running_user = self.user_sim
    # self.run_mode = dialog_config.run_mode

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
    user_action = self.running_user.take_first_turn()
    if user_type == 'rule':
      self.world_model.goal = self.user_sim.goal
    self.state_tracker.update(user_action=user_action)
    self.print_function(user_action, 'user')

  def next(self, record_agent_data=True, record_user_data=True):
    """ Initiates exchange between agent and user (agent first)
    a POMDP takes in the dialogue state with latent intent
      input - dialogue state consisting of:
        1) current user intent --> act(slot-relation-value) + confidence score
        2) previous agent action
        3) knowledge base query results
        4) turn count
        5) complete semantic frame
      output - next agent action
    """
    #   CALL AGENT TO TAKE HER TURN
    self.agent_state = self.state_tracker.get_state_for_agent()
    model_action = self.model.state_to_action(self.agent_state)
    #   Register AGENT action with the state_tracker
    self.state_tracker.update(agent_action=model_action)
    self.user_state = self.state_tracker.get_state_for_user()

    self.model.action_to_nl(model_action)  # add NL to Agent Dia_Act
    self.print_function(model_action['slot_action'], 'agent')

    #   CALL USER TO TAKE HER TURN
    self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
    if self.use_world_model:
      self.user_action, self.episode_over, self.reward = self.running_user.next(
              self.user_state, model_action)
    else:
      self.user_action, self.episode_over, dialog_status = self.running_user.next(self.sys_action)
      self.reward = self.model.reward_function(dialog_status)
    """ Uncomment to use the belief tracker
    user_utterance = self.user_action['nl']
    pred_action = self.model.belief_tracker.classify_intent(user_utterance)
    self.user_action = pred_action """

    #   Update state tracker with latest user action
    if self.episode_over != True:
      self.state_tracker.update(user_action=self.user_action)
      self.print_function(self.user_action, 'user')
    next_agent_state = self.state_tracker.get_state_for_agent()

    #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over, s_t_u, user_world_model)
    if record_agent_data:
      self.model.use_world_model = self.use_world_model
      self.model.store_experience(self.agent_state, model_action['action_id'],
        self.reward, next_agent_state, self.episode_over)

    #  Inform world model of the outcome for this timestep
    # (s_t, a_t, s_{t+1}, r, t, ua_t)
    if record_user_data and not self.use_world_model:
      self.world_model.store_experience(self.user_state,
        model_action['action_id'], next_agent_state, self.reward,
        self.episode_over, self.user_action)

    return (self.episode_over, self.reward)

  def respond_to_turker(self, raw_user_input):
    # intent classification
    if self.running_user.agent_input_mode == 'natural_language':
      user_input = self.model.belief_tracker.classify_intent(raw_user_input)
    elif self.running_user.agent_input_mode == 'dialogue_act':
      user_input = self.parse_raw_input(raw_user_input)
    print(json.dumps(user_input, indent=2))
    self.state_tracker.update(user_action=user_input)
    # policy management
    self.agent_state = self.state_tracker.get_state_for_agent()
    model_action = self.model.state_to_action(self.agent_state)
    self.state_tracker.update(agent_action=model_action)
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
    if dialog_config.auto_suggest and kind == "agent":
      output = self.state_tracker.make_suggestion(action_dict['request_slots'])
      print(f'(Suggested Values: {output})')



  # def print_function(self, agent_action=None, user_action=None):
  #   if agent_action:
  #     if self.run_mode == 0:
  #       if self.model.__class__.__name__ != 'AgentCmd':
  #         print("Turn %d sys: %s" % (agent_action['turn_count'], agent_action['nl']))
  #     elif self.run_mode == 1:
  #       if self.model.__class__.__name__ != 'AgentCmd':
  #         print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (
  #           agent_action['turn_count'], agent_action['dialogue_act'], agent_action['inform_slots'],
  #           agent_action['request_slots']))
  #     elif self.run_mode == 2:  # debug mode
  #       print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (
  #         agent_action['turn_count'], agent_action['dialogue_act'], agent_action['inform_slots'],
  #         agent_action['request_slots']))
  #       print("Turn %d sys: %s" % (agent_action['turn_count'], agent_action['nl']))

  #     if dialog_config.auto_suggest == 1:
  #       print(
  #         '(Suggested Values: %s)' % (
  #         self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))
  #   elif user_action:
  #     if self.run_mode == 0:
  #       print("Turn %d usr: %s" % (user_action['turn_count'], user_action['nl']))
  #     elif self.run_mode == 1:
  #       print("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (
  #         user_action['turn_count'], user_action['dialogue_act'], user_action['inform_slots'],
  #         user_action['request_slots']))
  #     elif self.run_mode == 2:  # debug mode, show both
  #       print("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (
  #         user_action['turn_count'], user_action['dialogue_act'], user_action['inform_slots'],
  #         user_action['request_slots']))
  #       print("Turn %d usr: %s" % (user_action['turn_count'], user_action['nl']))

  #     if self.model.__class__.__name__ == 'AgentCmd':  # command line agent
  #       user_request_slots = user_action['request_slots']
  #       if 'ticket' in user_request_slots.keys(): del user_request_slots['ticket']
  #       if len(user_request_slots) > 0:
  #         possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
  #         for slot in possible_values.keys():
  #           if len(possible_values[slot]) > 0:
  #             print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
  #           elif len(possible_values[slot]) == 0:
  #             print('(Suggested Values: there is no available %s)' % (slot))
  #       else:
  #         kb_results = self.state_tracker.get_current_kb_results()
  #         print('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))
