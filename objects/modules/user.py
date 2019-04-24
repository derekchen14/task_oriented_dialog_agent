import pdb, sys
import random, copy
import numpy as np
import re

from utils.external import dialog_constants
from collections import namedtuple, deque
from objects.models.user_model import SimulatorModel

import torch
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'agent_action', 'next_state', 'reward', 'term', 'user_action'))

class BaseUser(object):
  def __init__(self, args, ontology, goal_set=None):
    self.task = args.task
    self.max_turn = args.max_turn
    self.num_episodes = args.epochs
    self.debug = args.debug
    self.verbose = args.verbose
    self.learning_phase = 'all'  # vs. train and test

    self.act_set = {act: i for i, act in enumerate(ontology.acts)}
    self.slot_set = {slot: j for j, slot in enumerate(ontology.slots)}
    self.relation_set = ontology.relations
    self.value_set = ontology.values
    self.goal_set = goal_set

    self.slot_err_probability = 0.0
    self.slot_err_mode = 0
    self.intent_err_probability = 0.0
    self.simulator_act_level = 0
    self.simulator_run_mode = dialog_constants.run_mode
    self.agent_input_mode =  "natural_language" # "dialogue_act"

    self.nlu_model = None
    self.nlg_model = None

  def _sample_goal(self):
    return random.choice(self.goal_set[self.learning_phase])

  def initialize_episode(self):
    """ Initialize a new episode (dialog)
    state['history_slots']: keeps all the informed_slots
    state['rest_slots']: keep all the slots (which is still in the stack yet)
    """
    self.state = {}
    self.state['history_slots'] = {}
    self.state['inform_slots'] = {}
    self.state['request_slots'] = {}
    self.state['rest_slots'] = []
    self.state['turn_count'] = 0

    self.episode_over = False
    self.dialog_status = dialog_constants.NO_OUTCOME_YET
    self.constraint_check = dialog_constants.CONSTRAINT_CHECK_FAILURE

    self.goal = self._sample_goal()
    self.goal['request_slots']['ticket'] = 'UNK'
    if self.verbose and self.debug:
      print("new episode being initialized, generating goal ...")
      print(self.goal)

  def take_first_turn(self):
    self.user_action = {
      'dialogue_act': 'UNK',
      'inform_slots': {},
      'request_slots': {},
      'turn_count': 0
    }
    return self.user_action

  def next(self, agent_action):
    raise(NotImplementedError, "User cannot take next step")

  def add_nl_to_action(self, user_action):
    """ Add NL to User Dia_Act """

    user_nlg_sentence = self.nlg_model.generate(user_action, 'usr')
    user_action['nl'] = user_nlg_sentence

    if self.simulator_act_level == 1:
      user_nlu_res = self.nlu_model.generate_dia_act(user_action['nl'])  # NLU
      if user_nlu_res != None:
        # user_nlu_res['dialogue_act'] = user_action['dialogue_act'] # or not?
        user_action.update(user_nlu_res)

class CommandLineUser(BaseUser):
  def __init__(self, args, ontology, goal_set):
    super().__init__(args, ontology, goal_set)
    self.learning_phase = "train"
    self.agent_input_mode =  "natural_language" # "dialogue_act"

  def initialize_episode(self):
    self.state = {
      'history_slots': {},  # slots that have already been processed (post)
      'inform_slots': {},
      'request_slots': {},
      'remaining_slots': [],   # slots that have yet to be processed (pre)
      'turn_count': 0
    }

    self.goal = self._sample_goal()
    self.turn_count = 0
    self.finish_episode = False

  def take_first_turn(self):
    super().take_first_turn()
    if self.agent_input_mode == "natural_language":
      print("Your input will be raw text so the system expects the dialogue \
        model to include a NLU module for intent classification")
    elif self.agent_input_mode == "dialogue_act":
      print("The system expects a properly written user intent of the form " +
        "act(slot=value) such as inform(city=Los_Angeles).  Multiple intents " +
        "can be included by joining them with a comma. Spaces will be removed.")
    print("Your goal: {}".format(self.goal['request_slots']))
    print("Your constraints: {}".format(self.goal['inform_slots']))

    command_line_input = input("{}) user: ".format(self.turn_count))
    if self.agent_input_mode == "natural_language":
      self.user_action['nl'] = command_line_input
    elif self.agent_input_mode == "dialogue_act":
      self.parse_intent(command_line_input)
    self.user_action['turn_count'] = self.turn_count

    return self.user_action

  def next(self, agent_action):
    # Generate an action by getting input interactively from the command line
    self.turn_count += 2
    if 'taskcomplete' in agent_action['inform_slots'].keys():
      self.finish_episode = True

    if self.turn_count > self.max_turn or self.finish_episode:
      episode_over = True
      dialog_status = self.determine_success(agent_action)
    else:
      episode_over = False
      dialog_status = dialog_constants.NO_OUTCOME_YET

    command_line_input = input("{}) user: ".format(self.turn_count))
    if self.agent_input_mode == "natural_language":
      self.user_action['nl'] = command_line_input
    elif self.agent_input_mode == "dialogue_act":
      self.parse_intent(command_line_input)
    self.user_action['turn_count'] = self.turn_count

    print("From next", self.user_action['nl'])
    return self.user_action, episode_over, dialog_status

  def determine_success(self, agent_action):
    if self.verbose:
      print(agent_action)

    for slot in self.goal['inform_slots'].keys():
      # missing_constraint
      if slot not in agent_action['inform_slots'].keys():
        print(f"Agent was missing the {slot} constraint!")
        return dialog_constants.FAILED_DIALOG
      # incorrect_value
      if self.goal['inform_slots'][slot].lower() != agent_action['inform_slots'][slot].lower():
        print(f"Agent had the wrong value for {slot}!")
        return dialog_constants.FAILED_DIALOG
    # reaching the end means none of the slots had any errors !
    print("The agent was successful in matching all slots!")
    return dialog_constants.SUCCESS_DIALOG

  def parse_intent(self, command_line_input):
    """ Parse input from command line into dialogue act form """
    cleaned = command_line_input.strip(' ').strip('\n').strip('\r')
    intents = cleaned.lower().split(',')
    for intent in intents:
      idx = intent.find('(')
      act = intent[0:idx]
      if re.search(r'thanks?', act):
        self.finish_episode = True
      else:
        slot, value = intent[idx+1:-1].split("=") # -1 is to skip the closing ')'
        self.user_action["{}_slots".format(act)][slot] = value
        self.error_checking(idx, act, slot, value)

      self.user_action["dialogue_act"] = act
      self.user_action["nl"] = cleaned

  def error_checking(self, idx, act, slot, value):
    if idx < 0:
      raise(ValueError("input is not properly formatted as a user intent"))
    if act not in self.act_set:
      raise(ValueError("{} is not part of allowable dialogue act set".format(act)))
    if slot not in self.slot_set:
      raise(ValueError("{} is not part of allowable slot set".format(slot)))
    if slot != 'ticket' and value not in self.value_set[slot]:
      raise(ValueError("{} is not part of the available value set".format(value)))

  def display_outcome(self):
    print("Your goal:")
    print(self.goal)
    print("Predicted frame:")
    print(self.state['history_slots'])

class MechanicalTurkUser(BaseUser):

  def next(self, agent_action):
    # Generate an action by getting input interactively from the command line
    self.turn_count += 2
    if 'taskcomplete' in agent_action['inform_slots'].keys():
      self.finish_episode = True

    if self.turn_count > self.max_turn or self.finish_episode:
      episode_over = True
      dialog_status = self.determine_success(agent_action)
    else:
      episode_over = False
      dialog_status = dialog_constants.NO_OUTCOME_YET

    command_line_input = input("{}) user: ".format(self.turn_count))
    if self.agent_input_mode == "raw_text":
      self.user_action['nl'] = command_line_input
    elif self.agent_input_mode == "dialogue_act":
      self.parse_intent(command_line_input)
    self.user_action['turn_count'] = self.turn_count

    return self.user_action, episode_over, dialog_status

  def determine_success(self, agent_action):
    if self.verbose:
      print(agent_action)

    for slot in self.goal['inform_slots'].keys():
      # missing_constraint
      if slot not in agent_action['inform_slots'].keys():
        print(f"Agent was missing the {slot} constraint!")
        return dialog_constants.FAILED_DIALOG
      # incorrect_value
      if self.goal['inform_slots'][slot].lower() != agent_action['inform_slots'][slot].lower():
        print(f"Agent had the wrong value for {slot}!")
        return dialog_constants.FAILED_DIALOG
    # reaching the end means none of the slots had any errors !
    print("The agent was successful in matching all slots!")
    return dialog_constants.SUCCESS_DIALOG

  def parse_intent(self, command_line_input):
    """ Parse input from command line into dialogue act form """
    cleaned = command_line_input.strip(' ').strip('\n').strip('\r')
    intents = cleaned.lower().split(',')
    for intent in intents:
      idx = intent.find('(')
      act = intent[0:idx]
      if re.search(r'thanks?', act):
        self.finish_episode = True
      else:
        slot, value = intent[idx+1:-1].split("=") # -1 is to skip the closing ')'
        self.user_action["{}_slots".format(act)][slot] = value
        self.error_checking(idx, act, slot, value)

      self.user_action["dialogue_act"] = act
      self.user_action["nl"] = cleaned

class RuleSimulator(BaseUser):
  """ A rule-based user simulator for testing dialog policy """

  def take_first_turn(self):
    """ randomly sample a start action based on user goal """

    starting_acts = dialog_constants.starting_dialogue_acts
    if self.task == 'end_to_end': starting_acts.append('greeting')
    self.state['dialogue_act'] = random.choice(starting_acts)

    # "sample" informed slots
    if len(self.goal['inform_slots']) > 0:
      known_slot = random.choice(list(self.goal['inform_slots'].keys()))
      self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

      if 'moviename' in self.goal['inform_slots'].keys(): # 'moviename' must appear in the first user turn
        self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']

      for slot in self.goal['inform_slots'].keys():
        if known_slot == slot or slot == 'moviename': continue
        self.state['rest_slots'].append(slot)

    self.state['rest_slots'].extend(self.goal['request_slots'].keys())

    # "sample" a requested slot
    request_slot_set = list(self.goal['request_slots'].keys())
    request_slot_set.remove('ticket')
    if len(request_slot_set) > 0:
      request_slot = random.choice(request_slot_set)
    else:
      request_slot = 'ticket'
    self.state['request_slots'][request_slot] = 'UNK'

    if len(self.state['request_slots']) == 0:
      self.state['dialogue_act'] = 'inform'

    if (self.state['dialogue_act'] in ['thanks','closing']): self.episode_over = True #episode_over = True
    else: self.episode_over = False #episode_over = False

    sample_action = {}
    sample_action['dialogue_act'] = self.state['dialogue_act']
    sample_action['inform_slots'] = self.state['inform_slots']
    sample_action['request_slots'] = self.state['request_slots']
    sample_action['turn_count'] = self.state['turn_count']

    self.add_nl_to_action(sample_action)
    return sample_action

  def corrupt(self, user_action):
    """ Randomly corrupt an action with error probs (slot_err_probability and slot_err_mode) on Slot and Intent (intent_err_probability). """

    for slot in user_action['inform_slots'].keys():
      slot_err_prob_sample = random.random()
      if slot_err_prob_sample < self.slot_err_probability: # add noise for slot level
        if self.slot_err_mode == 0: # replace the slot_value only
          if slot in self.value_set.keys(): user_action['inform_slots'][slot] = random.choice(self.value_set[slot])
        elif self.slot_err_mode == 1: # combined
          slot_err_random = random.random()
          if slot_err_random <= 0.33:
            if slot in self.value_set.keys(): user_action['inform_slots'][slot] = random.choice(self.value_set[slot])
          elif slot_err_random > 0.33 and slot_err_random <= 0.66:
            del user_action['inform_slots'][slot]
            random_slot = random.choice(list(self.value_set.keys()))
            user_action[random_slot] = random.choice(self.value_set[random_slot])
          else:
            del user_action['inform_slots'][slot]
        elif self.slot_err_mode == 2: #replace slot and its values
          del user_action['inform_slots'][slot]
          random_slot = random.choice(list(self.value_set.keys()))
          user_action[random_slot] = random.choice(self.value_set[random_slot])
        elif self.slot_err_mode == 3: # delete the slot
          del user_action['inform_slots'][slot]

    intent_err_sample = random.random()
    if intent_err_sample < self.intent_err_probability: # add noise for intent level
      user_action['dialogue_act'] = random.choice(list(self.act_set.keys()))

  def debug_fake_goal(self):
    """ Debug function: build a fake goal mannually (Can be moved in future) """

    self.goal['inform_slots'].clear()
    #self.goal['inform_slots']['city'] = 'seattle'
    self.goal['inform_slots']['numberofpeople'] = '2'
    #self.goal['inform_slots']['theater'] = 'amc pacific place 11 theater'
    #self.goal['inform_slots']['starttime'] = '10:00 pm'
    #self.goal['inform_slots']['date'] = 'tomorrow'
    self.goal['inform_slots']['moviename'] = 'zoology'
    self.goal['inform_slots']['distanceconstraints'] = 'close to 95833'
    self.goal['request_slots'].clear()
    self.goal['request_slots']['ticket'] = 'UNK'
    self.goal['request_slots']['theater'] = 'UNK'
    self.goal['request_slots']['starttime'] = 'UNK'
    self.goal['request_slots']['date'] = 'UNK'

  def next(self, agent_action):
    """ Generate next User Action based on last System Action """

    self.state['turn_count'] += 2
    self.episode_over = False
    self.dialog_status = dialog_constants.NO_OUTCOME_YET

    sys_act = agent_action['dialogue_act']


    if (self.max_turn > 0 and self.state['turn_count'] > self.max_turn):
      self.dialog_status = dialog_constants.FAILED_DIALOG
      self.episode_over = True
      self.state['request_slots'].clear()
      self.state['inform_slots'].clear()
      self.state['dialogue_act'] = "closing"
    else:
      self.state['history_slots'].update(self.state['inform_slots'])
      self.state['inform_slots'].clear()

      if sys_act == "inform":
        self.response_inform(agent_action)
      elif sys_act == "multiple_choice":
        self.response_multiple_choice(agent_action)
      elif sys_act == "request":
        self.response_request(agent_action)
      elif sys_act == "thanks":
        self.response_thanks(agent_action)
      elif sys_act == "confirm_answer":
        self.response_confirm_answer(agent_action)
      elif sys_act == "closing":
        self.episode_over = True
        self.state['dialogue_act'] = "thanks"
        self.state['request_slots'].clear()

    if self.state['dialogue_act'] == "thanks":
      self.state['request_slots'].clear()
      self.state['inform_slots'].clear()

    self.corrupt(self.state)

    response_action = {}
    response_action['dialogue_act'] = self.state['dialogue_act']
    response_action['inform_slots'] = self.state['inform_slots']
    response_action['request_slots'] = self.state['request_slots']
    response_action['turn_count'] = self.state['turn_count']
    response_action['nl'] = ""
    # add NL to dia_act
    self.add_nl_to_action(response_action)
    return response_action, self.episode_over, self.dialog_status

  def response_confirm_answer(self, agent_action):
    """ Response for Confirm_Answer (System Action) """

    if len(self.state['rest_slots']) > 0:
      request_slot = random.choice(self.state['rest_slots'])

      if request_slot in self.goal['request_slots'].keys():
        self.state['request_slots'].clear()
        self.state['dialogue_act'] = "request"
        self.state['request_slots'][request_slot] = "UNK"
      elif request_slot in self.goal['inform_slots'].keys():
        self.state['dialogue_act'] = "inform"
        self.state['inform_slots'][request_slot] = self.goal['inform_slots'][request_slot]
        self.state['request_slots'].clear()
        if request_slot in self.state['rest_slots']:
          self.state['rest_slots'].remove(request_slot)
    else:
      self.state['dialogue_act'] = "thanks"
      self.state['request_slots'].clear()

  def response_thanks(self, agent_action):
    """ Response for Thanks (System Action) """
    self.episode_over = True
    request_slot_set = list(self.state['request_slots'].keys()).copy()
    rest_slot_set = self.state['rest_slots'].copy()

    if 'ticket' in request_slot_set:
      request_slot_set.remove('ticket')
    if 'ticket' in rest_slot_set:
      rest_slot_set.remove('ticket')

    # by default we are successful
    self.dialog_status = dialog_constants.SUCCESS_DIALOG
    # there are still remaining user questions to answer
    if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
      self.dialog_status = dialog_constants.FAILED_DIALOG

    # for any given requirement in the goal
    for info_slot in self.state['history_slots'].keys():
      # one of the constraints was not discussed
      if self.state['history_slots'][info_slot] == dialog_constants.NO_VALUE_MATCH:
        self.dialog_status = dialog_constants.FAILED_DIALOG
      # frame predicted the wrong value for some constraint
      if info_slot in self.goal['inform_slots'].keys():
        if self.state['history_slots'][info_slot] != self.goal['inform_slots'][info_slot]:
          self.dialog_status = dialog_constants.FAILED_DIALOG

    # the final ticket was not offered to the user
    if 'ticket' in agent_action['inform_slots'].keys():
      if agent_action['inform_slots']['ticket'] == dialog_constants.NO_VALUE_MATCH:
        self.dialog_status = dialog_constants.FAILED_DIALOG
    # there was some prior constraint check that failed
    if self.constraint_check == dialog_constants.CONSTRAINT_CHECK_FAILURE:
      self.dialog_status = dialog_constants.FAILED_DIALOG

  def response_request(self, agent_action):
    """ Response for Request (System Action) """

    if len(agent_action['request_slots'].keys()) > 0:
      slot = list(agent_action['request_slots'].keys())[0] # only one slot
      if slot in self.goal['inform_slots'].keys(): # request slot in user's constraints  #and slot not in self.state['request_slots'].keys():
        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
        self.state['dialogue_act'] = "inform"
        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
        if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
        self.state['request_slots'].clear()
      elif slot in self.goal['request_slots'].keys() and slot not in self.state['rest_slots'] and slot in self.state['history_slots'].keys(): # the requested slot has been answered
        self.state['inform_slots'][slot] = self.state['history_slots'][slot]
        self.state['request_slots'].clear()
        self.state['dialogue_act'] = "inform"
      elif slot in self.goal['request_slots'].keys() and slot in self.state['rest_slots']: # request slot in user's goal's request slots, and not answered yet
        self.state['request_slots'].clear() # changed on Dec 08 for unique action
        self.state['dialogue_act'] = "request" # "confirm_question"
        self.state['request_slots'][slot] = "UNK"


        ########################################################################
        # Inform the rest of informable slots
        ########################################################################

        #Chnaged at Dec 07 to have single slots for request action
        # for info_slot in self.state['rest_slots']:
        #     if info_slot in self.goal['inform_slots'].keys():
        #         self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]
        #
        # for info_slot in self.state['inform_slots'].keys():
        #     if info_slot in self.state['rest_slots']:
        #         self.state['rest_slots'].remove(info_slot)
      else:
        if len(self.state['request_slots']) == 0 and len(self.state['rest_slots']) == 0:
          self.state['dialogue_act'] = "thanks"
        else:
          self.state['dialogue_act'] = "inform"
        self.state['inform_slots'][slot] = dialog_constants.I_DO_NOT_CARE
        self.state['request_slots'].clear() # changed for unique action
    else: # this case should not appear
      if len(self.state['rest_slots']) > 0:
        random_slot = random.choice(self.state['rest_slots'])
        if random_slot in self.goal['inform_slots'].keys():
          self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
          self.state['rest_slots'].remove(random_slot)
          self.state['dialogue_act'] = "inform"
        elif random_slot in self.goal['request_slots'].keys():
          self.state['request_slots'][random_slot] = self.goal['request_slots'][random_slot]
          self.state['dialogue_act'] = "request"

  def response_multiple_choice(self, agent_action):
    """ Response for Multiple_Choice (System Action) """

    slot = list(agent_action['inform_slots'].keys())[0]
    if slot in self.goal['inform_slots'].keys():
      self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
    elif slot in self.goal['request_slots'].keys():
      self.state['inform_slots'][slot] = random.choice(agent_action['inform_slots'][slot])

    self.state['dialogue_act'] = "inform"
    if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
    if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]

  def response_inform(self, agent_action):
    """ Response for Inform (System Action) """

    if 'taskcomplete' in agent_action['inform_slots'].keys(): # check all the constraints from agents with user goal
      self.state['dialogue_act'] = "thanks"
      #if 'ticket' in self.state['rest_slots']: self.state['request_slots']['ticket'] = 'UNK'
      self.constraint_check = dialog_constants.CONSTRAINT_CHECK_SUCCESS

      if agent_action['inform_slots']['taskcomplete'] == dialog_constants.NO_VALUE_MATCH:
        self.state['history_slots']['ticket'] = dialog_constants.NO_VALUE_MATCH
        if 'ticket' in self.state['rest_slots']: self.state['rest_slots'].remove('ticket')
        if 'ticket' in self.state['request_slots'].keys(): del self.state['request_slots']['ticket']

        self.state['request_slots'].clear() # changed on Dec08

      for slot in self.goal['inform_slots'].keys():
        #  Deny, if the answers from agent can not meet the constraints of user
        if slot not in agent_action['inform_slots'].keys() or (self.goal['inform_slots'][slot].lower() != agent_action['inform_slots'][slot].lower()):
          self.state['dialogue_act'] = "deny"
          self.state['request_slots'].clear()
          self.state['inform_slots'].clear()
          self.constraint_check = dialog_constants.CONSTRAINT_CHECK_FAILURE
          break

      self.state['request_slots'].clear()
    else:
      for slot in agent_action['inform_slots'].keys():
        self.state['history_slots'][slot] = agent_action['inform_slots'][slot]

        if slot in self.goal['inform_slots'].keys():
          if agent_action['inform_slots'][slot] == self.goal['inform_slots'][slot]:
            if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)

            if len(self.state['request_slots']) > 0:
              self.state['dialogue_act'] = "request"
            elif len(self.state['rest_slots']) > 0:
              rest_slot_set = list(self.state['rest_slots']).copy()
              if 'ticket' in rest_slot_set:
                rest_slot_set.remove('ticket')

              if len(rest_slot_set) > 0:
                inform_slot = random.choice(rest_slot_set) # self.state['rest_slots']
                if inform_slot in self.goal['inform_slots'].keys():
                  self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                  self.state['dialogue_act'] = "inform"
                  self.state['rest_slots'].remove(inform_slot)
                elif inform_slot in self.goal['request_slots'].keys():
                  self.state['request_slots'][inform_slot] = 'UNK'
                  self.state['dialogue_act'] = "request"
              else:
                self.state['request_slots']['ticket'] = 'UNK'
                self.state['dialogue_act'] = "request"
            else: # how to reply here?
              self.state['dialogue_act'] = "thanks" # replies "closing"? or replies "confirm_answer"
              self.state['request_slots'].clear() # chagned on Dec08
          else: # != value  Should we deny here or ?
            ########################################################################
            # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
            ########################################################################
            self.state['dialogue_act'] = "inform"
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
            if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
            self.state['request_slots'].clear()
        else:
          if slot in self.state['rest_slots']:
            self.state['rest_slots'].remove(slot)
          if slot in self.state['request_slots'].keys():
            del self.state['request_slots'][slot]

          if len(self.state['request_slots']) > 0:
            request_set = list(self.state['request_slots'].keys())
            if 'ticket' in request_set:
              request_set.remove('ticket')

            if len(request_set) > 0:
              request_slot = random.choice(request_set)
            else:
              request_slot = 'ticket'

            self.state['request_slots'][request_slot] = "UNK"
            self.state['dialogue_act'] = "request"
          elif len(self.state['rest_slots']) > 0:
            rest_slot_set = self.state['rest_slots'].copy()
            if 'ticket' in rest_slot_set:
              rest_slot_set.remove('ticket')

            if len(rest_slot_set) > 0:
              inform_slot = random.choice(rest_slot_set) #self.state['rest_slots']
              if inform_slot in self.goal['inform_slots'].keys():
                self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                self.state['dialogue_act'] = "inform"
                self.state['rest_slots'].remove(inform_slot)

                # if 'ticket' in self.state['rest_slots']: # changed on Dec 8, should not request ticket now ?
                #     self.state['request_slots']['ticket'] = 'UNK'
                #     self.state['dialogue_act'] = "request"
              elif inform_slot in self.goal['request_slots'].keys():
                self.state['request_slots'][inform_slot] = self.goal['request_slots'][inform_slot]
                self.state['dialogue_act'] = "request"
            else:
              self.state['request_slots']['ticket'] = 'UNK'
              self.state['dialogue_act'] = "request"
          else:
            self.state['dialogue_act'] = "thanks" # or replies "confirm_answer"
            self.state['request_slots'].clear() # changed on Dec08

class NeuralSimulator(BaseUser):
  """ A rule-based user simulator for testing dialog policy """

  def __init__(self, params, ontology, goal_set=None):
    super().__init__(params, ontology, goal_set)

    self.task = params.task
    if self.task == "end_to_end":
      set_of_slots = set(self.slot_set.keys())
      for req in self.value_set["request"]:
        set_of_slots.add(req)

      set_of_slots.remove('request')
      set_of_slots.remove('act')
      set_of_slots.remove('other')
      self.slot_set = {slot: k for k, slot in enumerate(set_of_slots)}

    self.act_cardinality = len(self.act_set.keys())
    self.slot_cardinality = len(self.slot_set.keys())

    self.feasible_agent_actions = ontology.feasible_agent_actions
    self.feasible_user_actions = ontology.feasible_user_actions
    self.num_actions = len(self.feasible_agent_actions)
    self.num_actions_user = len(self.feasible_user_actions)

    self.max_turn = params.max_turn + 4
    self.state_dimension = 2 * self.act_cardinality + 9 * self.slot_cardinality + 3 + self.max_turn
    self.experience_replay_pool_size = params.pool_size

    self.agent_input_mode = "dialogue_act"
    self.hidden_size = params.hidden_dim
    self.training_examples = deque(maxlen=self.experience_replay_pool_size)
    self.predict_model = True

    self.model = SimulatorModel(self.num_actions, self.hidden_size, self.state_dimension, self.num_actions_user, 1)
    self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)

  def take_first_turn(self):
    """ randomly sample a start action based on user goal """

    starting_acts = dialog_constants.starting_dialogue_acts
    if self.task == 'end_to_end': starting_acts.append('greeting')
    self.state['dialogue_act'] = random.choice(starting_acts)

    # "sample" informed slots
    if len(self.goal['inform_slots']) > 0:
      known_slot = random.choice(list(self.goal['inform_slots'].keys()))
      self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

      if 'moviename' in self.goal['inform_slots'].keys():  # 'moviename' must appear in the first user turn
        self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']

      for slot in self.goal['inform_slots'].keys():
        if known_slot == slot or slot == 'moviename': continue
        self.state['rest_slots'].append(slot)

    self.state['rest_slots'].extend(self.goal['request_slots'].keys())

    # "sample" a requested slot
    request_slot_set = list(self.goal['request_slots'].keys())
    request_slot_set.remove('ticket')
    if len(request_slot_set) > 0:
      request_slot = random.choice(request_slot_set)
    else:
      request_slot = 'ticket'
    self.state['request_slots'][request_slot] = 'UNK'

    if len(self.state['request_slots']) == 0:
      self.state['dialogue_act'] = 'inform'

    if (self.state['dialogue_act'] in ['thanks', 'closing']):
      self.episode_over = True  # episode_over = True
    else:
      self.episode_over = False  # episode_over = False

    sample_action = {}
    sample_action['dialogue_act'] = self.state['dialogue_act']
    sample_action['inform_slots'] = self.state['inform_slots']
    sample_action['request_slots'] = self.state['request_slots']
    sample_action['turn_count'] = self.state['turn_count']

    self.add_nl_to_action(sample_action)
    return sample_action

  def prepare_user_goal_representation(self, user_goal):
    request_slots_rep = np.zeros((1, self.slot_cardinality))
    inform_slots_rep = np.zeros((1, self.slot_cardinality))
    for s in user_goal['request_slots']:
      stripped = s.strip()
      request_slots_rep[0, self.slot_set[stripped]] = 1
    for s in user_goal['inform_slots']:
      stripped = s.strip()
      inform_slots_rep[0, self.slot_set[stripped]] = 1

    self.user_goal_representation = np.hstack([request_slots_rep, inform_slots_rep])
    return self.user_goal_representation

  def train(self, batch_size=1, num_batches=1, verbose=False):
    """
    Train the world model with all the accumulated examples
    :param batch_size: self-explained
    :param num_batches: self-explained
    :return: None
    """
    self.total_loss = 0
    for iter_batch in range(num_batches):
      for iter in range(round(len(self.training_examples) / (batch_size))):
        self.optimizer.zero_grad()

        batch = self.sample_from_buffer(batch_size)
        state = torch.FloatTensor(batch.state)
        action = torch.LongTensor(batch.agent_action)
        reward = torch.FloatTensor(batch.reward)
        term = torch.FloatTensor(np.asarray(batch.term, dtype=np.int32))
        user_action = torch.LongTensor(batch.user_action).squeeze(1)

        reward_, term_, user_action_ = self.model(state, action)

        loss = F.mse_loss(reward_, reward) + \
             F.binary_cross_entropy_with_logits(term_, term) + \
             F.nll_loss(user_action_, user_action)
        loss.backward()

        self.optimizer.step()
        self.total_loss += loss.item()

    if verbose:
      print("Total cost on last batch for user modeling: %.4f, training replay pool %s" % (
        float(self.total_loss) / (float(len(self.training_examples)) / float(batch_size)),
        len(self.training_examples)))

  def train_by_iter(self, batch_size=1, num_batches=1):
    """
    Train the model with num_batches examples.
    :param batch_size:
    :param num_batches:
    :return: None
    """
    self.total_loss = 0
    for iter_batch in range(num_batches):
      self.optimizer.zero_grad()
      batch = self.sample_from_buffer(batch_size)
      state = torch.FloatTensor(batch.state)
      action = torch.LongTensor(batch.agent_action)
      reward = torch.FloatTensor(batch.reward)
      term = torch.FloatTensor(np.asarray(batch.term, dtype=np.int32))
      user_action = torch.LongTensor(batch.user_action).squeeze(1)

      reward_, term_, user_action_ = self.model(state, action)

      loss = F.mse_loss(reward_, reward) + \
           F.binary_cross_entropy_with_logits(term_, term) + \
           F.nll_loss(user_action_, user_action)
      loss.backward()

      self.optimizer.step()
      self.total_loss = loss.item()

    print("Total cost for last batch on user modeling: %.4f, training replay pool %s" % (
      float(self.total_loss), len(self.training_examples)))

  def next(self, dialogue_state, model_action):
    """
    Provide
    :param s: state representation from tracker
    :param a: last action from agent
    :return: next user action, termination and reward predicted by world model
    """
    self.state['turn_count'] += 2
    if (self.max_turn > 0 and self.state['turn_count'] >= self.max_turn):
      reward = - self.max_turn
      term = True
      self.state['request_slots'].clear()
      self.state['inform_slots'].clear()
      self.state['dialogue_act'] = "closing"
      response_action = {}
      response_action['dialogue_act'] = self.state['dialogue_act']
      response_action['inform_slots'] = self.state['inform_slots']
      response_action['request_slots'] = self.state['request_slots']
      response_action['turn_count'] = self.state['turn_count']
      return response_action, term, reward

    dialogue_state = self.prepare_state_representation(dialogue_state)
    goal_state = self.prepare_user_goal_representation(self.goal)
    stacked_state = np.hstack([dialogue_state, goal_state])

    state_tensor = torch.FloatTensor(stacked_state)
    action_tensor = torch.LongTensor(model_action['action_id']).view(-1,1)

    reward, term, action = self.model.predict(state_tensor, action_tensor)
    action = action.item()
    reward = reward.item()
    term = term.item()
    action = copy.deepcopy(self.feasible_user_actions[action])

    if action['dialogue_act'] == 'inform':
      if len(action['inform_slots'].keys()) > 0:
        slots = list(action['inform_slots'].keys())[0]
        if slots in self.goal['inform_slots'].keys():
          action['inform_slots'][slots] = self.goal['inform_slots'][slots]
        else:
          action['inform_slots'][slots] = dialog_constants.I_DO_NOT_CARE

    response_action = action

    term = term > 0.5

    if reward > 1:
      reward = 2 * self.max_turn
    elif reward < -1:
      reward = -self.max_turn
    else:
      reward = -1

    return response_action, term, reward

  def action_index(self, act_slot_response):
    """ Return the index of action """
    del act_slot_response['turn_count']
    del act_slot_response['nl']

    for i in act_slot_response['inform_slots'].keys():
      act_slot_response['inform_slots'][i] = 'PLACEHOLDER'

    # rule
    if act_slot_response['dialogue_act'] == 'request': act_slot_response['inform_slots'] = {}
    if act_slot_response['dialogue_act'] in ['thanks', 'deny', 'closing']:
      act_slot_response['inform_slots'] = {}
      act_slot_response['request_slots'] = {}

    for (i, action) in enumerate(self.feasible_user_actions):
      if act_slot_response == action:
        return i

    print('feasible_user_actions', self.feasible_user_actions)
    print('act_slot_response', act_slot_response)
    print("action index not found")
    pdb.set_trace()
    return None

  def prepare_state_representation(self, state):
    """ Create the representation for each state """

    user_action = state['user_action']
    current_slots = state['current_slots']
    kb_results_dict = state['kb_results_dict']
    agent_last = state['agent_action']

    ########################################################################
    #   Create one-hot of acts to represent the current user action
    ########################################################################
    user_act_rep = np.zeros((1, self.act_cardinality))
    user_act_rep[0, self.act_set[user_action['dialogue_act']]] = 1.0

    ########################################################################
    #     Create bag of inform slots representation to represent the current user action
    ########################################################################
    user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
    for slot in user_action['inform_slots'].keys():
      user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

    ########################################################################
    #   Create bag of request slots representation to represent the current user action
    ########################################################################
    user_request_slots_rep = np.zeros((1, self.slot_cardinality))
    for slot in user_action['request_slots'].keys():
      user_request_slots_rep[0, self.slot_set[slot]] = 1.0

    ########################################################################
    #   Creat bag of filled_in slots based on the current_slots
    ########################################################################
    current_slots_rep = np.zeros((1, self.slot_cardinality))
    for slot in current_slots['inform_slots']:
      current_slots_rep[0, self.slot_set[slot]] = 1.0

    ########################################################################
    #   Encode last agent act
    ########################################################################
    agent_act_rep = np.zeros((1, self.act_cardinality))
    if agent_last:
      agent_act_rep[0, self.act_set[agent_last['dialogue_act']]] = 1.0

    ########################################################################
    #   Encode last agent inform slots
    ########################################################################
    agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
    if agent_last:
      for slot in agent_last['inform_slots'].keys():
        agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

    ########################################################################
    #   Encode last agent request slots
    ########################################################################
    agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
    if agent_last:
      for slot in agent_last['request_slots'].keys():
        agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

    # turn_rep = np.zeros((1, 1)) + state['turn_count'] / 10.
    turn_rep = np.zeros((1, 1))

    ########################################################################
    #  One-hot representation of the turn count?
    ########################################################################
    turn_onehot_rep = np.zeros((1, self.max_turn))
    turn_onehot_rep[0, state['turn_count']] = 1.0

    ########################################################################
    #   Representation of KB results (scaled counts)
    ########################################################################
    kb_count_rep = np.zeros((1, self.slot_cardinality + 1))
    ########################################################################
    #   Representation of KB results (binary)
    ########################################################################
    kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

    # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
    # for slot in kb_results_dict:
    #     if slot in self.slot_set:
    #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
    #
    # ########################################################################
    # #   Representation of KB results (binary)
    # ########################################################################
    # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
    #     kb_results_dict['matching_all_constraints'] > 0.)
    # for slot in kb_results_dict:
    #     if slot in self.slot_set:
    #         kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

    self.final_representation = np.hstack(
      [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
       agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
    return self.final_representation

  def store_experience(self, current_state, agent_action, next_state, reward, term, user_action):
    """ Register feedback from the environment, to be stored as future training data for world model"""

    state_t_rep = self.prepare_state_representation(current_state)
    goal_rep = self.prepare_user_goal_representation(self.goal)
    state_t_rep = np.hstack([state_t_rep, goal_rep])
    # agent_action_t = agent_action
    # user_action_t = user_action

    action_idx = self.action_index(copy.deepcopy(user_action))
    reward_t = reward
    term_t = term

    if reward_t > 1:
      reward_t = 1
    elif reward_t < -1:
      reward_t = -1
    elif reward_t == -1:
      reward_t = -0.1

    state_tplus1_rep = self.prepare_state_representation(next_state)
    training_example_for_user = (state_t_rep, agent_action, state_tplus1_rep, reward_t, term, action_idx)

    if self.predict_model:
      self.training_examples.append(training_example_for_user)

  def sample_from_buffer(self, batch_size):
    """Sample batch size examples from experience buffer and convert it to torch readable format"""

    batch = [random.choice(self.training_examples) for i in range(batch_size)]
    np_batch = []
    for x in range(len(Transition._fields)):
      v = []
      for i in range(batch_size):
        v.append(batch[i][x])
      np_batch.append(np.vstack(v))

    return Transition(*np_batch)



 # Used for simulating the user portion of a cycle, mostly template retrieval

class UserSimulator(BaseUser):
  def __init__(self, args, ontology, kind="movie"):
    super().__init__(args, ontology, kind)
    self.slot_error_prob = 0.0
    self.intent_error_prob = 0.0
    self.text_generator = None
    self.learning_phase = None
    self.goal_set = None

  def take_first_turn(self):
    """ randomly sample a start action based on user goal """

    # dialogue_act = random.choice(list(dialog_constants.start_dia_acts.keys())
    self.state['dialogue_act'] = 'request' # since this is the only valid option
    # impossible to sample since not an option
    # if (self.state['dialogue_act'] in ['thanks','closing']):
    #   self.episode_over = True

    # sample informed slots
    if len(self.goal['inform_slots']) > 0:
      user_constraints = list(self.goal['inform_slots'].keys())
      chosen_slot = random.choice(user_constraints)
      self.state['inform_slots'][chosen_slot] = self.goal['inform_slots'][chosen_slot]
      if 'moviename' in user_constraints: # 'moviename' must appear in the first user turn
        self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']

      for slot in user_constraints:
        if slot == chosen_slot or slot == 'moviename': continue
        self.state['remaining_slots'].append(slot)

    request_slot_set = list(self.goal['request_slots'].keys())
    self.state['remaining_slots'].extend(request_slot_set)

    # sample a requested slot
    request_slot_set.remove('ticket')
    if len(request_slot_set) > 0:
      request_slot = random.choice(request_slot_set)
    else:
      request_slot = 'ticket'
    self.state['request_slots'][request_slot] = 'UNK'

    if len(self.state['request_slots']) == 0:
      print(request_slot_set)
      print(self.state)
      raise(RuntimeError("how is it possible to reach here?"))
      self.state['dialogue_act'] = 'inform'

    sample_action = {}
    sample_action['dialogue_act'] = self.state['dialogue_act']
    sample_action['inform_slots'] = self.state['inform_slots']
    sample_action['request_slots'] = self.state['request_slots']
    sample_action['turn_count'] = self.state['turn_count']

    return self.act_to_nl(sample_action)

  def next(self, agent_action):
    """ Generate next User Action based on last Agent Action """
    self.state['turn_count'] += 2
    self.episode_over = False
    self.dialog_status = dialog_constants.NO_OUTCOME_YET

    dialogue_act = agent_action['dialogue_act']
    informs = agent_action['inform_slots']
    requests = agent_action['request_slots']

    if (self.max_turn > 0 and self.state['turn_count'] > self.max_turn):
      self.dialog_status = dialog_constants.FAILED_DIALOG
      self.episode_over = True
      self.state['dialogue_act'] = "closing"
    else:
      self.state['history_slots'].update(self.state['inform_slots'])
      self.state['inform_slots'].clear()

      c = self.state['turn_count']

      if dialogue_act == "inform":
        if 'taskcomplete' in informs.keys():
          self.response_taskcomplete(informs)
        else:
          self.response_inform(informs)
      elif dialogue_act == "multiple_choice":
        self.response_multiple_choice(informs)
      elif dialogue_act == "request":
        self.response_request(requests)
      elif dialogue_act == "thanks":
        self.response_thanks(informs)
      elif dialogue_act == "confirm_answer":
        self.response_confirm_answer()
      elif dialogue_act == "closing":
        self.episode_over = True
        self.state['dialogue_act'] = "thanks"

    # self.corrupt(self.state)
    response_action = {'dialogue_act': self.state['dialogue_act'],
                       'inform_slots': self.state['inform_slots'],
                       'request_slots': self.state['request_slots'],
                       'turn_count': self.state['turn_count'],
                       'nl': "" }
    # add NL to dia_act
    self.act_to_nl(response_action)
    return response_action, self.episode_over, self.dialog_status

  def response_taskcomplete(self, informs):
    # if the agent believes it has completed the task already, wrap up the chat
    self.state['dialogue_act'] = "thanks"
    # check that the final ticket is the right value
    if informs['taskcomplete'] == dialog_constants.NO_VALUE_MATCH:
      self.state['history_slots']['ticket'] = dialog_constants.NO_VALUE_MATCH
      self.clear_option(slot='ticket')

    # Assume, but verify that all constraints are met
    self.constraint_check = dialog_constants.CONSTRAINT_CHECK_SUCCESS
    for slot in self.goal['inform_slots'].keys():
      #  Deny, if the answers from agent can not meet the constraints of user
      missing_slot = slot not in informs.keys()
      target_slot = self.goal['inform_slots'][slot].lower()
      predicted_slot = informs[slot].lower()

      if missing_slot or (target_slot != predicted_slot):
        self.state['dialogue_act'] = "deny"
        self.state['request_slots'].clear()
        self.state['inform_slots'].clear()
        self.constraint_check = dialog_constants.CONSTRAINT_CHECK_FAILURE
        pdb.set_trace()
        break
    #if 'ticket' in self.state['remaining_slots']: self.state['request_slots']['ticket'] = 'UNK'

  def response_inform(self, informs):
    """ In a typical dialogue, the assumption is that the main goal of the
    user is to put forward a set of constraints, so we start with Inform """

    # for every possible slot
    for slot in informs.keys():
      self.state['history_slots'][slot] = informs[slot]
      # if that slot is one of the current user constraints
      if slot in self.goal['inform_slots'].keys():
        self.check_inform_validity(informs, slot)
      else:  # the agent just informed something you didn't ask for yet
        self.clear_option(slot)
        # select a next slot from the request set
        if len(self.state['request_slots']) > 0:
          request_set = list(self.state['request_slots'].keys())
          if 'ticket' in request_set:
            request_set.remove('ticket')

          if len(request_set) > 0:
            request_slot = random.choice(request_set)
          else:
            request_slot = 'ticket'

          self.state['request_slots'][request_slot] = "UNK"
          self.state['dialogue_act'] = "request"

          # select the next slot from the remaining set
        elif len(self.state['remaining_slots']) > 0:
          # this copy is needed only because of the ticket thing, which
          # we will remove soon ...
          rest_slot_set = copy.deepcopy(self.state['remaining_slots'])
          if 'ticket' in rest_slot_set:
            rest_slot_set.remove('ticket')
          # self.choose_next_slot(remove_ticket=True)
          if len(rest_slot_set) > 0:
            inform_slot = random.choice(rest_slot_set) #self.state['remaining_slots']
            if inform_slot in self.goal['inform_slots'].keys():
              self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
              self.state['dialogue_act'] = "inform"
              self.state['remaining_slots'].remove(inform_slot)

              if 'ticket' in self.state['remaining_slots']:
                self.request_ticket()
            elif inform_slot in self.goal['request_slots'].keys():
              self.state['request_slots'][inform_slot] = self.goal['request_slots'][inform_slot]
              self.state['dialogue_act'] = "request"
          else:
            self.request_ticket()
        else:
          self.state['dialogue_act'] = "thanks" # or replies "confirm_answer"

  def response_multiple_choice(self, informs):
    """ Inform the agent about user's multiple choice desire """
    slot = informs.keys()[0]
    if slot in self.goal['inform_slots'].keys():
      self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
    elif slot in self.goal['request_slots'].keys():
      self.state['inform_slots'][slot] = random.choice(informs[slot])

    self.state['dialogue_act'] = "inform"
    self.clear_option(slot)

  def response_request(self, requests):
    """ Response for Request (System Action) """

    if len(requests.keys()) > 0:
      slot = list(requests.keys())[0] # only one slot
      is_inform = slot in self.goal['inform_slots'].keys()
      is_request = slot in self.goal['request_slots'].keys()
      is_remaining = slot in self.state['remaining_slots']
      is_history = slot in self.state['history_slots'].keys()

      # if the agents asks about a slot value,
      # and the user has the answer, then the user will inform the agent
      if is_inform:
        #and slot not in self.state['request_slots'].keys():
        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
        self.state['dialogue_act'] = "inform"
        self.clear_option(slot)
        self.state['request_slots'].clear()
      # or despite the fact that the requested slot has been previously
      # answered by the user, the user will inform again
      elif is_request and not is_remaining and is_history:
        self.state['dialogue_act'] = "inform"
        self.state['inform_slots'][slot] = self.state['history_slots'][slot]
        self.state['request_slots'].clear()
      # request slot in user's goal's request slots, and not answered yet
      elif is_request and is_remaining:
        # semantically, this probably means the user is actually trying
        # to confirm a slot, rather than requesting information from user
        self.state['dialogue_act'] = "request" # "confirm_question"
        self.state['request_slots'][slot] = "UNK"

        # Inform the rest of informable slots because ???
        for info_slot in self.state['remaining_slots']:
          if info_slot in self.goal['inform_slots'].keys():
            self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]
        for info_slot in self.state['inform_slots'].keys():
          if info_slot in self.state['remaining_slots']:
            self.state['remaining_slots'].remove(info_slot)
      else:
        # no_more_requests = len(self.state['request_slots']) == 0
        # no_more_remaining = len(self.state['remaining_slots']) == 0
        # if no_more_requests and no_more_remaining:
        #   self.state['dialogue_act'] = "thanks"
        # else:
        self.state['dialogue_act'] = "inform"
        self.state['inform_slots'][slot] = dialog_constants.I_DO_NOT_CARE
    else: # this case should not appear
      if len(self.state['remaining_slots']) > 0:
        self.choose_next_slot()
        # [chosen_slot] = self.goal['request_slots'][random_slot]

  def response_confirm_answer(self):
    """ After the agent has just confirmed an answer, that slot should now
     be complete, so we move on to choose another new slot to discuss"""

    # if there are any remaining slots to fill, then pick one at random
    if len(self.state['remaining_slots']) > 0:
      self.choose_next_slot()
    else:  # otherwise there are no slots let to fill, so we're done!
      self.state['dialogue_act'] = "thanks"

  def response_thanks(self, informs):
    self.episode_over = True
    self.dialog_status = dialog_constants.SUCCESS_DIALOG

    request_slot_set = copy.deepcopy(list(self.state['request_slots'].keys()))
    if 'ticket' in request_slot_set:
      request_slot_set.remove('ticket')
    remaining_slot_set = copy.deepcopy(self.state['remaining_slots'])
    if 'ticket' in remaining_slot_set:
      remaining_slot_set.remove('ticket')

    # if there are unmet requests or constraints
    if len(request_slot_set) > 0 or len(remaining_slot_set) > 0:
      self.dialog_status = dialog_constants.FAILED_DIALOG
    for hist_slot, hist_value in self.state['history_slots'].items():
      # if we failed to find a value from the user goal
      failed_to_find_match = (hist_value == dialog_constants.NO_VALUE_MATCH)
      # if we found the wrong value
      goals = self.goal['inform_slots'].keys()
      goal = self.goal['inform_slots']
      found_wrong_match = (hist_slot in goals) and (hist_value != goal[hist_slot])

      if failed_to_find_match or found_wrong_match:
        self.dialog_status = dialog_constants.FAILED_DIALOG

    if 'ticket' in informs.keys():
      if informs['ticket'] == dialog_constants.NO_VALUE_MATCH:
        self.dialog_status = dialog_constants.FAILED_DIALOG
    if self.constraint_check == dialog_constants.CONSTRAINT_CHECK_FAILURE:
      self.dialog_status = dialog_constants.FAILED_DIALOG

  def clear_option(self, slot):
    if slot in self.state['remaining_slots']:
      self.state['remaining_slots'].remove(slot)
    if slot in self.state['request_slots'].keys():
      del self.state['request_slots'][slot]

  def choose_next_slot(self):
    chosen_slot = random.choice(self.state['remaining_slots'])
    # if the chosen slot is a request, then set the act as accordingly
    if chosen_slot in self.goal['request_slots'].keys():
      self.state['dialogue_act'] = "request"
      # by placing the slot in the set of requests slots, it lets the
      # Policy Module know to check for this value when calculating reward
      self.state['request_slots'][chosen_slot] = "UNK"
    # if its a inform slot, then obviously set as a inform act instead
    elif chosen_slot in self.goal['inform_slots'].keys():
      self.state['dialogue_act'] = "inform"
      self.state['inform_slots'][chosen_slot] = self.goal['inform_slots'][chosen_slot]
      self.state['remaining_slots'].remove(chosen_slot)

  def request_ticket(self):
    self.state['request_slots']['ticket'] = 'UNK'
    self.state['dialogue_act'] = "request"

  def check_inform_validity(self, informs, slot):
    """ if the value that was informed by the agent was equal to the one
    determined by our original goal, then choose the next constraint
    otherwise, we help the agent out by repeating out constraint
    alternatively, we could just "deny" when the agent gets it wrong """
    if informs[slot] == self.goal['inform_slots'][slot]:
      # the user has informed us about some slot we had a question for
      if slot in self.state['remaining_slots']:
        self.state['remaining_slots'].remove(slot)
      if len(self.state['request_slots']) > 0:
        self.state['dialogue_act'] = "request"
      elif len(self.state['remaining_slots']) > 0:
        rest_slot_set = copy.deepcopy(self.state['remaining_slots'])
        if 'ticket' in rest_slot_set:
          rest_slot_set.remove('ticket')

        if len(rest_slot_set) > 0:
          self.choose_next_slot()
        else:
          self.request_ticket()
      else: # how to reply here? perhaps "closing" or "confirm_answer"?
        self.state['dialogue_act'] = "thanks"
    else:
      # When agent informs(slot=value), where the value is different from the
      # constraint in user goal, should we deny or just repeat the correct value?
      self.state['dialogue_act'] = "inform"
      self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
      if slot in self.state['remaining_slots']:
        self.state['remaining_slots'].remove(slot)

  def act_to_nl(self, user_action):
    if self.text_generator is None:
      raise(RuntimeError("text generator has not been set"))

    user_action['nl'] = self.text_generator.generate(user_action, "usr")
    return user_action
    """ Add natural language to user dialogue act

    then proxy the natural language understanding component
    if self.simulator_act_level == 1:
        user_nlu_res = self.nlu_model.generate_dia_act(user_action['nl']) # NLU
        if user_nlu_res != None:
            #user_nlu_res['dialogue_act'] = user_action['dialogue_act'] # or not?
            user_action.update(user_nlu_res)
    """
