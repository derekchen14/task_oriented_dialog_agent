import pdb, sys
import random, copy
from datasets.ddq import constants as dialog_config

class BaseUser(object):
  def __init__(self, args, ontology, kind="movie"):
    self.max_turn = args.max_turn
    self.num_episodes = args.epochs
    self.kind = kind
    self.do_print= True

    self.act_set = ontology.acts
    self.slot_set = ontology.slots
    self.relation_set = ontology.relations
    self.value_set = ontology.values

  def _sample_goal(self):
    return random.choice(self.goal_sets[self.learning_phase])

  def initialize_episode(self):
    raise(NotImplementedError, "User episode is not initialized")
  def next(self, agent_action):
    raise(NotImplementedError, "User cannot take next step")

# Used for simulating the user portion of a cycle, mostly template retrieval
class UserSimulator(BaseUser):
  def __init__(self, args, ontology, kind="movie"):
    super().__init__(args, ontology, kind)
    self.slot_error_prob = 0.0
    self.intent_error_prob = 0.0
    self.text_generator = None
    self.learning_phase = None
    self.goal_sets = None

  def initialize_episode(self):
    self.state = {
      'history_slots': {},
      'inform_slots': {},
      'request_slots': {},
      'remaining_slots': [],
      'turn_count': 0
    }

    self.goal = self._sample_goal()
    self.goal['request_slots']['ticket'] = 'UNK'
    self.episode_over = False
    self.user_action = self._sample_action()

    self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE
    self.dialogue_status = dialog_config.NO_OUTCOME_YET

  def _sample_action(self):
    """ randomly sample a start action based on user goal """

    # dialogue_act = random.choice(list(dialog_config.start_dia_acts.keys())
    self.state['diaact'] = 'request' # since this is the only valid option
    # if (self.state['diaact'] in ['thanks','closing']):  impossible to sample since not an option
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
      self.state['diaact'] = 'inform'

    sample_action = {}
    sample_action['diaact'] = self.state['diaact']
    sample_action['inform_slots'] = self.state['inform_slots']
    sample_action['request_slots'] = self.state['request_slots']
    sample_action['turn_count'] = self.state['turn_count']

    return self.act_to_nl(sample_action)

  def next(self, system_action):
    """ Generate next User Action based on last System Action """
    self.state['turn_count'] += 2
    self.episode_over = False
    self.dialog_status = dialog_config.NO_OUTCOME_YET

    sys_act = system_action['diaact']

    if (self.max_turn > 0 and self.state['turn_count'] > self.max_turn):
      self.dialog_status = dialog_config.FAILED_DIALOG
      self.episode_over = True
      self.state['diaact'] = "closing"
    else:
      self.state['history_slots'].update(self.state['inform_slots'])
      self.state['inform_slots'].clear()

      if sys_act == "inform":
        self.response_inform(system_action)
      elif sys_act == "multiple_choice":
        self.response_multiple_choice(system_action)
      elif sys_act == "request":
        self.response_request(system_action)
      elif sys_act == "thanks":
        self.response_thanks(system_action)
      elif sys_act == "confirm_answer":
        self.response_confirm_answer(system_action)
      elif sys_act == "closing":
        self.episode_over = True
        self.state['diaact'] = "thanks"

    # self.corrupt(self.state)
    response_action = {'diaact': self.state['diaact'],
                       'inform_slots': self.state['inform_slots'],
                       'request_slots': self.state['request_slots'],
                       'turn_count': self.state['turn_count'],
                       'nl': "" }
    # add NL to dia_act
    self.act_to_nl(response_action)
    return response_action, self.episode_over, self.dialog_status


  def response_confirm_answer(self, system_action):
    if len(self.state['remaining_slots']) > 0:
      request_slot = random.choice(self.state['remaining_slots'])

      if request_slot in self.goal['request_slots'].keys():
        self.state['diaact'] = "request"
        self.state['request_slots'][request_slot] = "UNK"
      elif request_slot in self.goal['inform_slots'].keys():
        self.state['diaact'] = "inform"
        self.state['inform_slots'][request_slot] = self.goal['inform_slots'][request_slot]
        if request_slot in self.state['remaining_slots']:
          self.state['remaining_slots'].remove(request_slot)
    else:
      self.state['diaact'] = "thanks"

  def response_thanks(self, system_action):
    self.episode_over = True
    self.dialog_status = dialog_config.SUCCESS_DIALOG

    request_slot_set = copy.deepcopy(list(self.state['request_slots'].keys()))
    if 'ticket' in request_slot_set:
      request_slot_set.remove('ticket')
    rest_slot_set = copy.deepcopy(self.state['remaining_slots'])
    if 'ticket' in rest_slot_set:
      rest_slot_set.remove('ticket')

    if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
      self.dialog_status = dialog_config.FAILED_DIALOG

    for info_slot in self.state['history_slots'].keys():
      if self.state['history_slots'][info_slot] == dialog_config.NO_VALUE_MATCH:
        self.dialog_status = dialog_config.FAILED_DIALOG
      if info_slot in self.goal['inform_slots'].keys():
        if self.state['history_slots'][info_slot] != self.goal['inform_slots'][info_slot]:
          self.dialog_status = dialog_config.FAILED_DIALOG

    if 'ticket' in system_action['inform_slots'].keys():
      if system_action['inform_slots']['ticket'] == dialog_config.NO_VALUE_MATCH:
        self.dialog_status = dialog_config.FAILED_DIALOG

    if self.constraint_check == dialog_config.CONSTRAINT_CHECK_FAILURE:
      self.dialog_status = dialog_config.FAILED_DIALOG

  def response_request(self, system_action):
    """ Response for Request (System Action) """

    if len(system_action['request_slots'].keys()) > 0:
      slot = system_action['request_slots'].keys()[0] # only one slot
      if slot in self.goal['inform_slots'].keys(): # request slot in user's constraints  #and slot not in self.state['request_slots'].keys():
        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
        self.state['diaact'] = "inform"
        if slot in self.state['remaining_slots']: self.state['remaining_slots'].remove(slot)
        if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
        self.state['request_slots'].clear()
      elif slot in self.goal['request_slots'].keys() and slot not in self.state['remaining_slots'] and slot in self.state['history_slots'].keys(): # the requested slot has been answered
        self.state['inform_slots'][slot] = self.state['history_slots'][slot]
        self.state['request_slots'].clear()
        self.state['diaact'] = "inform"
      elif slot in self.goal['request_slots'].keys() and slot in self.state['remaining_slots']: # request slot in user's goal's request slots, and not answered yet
        self.state['diaact'] = "request" # "confirm_question"
        self.state['request_slots'][slot] = "UNK"

        ########################################################################
        # Inform the rest of informable slots
        ########################################################################
        for info_slot in self.state['remaining_slots']:
          if info_slot in self.goal['inform_slots'].keys():
            self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]

        for info_slot in self.state['inform_slots'].keys():
          if info_slot in self.state['remaining_slots']:
            self.state['remaining_slots'].remove(info_slot)
      else:
        if len(self.state['request_slots']) == 0 and len(self.state['remaining_slots']) == 0:
          self.state['diaact'] = "thanks"
        else:
          self.state['diaact'] = "inform"
        self.state['inform_slots'][slot] = dialog_config.I_DO_NOT_CARE
    else: # this case should not appear
      if len(self.state['remaining_slots']) > 0:
        random_slot = random.choice(self.state['remaining_slots'])
        if random_slot in self.goal['inform_slots'].keys():
          self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
          self.state['remaining_slots'].remove(random_slot)
          self.state['diaact'] = "inform"
        elif random_slot in self.goal['request_slots'].keys():
          self.state['request_slots'][random_slot] = self.goal['request_slots'][random_slot]
          self.state['diaact'] = "request"

  def response_multiple_choice(self, system_action):
    """ Response for Multiple_Choice (System Action) """

    slot = system_action['inform_slots'].keys()[0]
    if slot in self.goal['inform_slots'].keys():
      self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
    elif slot in self.goal['request_slots'].keys():
      self.state['inform_slots'][slot] = random.choice(system_action['inform_slots'][slot])

    self.state['diaact'] = "inform"
    if slot in self.state['remaining_slots']: self.state['remaining_slots'].remove(slot)
    if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]

  def response_inform(self, system_action):
    """ Response for Inform (System Action) """

    if 'taskcomplete' in system_action['inform_slots'].keys(): # check all the constraints from agents with user goal
      self.state['diaact'] = "thanks"
      #if 'ticket' in self.state['remaining_slots']: self.state['request_slots']['ticket'] = 'UNK'
      self.constraint_check = dialog_config.CONSTRAINT_CHECK_SUCCESS

      if system_action['inform_slots']['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
        self.state['history_slots']['ticket'] = dialog_config.NO_VALUE_MATCH
        if 'ticket' in self.state['remaining_slots']: self.state['remaining_slots'].remove('ticket')
        if 'ticket' in self.state['request_slots'].keys(): del self.state['request_slots']['ticket']

      for slot in self.goal['inform_slots'].keys():
        #  Deny, if the answers from agent can not meet the constraints of user
        if slot not in system_action['inform_slots'].keys() or (self.goal['inform_slots'][slot].lower() != system_action['inform_slots'][slot].lower()):
          self.state['diaact'] = "deny"
          self.state['request_slots'].clear()
          self.state['inform_slots'].clear()
          self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE
          break
    else:
      for slot in system_action['inform_slots'].keys():
        self.state['history_slots'][slot] = system_action['inform_slots'][slot]

        if slot in self.goal['inform_slots'].keys():
          if system_action['inform_slots'][slot] == self.goal['inform_slots'][slot]:
            if slot in self.state['remaining_slots']: self.state['remaining_slots'].remove(slot)

            if len(self.state['request_slots']) > 0:
              self.state['diaact'] = "request"
            elif len(self.state['remaining_slots']) > 0:
              rest_slot_set = copy.deepcopy(self.state['remaining_slots'])
              if 'ticket' in rest_slot_set:
                rest_slot_set.remove('ticket')

              if len(rest_slot_set) > 0:
                inform_slot = random.choice(rest_slot_set) # self.state['remaining_slots']
                if inform_slot in self.goal['inform_slots'].keys():
                  self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                  self.state['diaact'] = "inform"
                  self.state['remaining_slots'].remove(inform_slot)
                elif inform_slot in self.goal['request_slots'].keys():
                  self.state['request_slots'][inform_slot] = 'UNK'
                  self.state['diaact'] = "request"
              else:
                self.state['request_slots']['ticket'] = 'UNK'
                self.state['diaact'] = "request"
            else: # how to reply here?
              self.state['diaact'] = "thanks" # replies "closing"? or replies "confirm_answer"
          else: # != value  Should we deny here or ?
            ########################################################################
            # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
            ########################################################################
            self.state['diaact'] = "inform"
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
            if slot in self.state['remaining_slots']: self.state['remaining_slots'].remove(slot)
        else:
          if slot in self.state['remaining_slots']:
            self.state['remaining_slots'].remove(slot)
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
            self.state['diaact'] = "request"
          elif len(self.state['remaining_slots']) > 0:
            rest_slot_set = copy.deepcopy(self.state['remaining_slots'])
            if 'ticket' in rest_slot_set:
              rest_slot_set.remove('ticket')

            if len(rest_slot_set) > 0:
              inform_slot = random.choice(rest_slot_set) #self.state['remaining_slots']
              if inform_slot in self.goal['inform_slots'].keys():
                self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                self.state['diaact'] = "inform"
                self.state['remaining_slots'].remove(inform_slot)

                if 'ticket' in self.state['remaining_slots']:
                  self.state['request_slots']['ticket'] = 'UNK'
                  self.state['diaact'] = "request"
              elif inform_slot in self.goal['request_slots'].keys():
                self.state['request_slots'][inform_slot] = self.goal['request_slots'][inform_slot]
                self.state['diaact'] = "request"
            else:
              self.state['request_slots']['ticket'] = 'UNK'
              self.state['diaact'] = "request"
          else:
            self.state['diaact'] = "thanks" # or replies "confirm_answer"

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
            #user_nlu_res['diaact'] = user_action['diaact'] # or not?
            user_action.update(user_nlu_res)
    """

  def debug_with_fake_goal(self):
      """ Build a fake goal mannuall for debugging purposes """
      self.goal['inform_slots'].clear()
      self.goal['inform_slots']['city'] = 'seattle'
      self.goal['inform_slots']['numberofpeople'] = '1'
      #self.goal['inform_slots']['theater'] = 'amc pacific place 11 theater'
      self.goal['inform_slots']['starttime'] = '6pm'
      self.goal['inform_slots']['date'] = 'March 16th'
      self.goal['inform_slots']['restaurantname'] = 'wild ginger seattle'
      #self.goal['inform_slots']['distanceconstraints'] = 'close to 95833'
      self.goal['request_slots'].clear()
      self.goal['request_slots']['reservation'] = 'UNK'
      self.goal['request_slots']['address'] = 'UNK'
      #self.goal['request_slots']['starttime'] = 'UNK'
      #self.goal['request_slots']['date'] = 'UNK'

class CommandLineUser(BaseUser):
  def __init__(self, args, ontology, kind="movie"):
    super().__init__(args, ontology, kind)
    self.learning_phase = "train"
    self.agent_input_mode = "raw_text" # or "dialogue_act"
    self.do_print = False

  def initialize_episode(self):
    self.goal = self._sample_goal()
    self.turn_count = -2
    self.user_action = {'diaact':'UNK', 'inform_slots':{}, 'request_slots':{}}

    if self.agent_input_mode == "raw_text":
      print("Your input will be raw text so the system expects the dialogue \
        model to include a NLU module for intent classification")
    elif self.agent_input_mode == "dialogue_act":
      print("The system expects a properly written user intent of the form \
        act(slot=value) such as inform(city=Los_Angeles).  Multiple intents \
        can be included by joining them with a comma. Spaces will be removed.")
    print("Your goal: {}".format(self.goal))
    self.next(agent_action={})

  def next(self, agent_action):
    # Generate an action by getting input interactively from the command line
    self.turn_count += 2
    command_line_input = input("{}) user: ".format(self.turn_count))
    """ a command line user cannot end the dialogue.  thus, they cannot
    decide that the episode is over and the dialogue status is always 0.
    (meaning No Outcome Yet, as opposed to -1 of Fail and 1 of Success)"""
    episode_over = (self.max_turn < self.turn_count)
    dialog_status = 0

    if self.agent_input_mode == "raw_text":
      self.user_action['nl'] = command_line_input
    elif self.agent_input_mode == "dialogue_act":
      self.parse_intent(command_line_input)
    self.user_action['turn_count'] = self.turn_count

    return self.user_action, episode_over, dialog_status

  def parse_intent(self, command_line_input):
    """ Parse input from command line into dialogue act form """
    intents = command_line_input.strip(' ').strip('\n').strip('\r').split(',')
    for intent in intents:
      idx = intent.find('(')
      act = intent[0:idx]
      slot, value = intent[idx+1:-1].split("=") # -1 is to skip the closing ')'

      self.error_checking(idx, act, slot, value)
      self.user_action["diaact"] = act
      self.user_action["nl"] = "N/A"
      self.user_action["{}_slots".format(act)][slot] = value

  def error_checking(self, idx, act, slot, value):
    if idx < 0:
      raise(ValueError("input is not properly formatted as a user intent"))
    if act not in self.act_set:
      raise(ValueError("dialogue act is not part of allowable act set"))
    if slot not in self.slot_set:
      raise(ValueError("slot is not part of allowable slot set"))
    if value not in self.value_set[slot]:
      raise(ValueError("value is not part of the available value set"))