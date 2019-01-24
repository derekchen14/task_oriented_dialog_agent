import pdb, sys
import random, copy

# Used for simulating the user portion of a cycle, mostly template retrieval
class UserSimulator(object):

  def __init__(self, args, intent_sets, kind="movie"):
    self.max_turns = args.max_turns
    self.num_episodes = args.num_iters
    self.kind = kind

    self.slot_error_prob = 0.0
    self.intent_error_prob = 0.0
    self.run_mode = None
    self.act_level = None
    self.learning_phase = "supervised"

    act_set, slot_set, value_set, goal_set = intent_sets
    self.act_set = act_set
    self.slot_set = slot_set
    self.relation_set = ["="]
    self.value_set = value_set
    self.goal_set = goal_set

  def initialize_episode(self):
    print("Initializing user simulator, generating a user goal")
    # logger.info("Initializing user simulator, generating a user goal")
    self.state = {
      'historical_slots': {},
      'inform_slots': {},
      'request_slots': {},
      'remaining_slots': {},
      'turn_count': 0
    }

    self.goal = self._sample_goal(self.start_set)
    self.goal['request_slots']['ticket'] = 'UNK'
    self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

    episode_over = True
    while episode_over:
      # keep sampling actions until we get one that isn't the end
      episode_over, user_action = self._sample_action()
    self.episode_over = episode_over
    self.dialogue_status = dialog_config.NO_OUTCOME_YET

    return user_action

  def _sample_goal(self):
    return random.choice(self.start[self.learning_phase])
  def _sample_action(self):
    raise(NotImplementedError, "User simulator cannot sample actions")

  # def next(self):
  #   raise(NotImplementedError, "User simulator cannot take next step")
  def next(self, system_action):
    """ Generate next User Action based on last System Action """
    self.state['turn'] += 2
    self.episode_over = False
    self.dialog_status = dialog_config.NO_OUTCOME_YET

    sys_act = system_action['diaact']

    if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
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

    self.corrupt(self.state)

    response_action = {'diaact': self.state['diaact'],
                       'inform_slots': self.state['inform_slots'],
                       'request_slots': self.state['request_slots'],
                       'turn': self.state['turn'],
                       'nl': "" }
    # add NL to dia_act
    self.add_nl_to_action(response_action)
    return response_action, self.episode_over, self.dialog_status


  def response_confirm_answer(self, system_action):
    if len(self.state['rest_slots']) > 0:
      request_slot = random.choice(self.state['rest_slots'])

      if request_slot in self.goal['request_slots'].keys():
        self.state['diaact'] = "request"
        self.state['request_slots'][request_slot] = "UNK"
      elif request_slot in self.goal['inform_slots'].keys():
        self.state['diaact'] = "inform"
        self.state['inform_slots'][request_slot] = self.goal['inform_slots'][request_slot]
        if request_slot in self.state['rest_slots']:
          self.state['rest_slots'].remove(request_slot)
    else:
      self.state['diaact'] = "thanks"

  def response_thanks(self, system_action):
    self.episode_over = True
    self.dialog_status = dialog_config.SUCCESS_DIALOG

    request_slot_set = copy.deepcopy(self.state['request_slots'].keys())
    if 'ticket' in request_slot_set:
      request_slot_set.remove('ticket')
    rest_slot_set = copy.deepcopy(self.state['rest_slots'])
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
        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
        if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
        self.state['request_slots'].clear()
      elif slot in self.goal['request_slots'].keys() and slot not in self.state['rest_slots'] and slot in self.state['history_slots'].keys(): # the requested slot has been answered
        self.state['inform_slots'][slot] = self.state['history_slots'][slot]
        self.state['request_slots'].clear()
        self.state['diaact'] = "inform"
      elif slot in self.goal['request_slots'].keys() and slot in self.state['rest_slots']: # request slot in user's goal's request slots, and not answered yet
        self.state['diaact'] = "request" # "confirm_question"
        self.state['request_slots'][slot] = "UNK"

        ########################################################################
        # Inform the rest of informable slots
        ########################################################################
        for info_slot in self.state['rest_slots']:
          if info_slot in self.goal['inform_slots'].keys():
            self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]

        for info_slot in self.state['inform_slots'].keys():
          if info_slot in self.state['rest_slots']:
            self.state['rest_slots'].remove(info_slot)
      else:
        if len(self.state['request_slots']) == 0 and len(self.state['rest_slots']) == 0:
          self.state['diaact'] = "thanks"
        else:
          self.state['diaact'] = "inform"
        self.state['inform_slots'][slot] = dialog_config.I_DO_NOT_CARE
    else: # this case should not appear
      if len(self.state['rest_slots']) > 0:
        random_slot = random.choice(self.state['rest_slots'])
        if random_slot in self.goal['inform_slots'].keys():
          self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
          self.state['rest_slots'].remove(random_slot)
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
    if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
    if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]

  def response_inform(self, system_action):
    """ Response for Inform (System Action) """

    if 'taskcomplete' in system_action['inform_slots'].keys(): # check all the constraints from agents with user goal
      self.state['diaact'] = "thanks"
      #if 'ticket' in self.state['rest_slots']: self.state['request_slots']['ticket'] = 'UNK'
      self.constraint_check = dialog_config.CONSTRAINT_CHECK_SUCCESS

      if system_action['inform_slots']['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
        self.state['history_slots']['ticket'] = dialog_config.NO_VALUE_MATCH
        if 'ticket' in self.state['rest_slots']: self.state['rest_slots'].remove('ticket')
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
            if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)

            if len(self.state['request_slots']) > 0:
              self.state['diaact'] = "request"
            elif len(self.state['rest_slots']) > 0:
              rest_slot_set = copy.deepcopy(self.state['rest_slots'])
              if 'ticket' in rest_slot_set:
                rest_slot_set.remove('ticket')

              if len(rest_slot_set) > 0:
                inform_slot = random.choice(rest_slot_set) # self.state['rest_slots']
                if inform_slot in self.goal['inform_slots'].keys():
                  self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                  self.state['diaact'] = "inform"
                  self.state['rest_slots'].remove(inform_slot)
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
            if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
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
            self.state['diaact'] = "request"
          elif len(self.state['rest_slots']) > 0:
            rest_slot_set = copy.deepcopy(self.state['rest_slots'])
            if 'ticket' in rest_slot_set:
              rest_slot_set.remove('ticket')

            if len(rest_slot_set) > 0:
              inform_slot = random.choice(rest_slot_set) #self.state['rest_slots']
              if inform_slot in self.goal['inform_slots'].keys():
                self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                self.state['diaact'] = "inform"
                self.state['rest_slots'].remove(inform_slot)

                if 'ticket' in self.state['rest_slots']:
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


  def set_text_generator(self, text_generator):
    self.text_generator = text_generator
  def generate(self, user_action):
    natural_language_response = self.text_generator.act_2_nl(user_action, "usr")
    user_action['nl'] = natural_language_response

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


class RealUser(object):
  def __init__(self):
    print("figure out how to add command line user here")