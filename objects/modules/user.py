import pdb, sys
import random, copy
from datasets.ddq import constants as dialog_config

class BaseUser(object):
  def __init__(self, args, ontology, kind="movie"):
    self.max_turn = args.max_turn
    self.num_episodes = args.epochs
    self.debug = args.debug
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
      'history_slots': {},  # slots that have already been processed (post)
      'inform_slots': {},
      'request_slots': {},
      'remaining_slots': [],   # slots that have yet to be processed (pre)
      'turn_count': 0
    }
    # movie -> ticket, taxi -> ride, restaurant -> reservation
    self.item_in_use = False

    self.goal = self._sample_goal()
    self.override_with_fake_goal()
    self.goal['request_slots']['ticket'] = 'UNK'
    self.episode_over = False
    self.user_action = self._sample_action()

    self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE
    self.dialogue_status = dialog_config.NO_OUTCOME_YET

  def _sample_action(self):
    """ randomly sample a start action based on user goal """

    # dialogue_act = random.choice(list(dialog_config.start_dia_acts.keys())
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
    self.dialog_status = dialog_config.NO_OUTCOME_YET

    dialogue_act = agent_action['dialogue_act']
    informs = agent_action['inform_slots']
    requests = agent_action['request_slots']

    if (self.max_turn > 0 and self.state['turn_count'] > self.max_turn):
      self.dialog_status = dialog_config.FAILED_DIALOG
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
    if informs['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
      self.state['history_slots']['ticket'] = dialog_config.NO_VALUE_MATCH
      self.clear_option(slot='ticket')

    # Assume, but verify that all constraints are met
    self.constraint_check = dialog_config.CONSTRAINT_CHECK_SUCCESS
    for slot in self.goal['inform_slots'].keys():
      #  Deny, if the answers from agent can not meet the constraints of user
      missing_slot = slot not in informs.keys()
      target_slot = self.goal['inform_slots'][slot].lower()
      predicted_slot = informs[slot].lower()

      if missing_slot or (target_slot != predicted_slot):
        self.state['dialogue_act'] = "deny"
        self.state['request_slots'].clear()
        self.state['inform_slots'].clear()
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE
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
        self.state['inform_slots'][slot] = dialog_config.I_DO_NOT_CARE
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
    self.dialog_status = dialog_config.SUCCESS_DIALOG

    request_slot_set = copy.deepcopy(list(self.state['request_slots'].keys()))
    if 'ticket' in request_slot_set:
      request_slot_set.remove('ticket')
    remaining_slot_set = copy.deepcopy(self.state['remaining_slots'])
    if 'ticket' in remaining_slot_set:
      remaining_slot_set.remove('ticket')

    # if there are unmet requests or constraints
    if len(request_slot_set) > 0 or len(remaining_slot_set) > 0:
      self.dialog_status = dialog_config.FAILED_DIALOG
    for hist_slot, hist_value in self.state['history_slots'].items():
      # if we failed to find a value from the user goal
      failed_to_find_match = (hist_value == dialog_config.NO_VALUE_MATCH)
      # if we found the wrong value
      goals = self.goal['inform_slots'].keys()
      goal = self.goal['inform_slots']
      found_wrong_match = (hist_slot in goals) and (hist_value != goal[hist_slot])

      if failed_to_find_match or found_wrong_match:
        self.dialog_status = dialog_config.FAILED_DIALOG

    if 'ticket' in informs.keys():
      if informs['ticket'] == dialog_config.NO_VALUE_MATCH:
        self.dialog_status = dialog_config.FAILED_DIALOG
    if self.constraint_check == dialog_config.CONSTRAINT_CHECK_FAILURE:
      self.dialog_status = dialog_config.FAILED_DIALOG

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

  def override_with_fake_goal(self):
      """ Build a fake goal mannual for debugging purposes """
      # self.goal['inform_slots'].clear()
      # self.goal['inform_slots']['city'] = 'seattle'
      # self.goal['inform_slots']['numberofpeople'] = '14'
      # self.goal['inform_slots']['theater'] = 'amc pacific place 11 theater'
      # self.goal['inform_slots']['date'] = 'March 16th'
      if "date" in self.goal['inform_slots'].keys():
        if self.goal['inform_slots']['date'] == 'tomorrow night':
          self.goal['inform_slots']['date'] = 'tomorrow'
      if "other" in self.goal['inform_slots'].keys():
        del self.goal['inform_slots']['other']
      # self.goal['inform_slots']['moviename'] = 'zootopia'
      # self.goal['inform_slots']['distanceconstraints'] = 'close to 95833'
      # self.goal['request_slots'].clear()
      # self.goal['inform_slots']['starttime'] = 'UNK'
      # self.goal['request_slots']['city'] = 'UNK'
      # self.goal['request_slots']['theater'] = 'UNK'


class CommandLineUser(BaseUser):
  def __init__(self, args, ontology, kind="movie"):
    super().__init__(args, ontology, kind)
    self.learning_phase = "train"
    self.agent_input_mode =  "dialogue_act" # "raw_text"
    self.do_print = False

  def initialize_episode(self):
    self.goal = self._sample_goal()
    self.turn_count = -2
    self.user_action = {'dialogue_act':'UNK', 'inform_slots':{}, 'request_slots':{}}

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
    cleaned = command_line_input.strip(' ').strip('\n').strip('\r')
    intents = cleaned.lower().split(',')
    for intent in intents:
      idx = intent.find('(')
      act = intent[0:idx]
      slot, value = intent[idx+1:-1].split("=") # -1 is to skip the closing ')'

      self.error_checking(idx, act, slot, value)
      self.user_action["dialogue_act"] = act
      self.user_action["nl"] = "N/A"
      self.user_action["{}_slots".format(act)][slot] = value

  def error_checking(self, idx, act, slot, value):
    if idx < 0:
      raise(ValueError("input is not properly formatted as a user intent"))
    if act not in self.act_set:
      raise(ValueError("{} is not part of allowable dialogue act set".format(act)))
    if slot not in self.slot_set:
      raise(ValueError("{} is not part of allowable slot set".format(slot)))
    if value not in self.value_set[slot]:
      raise(ValueError("{} is not part of the available value set".format(value)))