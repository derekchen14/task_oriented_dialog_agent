"""
Created on May 20, 2016, Updated April 5, 2019
state tracker
@author: xiul, t-zalipt, derekchen14
"""
from objects.modules.kb_operator import KBHelper
import numpy as np
import copy

class DialogState:
  """ Tracks local dialogue state across turns.  (see also: RewardMonitor)
    Main responsibility is to maintain a record six (6) items:
      1a) user intent - a one-hot vector with size equal to number of
        possible slot-values, where ones are placed on the argmax value
        of each slot.  Note all slots have a "not yet mentioned" value.
      1b) user belief - a vector with size equal to the number of slot-values,
        where each dimension is the uncertainty level / confidence score
        associated with each possible slot value
      It is assumed domain is given, dialogue act is in ["inform", "request"],
        and relation is "=", so these are not covered in the intent

      2) context - vector embedding of previous user utterancees
      3) history - a sequence of previous agent actions
      4) turn count - cumulative episode turn count as a scalar
      5) kb results - number of KB results matching all user constraints
      6) semantic frame - a memory vector containing the aggregation
        of all prior and current user beliefs.  This is modeled with EntNet.
  """
  def __init__(self, ontology, movie_kb):
    self.movie_kb = movie_kb
    self.initialize_episode()
    self.history_vectors = None
    self.history_dictionaries = None
    self.current_slots = None
    self.action_dimension = 10      # TODO REPLACE WITH REAL VALUE
    self.kb_result_dimension = 10   # TODO  REPLACE WITH REAL VALUE
    self.turn_count = 0
    self.kb_helper = KBHelper(movie_kb)

    self.act_set = ontology.acts
    self.slot_set = ontology.slots
    self.relation_set = ontology.relations
    self.value_set = ontology.values

  def initialize_episode(self):
    """ Initialize a new episode (dialog),
    flush the current state and tracked slots """
    self.action_dimension = 10
    self.history_vectors = np.zeros((1, self.action_dimension))
    self.history_dictionaries = []
    self.turn_count = 0

    self.current_slots = {}
    self.current_slots['inform_slots'] = {}
    self.current_slots['request_slots'] = {}
    self.current_slots['act_slots'] = {}
    self.current_slots['proposed_slots'] = {}
    self.current_slots['agent_request_slots'] = {}

  def get_state(self, actor):
    # Get the state representatons of the actor, either "agent" or "user"
    state = {'current_slots': self.current_slots,
       'kb_results_dict':self.kb_helper.database_results(self.current_slots),
       'turn_count': self.turn_count, 'history': self.history_dictionaries }

    other = 'user' if actor == 'agent' else 'agent'
    state[f'{other}_action'] = self.history_dictionaries[-1]
    lhd = len(self.history_dictionaries)
    state[f'{actor}_action'] = self.history_dictionaries[-2] if lhd > 1 else None

    return copy.deepcopy(state)

  def make_suggestion(self, request_slots):
    if len(request_slots) > 0:
      return self.kb_helper.suggest(request_slots, self.current_slots)
    else:
      return {}

  def update_agent_state(self, action):
    if action['slot_action']:
      response = copy.deepcopy(action['slot_action'])

      inform_slots = self.kb_helper.fill_inform_slots(
                              response['inform_slots'], self.current_slots)
      agent_action_values = {'speaker': "agent",
                              'dialogue_act': response['dialogue_act'],
                              'inform_slots': inform_slots,
                              'request_slots':response['request_slots'],
                              'turn_count': self.turn_count }

      action['slot_action'].update({
                              'dialogue_act': response['dialogue_act'],
                              'inform_slots': inform_slots,
                              'request_slots':response['request_slots'],
                              'turn_count':self.turn_count })

    elif action['slot_value_action']:
      agent_action_values = copy.deepcopy(action['slot_value_action'])
      agent_action_values['turn_count'] = self.turn_count
      agent_action_values['speaker'] = "agent"

    #   Should execute regardless of which kind of agent produced action
    for slot in agent_action_values['inform_slots'].keys():
      self.current_slots['proposed_slots'][slot] = agent_action_values['inform_slots'][slot]
      self.current_slots['inform_slots'][slot] = agent_action_values['inform_slots'][slot] # add into inform_slots
      if slot in self.current_slots['request_slots'].keys():
        del self.current_slots['request_slots'][slot]

    for slot in agent_action_values['request_slots'].keys():
      if slot not in self.current_slots['agent_request_slots']:
        self.current_slots['agent_request_slots'][slot] = "UNK"

    self.history_dictionaries.append(agent_action_values)
    current_agent_vector = np.ones((1, self.action_dimension))
    self.history_vectors = np.vstack([self.history_vectors, current_agent_vector])
    self.turn_count += 1

  def update_user_state(self, intent):
    #   Update the state to reflect the newly predicted user intent
    for slot in intent['inform_slots'].keys():
      self.current_slots['inform_slots'][slot] = intent['inform_slots'][slot]
      if slot in self.current_slots['request_slots'].keys():
        del self.current_slots['request_slots'][slot]

    for slot in intent['request_slots'].keys():
      if slot not in self.current_slots['request_slots']:
        self.current_slots['request_slots'][slot] = "UNK"

    self.history_vectors = np.vstack([self.history_vectors, np.zeros((1,self.action_dimension))])
    new_move = {'turn_count': self.turn_count, 'speaker': "user",
              'request_slots': intent['request_slots'],
              'inform_slots': intent['inform_slots'],
              'dialogue_act': intent['dialogue_act']}
    self.history_dictionaries.append(copy.deepcopy(new_move))
    self.turn_count += 1


  """
  constructor for statetracker takes movie knowledge base and initializes a new episode

  Arguments:
  act_set                 --  The set of all acts availavle
  slot_set                --  The total set of available slots
  value_set               --  A representation of all the available movies.
          Generally this object is accessed via the KBHelper class
  Class Variables:
  history_vectors         --  A record of the current dialog so far in vector format (act-slot, but no values)
  history_dictionaries    --  A record of the current dialog in dictionary format
  current_slots           --  A dictionary that keeps a running record of which slots are filled current_slots['inform_slots'] and which are requested current_slots['request_slots'] (but not filed)
  action_dimension        --  # TODO indicates the dimensionality of the vector representaiton of the action
  kb_result_dimension     --  A single integer denoting the dimension of the kb_results features.
  turn_count              --  A running count of which turn we are at in the present dialog


  def kb_results_for_state(self):
    Return the information about the database results based on the currently informed slots
    TODO Calculate results based on current informed slots
    replace this with something less ridiculous
    kb_results = self.kb_helper.database_results(self.current_slots)
    TODO turn results into vector (from dictionary)
    return np.zeros((0, self.kb_result_dimension))

  def get_current_kb_results(self):
    return self.kb_helper.available_results_from_kb(self.current_slots)
  """
