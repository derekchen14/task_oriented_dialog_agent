"""
Created on May 20, 2016, Updated Feb 5, 2019

state tracker

@author: xiul, t-zalipt, derekchen14
"""

from objects.modules.kb_operator import KBHelper
import numpy as np
import copy


class DialogueState:
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

  def __init__(self, knowledge_base, ontology):
    """ constructor for statetracker takes movie knowledge base and initializes a new episode

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
    """
    self.act_set = ontology.acts
    self.slot_set = ontology.slots
    self.relation_set = ontology.relations
    self.value_set = ontology.values

    self.initialize_episode()
    self.history_vectors = None
    self.history_dictionaries = None
    self.current_slots = None
    self.action_dimension = 10      # TODO REPLACE WITH REAL VALUE
    self.kb_result_dimension = 10   # TODO  REPLACE WITH REAL VALUE
    self.turn_count = 0
    self.kb_helper = KBHelper(knowledge_base)


  def initialize_episode(self):
    """ Initialize a new episode (dialog), flush the current state and tracked slots """
    self.action_dimension = 10
    self.history_vectors = np.zeros((1, self.action_dimension))
    self.history_dictionaries = []
    self.turn_count = 0
    self.current_slots = {}

    self.current_slots['inform_slots'] = {}
    self.current_slots['request_slots'] = {}
    self.current_slots['proposed_slots'] = {}
    self.current_slots['agent_request_slots'] = {}

  def dialog_history_vectors(self):
    """ Return the dialog history (both user and agent actions) in vector representation """
    return self.history_vectors
  def dialog_history_dictionaries(self):
    """  Return the dictionary representation of the dialog history (includes values) """
    return self.history_dictionaries

  def kb_results_for_state(self):
    """ Return the information about the database results based on the currently informed slots """
    ########################################################################
    # TODO Calculate results based on current informed slots
    ########################################################################
    kb_results = self.kb_helper.database_results_for_agent(self.current_slots) # replace this with something less ridiculous
    # TODO turn results into vector (from dictionary)
    results = np.zeros((0, self.kb_result_dimension))
    return results

  def get_state_for_agent(self):
    """ Get the state representatons to send to agent """
    #state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, 'kb_results': self.kb_results_for_state()}
    state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, #'kb_results': self.kb_results_for_state(),
         'kb_results_dict':self.kb_helper.database_results_for_agent(self.current_slots), 'turn_count': self.turn_count, 'history': self.history_dictionaries,
         'agent_action': self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None}
    return copy.deepcopy(state)

  def get_suggest_slots_values(self, request_slots):
    """ Get the suggested values for request slots """

    suggest_slot_vals = {}
    if len(request_slots) > 0:
      suggest_slot_vals = self.kb_helper.suggest_slot_values(request_slots, self.current_slots)

    return suggest_slot_vals

  def get_current_kb_results(self):
    """ get the kb_results for current state """
    kb_results = self.kb_helper.available_results_from_kb(self.current_slots)
    return kb_results


  def update(self, agent_action=None, user_action=None):
    #  Ensure that one and only one action is being taken
    assert(not (user_action and agent_action))
    assert(user_action or agent_action)

    ########################################################################
    #   Update state to reflect a new action by the agent.
    ########################################################################
    if agent_action:
      #  Handles the act_slot response (with values needing to be filled)
      if agent_action['slot_action']:
        response = copy.deepcopy(agent_action['slot_action'])

        inform_slots = self.kb_helper.fill_inform_slots(response['inform_slots'], self.current_slots)
        agent_action_values = {'speaker': "agent",
                                'diaact': response['diaact'],
                                'inform_slots': inform_slots,
                                'request_slots':response['request_slots'],
                                'turn_count': self.turn_count }

        agent_action['slot_action'].update({
                                'diaact': response['diaact'],
                                'inform_slots': inform_slots,
                                'request_slots':response['request_slots'],
                                'turn_count':self.turn_count })

      elif agent_action['slot_value_action']:
        agent_action_values = copy.deepcopy(agent_action['slot_value_action'])
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

    ########################################################################
    #   Update the state to reflect a new action by the user
    ########################################################################
    elif user_action:
      #   Update the current slots
      for slot in user_action['inform_slots'].keys():
        self.current_slots['inform_slots'][slot] = user_action['inform_slots'][slot]
        if slot in self.current_slots['request_slots'].keys():
          del self.current_slots['request_slots'][slot]

      for slot in user_action['request_slots'].keys():
        if slot not in self.current_slots['request_slots']:
          self.current_slots['request_slots'][slot] = "UNK"

      self.history_vectors = np.vstack([self.history_vectors, np.zeros((1,self.action_dimension))])
      new_move = {'turn_count': self.turn_count, 'speaker': "user", 'request_slots': user_action['request_slots'], 'inform_slots': user_action['inform_slots'], 'diaact': user_action['diaact']}
      self.history_dictionaries.append(copy.deepcopy(new_move))

    ########################################################################
    #  Should update whether user or agent took the action
    ########################################################################
    self.turn_count += 1