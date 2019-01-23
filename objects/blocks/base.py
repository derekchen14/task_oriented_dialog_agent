
class BaseModel:
  """ Prototype for all agent classes,
  defining the interface they must uphold """

  def __init__(self, act_set, slot_set, relation_set, value_set, params):
    """ Constructor Arguments:
    act_set        --  The set of acts.
    slot_set       --  The set of available slots
    relation_set   --  The set of relations.
    value_set      --  This is here now but doesn't belong
    """
    self.act_set = act_set
    self.slot_set = slot_set
    self.relation_set = relation_set
    self.value_set = value_set
    self.act_cardinality = len(act_set.keys())
    self.slot_cardinality = len(slot_set.keys())

    self.epsilon = params['epsilon']
    self.agent_run_mode = params['agent_run_mode']
    self.agent_act_level = params['agent_act_level']

  def initialize_episode(self):
    """ This function is called every time a new episode is run.
        Previously called "current_action" and "state", renamed since
        belief state = context + frame + kb_results + turn count
    """
    self.frame = {
      'diaact': '',
      'inform_slots': {},
      'request_slots': {},
      'turn': -1
    }
    self.current_slot_id = 0
    # self.turn_count = -1

  def predict(self, state, available_actions):
    """ Take the current state and return an action according to the current
    exploration/exploitation policy. We define the agents flexibly so that
    they can either operate on act_slot representations or act_slot_value
    representations. We also define the responses flexibly, returning a
    dictionary with keys [act_slot_response, act_slot_value_response].
    This way the command-line agent can continue to operate with values

    Arguments:
    belief_state -- A tuple of (history, kb_results) where history is a sequence
                    of previous actions and kb_results contains information on
                    the number of results matching the current constraints.
    user_intent  -- A legacy representation used to run the command line agent.
                    We should remove this ASAP but not just yet
    available    -- A list of the allowable actions in the current state

    Returns:
    action_slot_only  --  An action consisting of one act and >= 0 slots
                          as well as which slots are informed vs requested.
    agent_action      --  An action consisting of acts slots and values in
          the legacy format. This can be used in the future for training agents
          that take value into account and interact directly with the database

    for the intent tracker, this method will invoke utterance_to_state()
    for the policy manager, this method will invoke state_to_action()
    for the text generator, this method will invoke action_to_response()

    action = {"act_slot_response": act_slot_response,
              "act_slot_value_response": act_slot_value_response}
    """


    action_slot_only = None
    agent_action = None
    return {"action_slot_only": action_slot_only, "agent_action": agent_action}


  def store_experience(self, current_state, current_action, reward,
                                    next_state, episode_over):
    """  Register feedback (s,a,r,s') from the environment,
    to be stored in experience replay buffer as future training data

    Arguments:
    current_state    --  The state in which the last action was taken
    current_action   --  The previous agent action
    reward           --  The reward received immediately following the action
    next_state       --  The state transition following the latest action
    episode_over     --  Boolean value representing whether this is final action.

    Returns: None
    """
    pass

  def set_text_generator(self, text_generator):
    # self.nlg_model = nlg_model
    self.text_generator = text_generator

  def set_belief_tracker(self, belief_tracker):
    # self.nlu_model = nlu_model
    self.belief_tracker = belief_tracker

  def add_nl_to_action(self, agent_action):
    """ Add natural language capabilities (NL) to Agent Dialogue Act """
    if agent_action['act_slot_response']:
      agent_action['act_slot_response']['nl'] = ""
      # self.nlg_model.translate_diaact(agent_action['act_slot_response']) # NLG
      # user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(
      #                               agent_action['act_slot_response'], 'agt')
      chosen_action = agent_action['act_slot_response']
      user_response = self.text_generator.generate(chosen_action, 'agt')
      agent_action['act_slot_response']['nl'] = user_response
    elif agent_action['act_slot_value_response']:
      agent_action['act_slot_value_response']['nl'] = ""
      # self.nlg_model.translate_diaact(agent_action['act_slot_value_response'])
      chosen_action = agent_action['act_slot_value_response']
      user_response = self.text_generator.generate(chosen_action, 'agt')
      agent_action['act_slot_response']['nl'] = user_response