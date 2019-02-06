import logging
import torch

from objects.modules.dialogue_state import StateTracker
from objects.modules.user import UserSimulator, CommandLineUser
import datasets.ddq.constants as dialog_config

class BasePolicyManager(object):
  def __init__(self, args, model, kb, ontology):
    self.verbose = args.verbose
    self.max_turn = args.max_turn
    self.batch_size = args.batch_size

    self.state_tracker = StateTracker(kb, ontology)
    self.agent = model
    self.user = CommandLineUser(args, ontology) if args.user == "command" else UserSimulator(args, ontology)

  def action_to_nl(self, agent_action):
    """ Add natural language capabilities (NL) to Agent Dialogue Act """
    if agent_action['slot_action']:
      agent_action['slot_action']['nl'] = ""
      # self.nlg_model.translate_diaact(agent_action['slot_action']) # NLG
      # user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(
      #                               agent_action['slot_action'], 'agt')
      chosen_action = agent_action['slot_action']
      user_response = self.user.text_generator.generate(chosen_action, 'agt')
      agent_action['slot_action']['nl'] = user_response
    elif agent_action['slot_value_action']:
      agent_action['slot_value_action']['nl'] = ""
      # self.nlg_model.translate_diaact(agent_action['slot_value_action'])
      chosen_action = agent_action['slot_value_action']
      user_response = self.user.text_generator.generate(chosen_action, 'agt')
      agent_action['slot_action']['nl'] = user_response

  def store_experience(self, current_state, action, reward, next_state, episode_over):
    """  Register feedback (s,a,r,s') from the environment,
    to be stored in experience replay buffer as future training data

    Arguments:
    current_state    --  The state in which the last action was taken
    current_action   --  The previous agent action
    reward           --  The reward received immediately following the action
    next_state       --  The state transition following the latest action
    episode_over     --  Boolean value representing whether this is final action.

    Returns: None

    The rulebased agent will keep as identity function because
      it does not need to store experiences for future training
    """
    pass

  def initialize_episode(self, sim=False):
    self.simulation_mode = sim
    self.episode_reward = 0
    self.episode_over = False

    self.state_tracker.initialize_episode()
    self.user.initialize_episode()
    self.agent.initialize_episode()

    ua = self.user.user_action.copy()
    self.state_tracker.update(user_action=ua)
    if self.verbose:
      print("New episode, user goal:")
      print(self.user.goal)
      self.print_function(user_action=self.user_action)

  def print_function(self, agent_action=None, user_action=None):
    if agent_action:
      if dialog_config.run_mode == 0:
        if self.agent.__class__.__name__ != 'AgentCmd':
          print ("Turn %d sys: %s" % (agent_action['turn_count'], agent_action['nl']))
      elif dialog_config.run_mode == 1:
        if self.agent.__class__.__name__ != 'AgentCmd':
          print ("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn_count'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
      elif dialog_config.run_mode == 2: # debug mode
        print ("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn_count'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
        print ("Turn %d sys: %s" % (agent_action['turn_count'], agent_action['nl']))

      if dialog_config.auto_suggest == 1:
        print('(Suggested Values: %s)' % (self.state.get_suggest_slots_values(agent_action['request_slots'])))

    elif user_action:
      if dialog_config.run_mode == 0:
        print ("Turn %d usr: %s" % (user_action['turn_count'], user_action['nl']))
      elif dialog_config.run_mode == 1:
        print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn_count'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
      elif dialog_config.run_mode == 2: # debug mode, show both
        print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn_count'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
        print ("Turn %d usr: %s" % (user_action['turn_count'], user_action['nl']))

      if self.agent.__class__.__name__ == 'AgentCmd': # command line agent
        user_request_slots = user_action['request_slots']
        if 'ticket'in user_request_slots.keys(): del user_request_slots['ticket']

        if 'reservation' in user_request_slots.keys(): del user_request_slots['reservation']
        if 'taxi' in user_request_slots.keys(): del user_request_slots['taxi']

        if len(user_request_slots) > 0:
          possible_values = self.state.get_suggest_slots_values(user_action['request_slots'])
          for slot in possible_values.keys():
            if len(possible_values[slot]) > 0:
              print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
            elif len(possible_values[slot]) == 0:
              print('(Suggested Values: there is no available %s)' % (slot))
        else:
          pass
          #kb_results = self.state.get_current_kb_results()
          #print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))


  def next(self, collect_data):
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
    self.state = self.state_tracker.get_state_for_agent()
    self.agent_action = self.agent.state_to_action(self.state)
    #   Register AGENT action within the state_tracker
    self.state_tracker.update(agent_action=self.agent_action)
    self.action_to_nl(self.agent_action) # add NL to BasePolicy Dia_Act
    if self.verbose:
      self.print_function(agent_action = self.agent_action['slot_action'])

    #   CALL USER TO TAKE HER TURN
    self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
    self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
    self.reward = self.reward_function(dialog_status)
    #   Update state tracker with latest user action
    if self.episode_over != True:
      self.state_tracker.update(user_action=self.user_action)
      # self.print_function(user_action=self.user_action)

    #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
    if collect_data:
      self.store_experience(self.state, self.agent_action, self.reward,
              self.state_tracker.get_state_for_agent(), self.episode_over)

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


class RulebasedPolicyManager(BasePolicyManager):

  def save_checkpoint(self, monitor, episode):
    print("New best model found!")
    print("Episode {} -- Success_rate: {:.4f}, Average Reward: {:.4f}, \
          Average Turns: {:.4f}".format(episode, monitor.success_rate,
          monitor.avg_reward, monitor.avg_turn))

class NeuralPolicyManager(BasePolicyManager):
  def __init__(self, args, model, ontology):
    super().__init__(args, model, ontology)
    self.hidden_dim = args.hidden_dim
    self.gamma = args.discount_rate
    self.warm_start = args.warm_start
    self.max_pool_size = args.pool_size
    self.epsilon = args.epsilon

  def save_checkpoint(self, monitor, episode):
    filename = '{}/{}.pt'.format(self.save_dir, identifier)
    logging.info('saving model to {}.pt'.format(identifier))
    state = {
      'args': vars(self.args),
      'model': self.state_dict(),
      'summary': summary,
      'optimizer': self.optimizer.state_dict(),
    }
    torch.save(state, filename)

  def store_experience(self, current_state, action, reward, next_state, episode_over):
    current_rep = self.prepare_state_representation(current_state)
    next_rep = self.prepare_state_representation(next_state)
    training_example = (current_rep, action, reward, next_rep, episode_over)

    if (len(self.experience_replay_pool) == self.max_pool_size):
      chosen_idx = random.randint(0,self.max_pool_size)
      self.experience_replay_pool[chosen_idx] = training_example
    else:
      self.experience_replay_pool.append(training_example)
