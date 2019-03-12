import pdb, sys
import random
import copy
import torch
import json
import numpy as np
from collections import deque

from objects.modules.dialogue_state import DialogueState
from objects.modules.user import UserSimulator, CommandLineUser
from objects.blocks.base import BasePolicyManager
from objects.models.ddq import DQN, Transition
from utils.external import dialog_config

import torch.nn.functional as F
from torch import optim

class RulePolicyManager(BasePolicyManager):

  def save_checkpoint(self, monitor, episode):
    print("New best model found!")
    print("Episode {} -- Success_rate: {:.4f}, Average Reward: {:.4f}, \
          Average Turns: {:.4f}".format(episode, monitor.success_rate,
          monitor.avg_reward, monitor.avg_turn))

class NeuralPolicyManager(BasePolicyManager):
  def __init__(self, args, model, device, planner, movie_dict=None, act_set=None, slot_set=None):
    super().__init__(args, model)
    self.movie_dict = movie_dict
    self.act_set = act_set
    self.slot_set = slot_set
    self.act_cardinality = len(act_set.keys())
    self.slot_cardinality = len(slot_set.keys())

    self.feasible_actions = dialog_config.feasible_actions
    self.num_actions = len(self.feasible_actions)

    self.epsilon = args.epsilon
    self.agent_run_mode = 0 # params['agent_run_mode']
    self.agent_act_level = 0 # params['agent_act_level']

    self.experience_replay_pool_size = args.pool_size
    self.experience_replay_pool = deque(
      maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
    self.experience_replay_pool_from_model = deque(
      maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
    self.running_expereince_pool = None # hold experience from both user and world model

    self.hidden_size = args.hidden_dim
    self.gamma = 0.9
    self.predict_mode = False
    self.warm_start = args.warm_start
    self.cur_bellman_err = 0
    self.max_turn = args.max_turn

    self.dqn = model
    self.user_planning = planner
    sizes = self.dqn.input_size, self.dqn.hidden_size, self.dqn.output_size

    self.target_dqn = DQN(*sizes).to(device)
    self.target_dqn.load_state_dict(self.dqn.state_dict())
    self.target_dqn.eval()

    self.opt = args.optimizer
    self.lr = args.learning_rate
    self.reg = args.weight_decay
    self.init_optimizer(self.dqn.parameters())

    # Prediction Mode: load trained DQN model
    # if params['trained_model_path'] != None:
    #     self.load(params['trained_model_path'])
    #     self.predict_mode = True
    #     self.warm_start = 2

  def initialize_episode(self):
    """ Initialize a new episode. This function is called every time a new episode is run. """
    self.current_slot_id = 0
    self.phase = 0
    self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

  def state_to_action(self, state):
    """ DQN: Input state, output action """
    # self.state['turn_count'] += 2
    self.representation = self.prepare_state_representation(state)
    self.action = self.run_policy(self.representation)
    if self.warm_start == 1:
      act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
    else:
      act_slot_response = copy.deepcopy(self.feasible_actions[self.action[0]])

    return {'slot_action': act_slot_response, 'slot_value_action': None}

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
    user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

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
      agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

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

    # turn_rep = np.zeros((1,1)) + state['turn_count'] / 10.
    turn_rep = np.zeros((1, 1))

    ########################################################################
    #  One-hot representation of the turn count?
    ########################################################################
    turn_onehot_rep = np.zeros((1, self.max_turn + 5))
    turn_onehot_rep[0, state['turn_count']] = 1.0
    kb_count_rep = np.zeros((1, self.slot_cardinality + 1))

    ########################################################################
    #   Representation of KB results (binary)
    ########################################################################
    kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

    self.final_representation = np.hstack(
      [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
       agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
    return self.final_representation

  def run_policy(self, representation):
    """ epsilon-greedy policy """

    if random.random() < self.epsilon:
      return random.randint(0, self.num_actions - 1)
    else:
      if self.warm_start == 1:
        if len(self.experience_replay_pool) > self.experience_replay_pool_size:
          self.warm_start = 2
        return self.rule_policy()
      else:
        return self.DQN_policy(representation)

  def rule_policy(self):
    """ Rule Policy """

    act_slot_response = {}

    if self.current_slot_id < len(self.request_set):
      slot = self.request_set[self.current_slot_id]
      self.current_slot_id += 1

      act_slot_response = {}
      act_slot_response['diaact'] = "request"
      act_slot_response['inform_slots'] = {}
      act_slot_response['request_slots'] = {slot: "UNK"}
    elif self.phase == 0:
      act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                 'request_slots': {}}
      self.phase += 1
    elif self.phase == 1:
      act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

    return self.action_index(act_slot_response)

  def DQN_policy(self, state_representation):
    """ Return action from DQN"""

    with torch.no_grad():
      action = self.dqn.predict(torch.FloatTensor(state_representation))
    return action

  def action_index(self, act_slot_response):
    """ Return the index of action """

    for (i, action) in enumerate(self.feasible_actions):
      if act_slot_response == action:
        return i
    print(act_slot_response)
    raise(Exception("action index not found"))
    return None

  def store_experience(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
    """ Register feedback from either environment or world model, to be stored as future training data """

    state_t_rep = self.prepare_state_representation(s_t)
    action_t = self.action
    reward_t = reward
    state_tplus1_rep = self.prepare_state_representation(s_tplus1)
    st_user = self.prepare_state_representation(s_tplus1)
    training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

    if self.predict_mode == False:  # Training Mode
      if self.warm_start == 1:
        self.experience_replay_pool.append(training_example)
    else:  # Prediction Mode
      if not from_model:
        self.experience_replay_pool.append(training_example)
      else:
        self.experience_replay_pool_from_model.append(training_example)

  def sample_from_buffer(self, batch_size):
    """Sample batch size examples from experience buffer and convert it to torch readable format"""
    # type: (int, ) -> Transition

    batch = [random.choice(self.running_expereince_pool) for i in range(batch_size)]
    np_batch = []
    for x in range(len(Transition._fields)):
      v = []
      for i in range(batch_size):
        v.append(batch[i][x])
      np_batch.append(np.vstack(v))

    return Transition(*np_batch)

  def train(self, batch_size=1, num_batches=100, verbose=False):
    """ Train DQN with experience buffer that comes from both user and world model interaction."""

    self.cur_bellman_err = 0.
    self.cur_bellman_err_planning = 0.
    self.running_expereince_pool = list(self.experience_replay_pool) + list(self.experience_replay_pool_from_model)

    for iter_batch in range(num_batches):
      for iter in range(round(len(self.running_expereince_pool) / (batch_size))):
        self.optimizer.zero_grad()
        batch = self.sample_from_buffer(batch_size)

        state_value = self.dqn(torch.FloatTensor(batch.state)).gather(1, torch.tensor(batch.action))
        next_state_value, _ = self.target_dqn(torch.FloatTensor(batch.next_state)).max(1)
        next_state_value = next_state_value.unsqueeze(1)
        term = np.asarray(batch.term, dtype=np.float32)
        expected_value = torch.FloatTensor(batch.reward) + self.gamma * next_state_value * (
          1 - torch.FloatTensor(term))

        loss = F.mse_loss(state_value, expected_value)
        loss.backward()
        self.optimizer.step()
        self.cur_bellman_err += loss.item()

    if verbose and len(self.experience_replay_pool) != 0:
      print("cur bellman error %.2f, experience replay pool %s, model replay pool %s, error for planning %.2f" % (
          float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
          len(self.experience_replay_pool), len(self.experience_replay_pool_from_model),
          self.cur_bellman_err_planning))

  def reward_function(self, dialog_status):
    # Reward Function 1: a reward function based on the dialog_status
    if dialog_status == dialog_config.FAILED_DIALOG:
      reward = -self.max_turn                     # -40
    elif dialog_status == dialog_config.SUCCESS_DIALOG:
      reward = 2 * self.max_turn                  # +80
    else:  # for per turn
      reward = -1                                 # -20 over time
    return reward

  ################################################################################
  #    Debug Functions
  ################################################################################
  def save_experience_replay_to_file(self, path):
    """ Save the experience replay pool to a file """

    try:
      pickle.dump(self.experience_replay_pool, open(path, "wb"))
      print('saved model in %s' % (path,))
    except Exception as e:
      print('Error: Writing model fails: %s' % (path,))
      print(e)

  def load_experience_replay_from_file(self, path):
    """ Load the experience replay pool from a file"""

    self.experience_replay_pool = pickle.load(open(path, 'rb'), encoding='latin1')

  def load_trained_DQN(self, path):
    """ Load the trained DQN from a file """
    trained_file = pickle.load(open(path, 'rb'), encoding='latin1')
    model = trained_file['model']
    print("Trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
    return model

  def save(self, filename):
    torch.save(self.dqn.state_dict(), filename)

  def load(self, filename):
    self.dqn.load_state_dict(torch.load(filename))

  def reset_dqn_target(self):
    self.target_dqn.load_state_dict(self.dqn.state_dict())


'''
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
    print('saving model to {}.pt'.format(identifier))
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
'''

class BasePolicyManagerLater(object):
  def __init__(self, args, model, kb, ontology):
    self.state = DialogueState(kb, ontology)
    self.user = CommandLineUser(args, ontology) if args.user == "command" else UserSimulator(args, ontology)

  def print_function(self, action_dict, kind):
    if not self.verbose: return
    if self.debug:
      for k, v in action_dict.items(): print(kind, k, v)
    else:
      print ("{}) {}: {}".format(action_dict['turn_count'], kind, action_dict['nl']))
    if dialog_config.auto_suggest == 1:
      print('(Suggested Values: %s)' % (self.agent_state.get_suggest_slots_values(agent_action['request_slots'])))

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
    self.agent_state = self.state.get_state_for_agent()
    self.agent_action = self.model.state_to_action(self.agent_state)
    #   Register AGENT action within the state
    self.state.update(agent_action=self.agent_action)
    self.action_to_nl(self.agent_action) # add NL to BasePolicy Dia_Act
    self.print_function(self.agent_action['slot_action'], "agent")

    #   CALL USER TO TAKE HER TURN
    self.agent_action = self.state.dialog_history_dictionaries()[-1]
    self.user_action, self.episode_over, dialog_status = self.user.next(self.agent_action)
    self.reward = self.reward_function(dialog_status)
    #   Update state tracker with latest user action
    if self.episode_over != True:
      self.state.update(user_action=self.user_action)
      if self.user.do_print:
        # print(self.user.state)
        # print("above is user state >>>>>>>>>>>>>>. below is user action")
        self.print_function(self.user_action, "user")

    #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
    if collect_data:
      self.store_experience(self.agent_state, self.agent_action, self.reward,
              self.state.get_state_for_agent(), self.episode_over)

    return (self.episode_over, self.reward)
