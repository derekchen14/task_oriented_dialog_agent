import pdb, sys
import random
import copy
import torch
import json
import numpy as np
from collections import deque

from objects.modules.user import UserSimulator, CommandLineUser
from objects.blocks.base import BasePolicyManager
from objects.models.ddq import DQN, Transition
from utils.external import dialog_constants

import torch.nn.functional as F
from torch import FloatTensor as tensor
from torch import optim

class RulePolicyManager(BasePolicyManager):

  def save_checkpoint(self, monitor, episode):
    print("New best model found!")
    print("Episode {} -- Success_rate: {:.4f}, Average Reward: {:.4f}, \
          Average Turns: {:.4f}".format(episode, monitor.success_rate,
          monitor.avg_reward, monitor.avg_turn))

class NeuralPolicyManager(BasePolicyManager):
  def __init__(self, args, model):
    super().__init__(args, model)
    self.epsilon = args.epsilon
    self.agent_run_mode = 0 # params['agent_run_mode']
    self.agent_act_level = 0 # params['agent_act_level']
    self.belief_state_type = args.belief

    self.experience_replay_pool_size = args.pool_size
    self.experience_replay_pool = deque(
      maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
    self.experience_replay_pool_from_model = deque(
      maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
    self.running_pool = None # hold experience from both user and world model

    self.hidden_size = args.hidden_dim
    self.gamma = 0.9
    self.predict_mode = False
    self.warm_start = args.warm_start
    self.cur_bellman_err = 0
    self.max_turn = args.max_turn
    self.use_existing = args.use_existing

  def configure_settings(self, device, world_sim, ontology, movie_dict, old_ont=None):
    self.dqn = self.model
    sizes = self.model.input_size, self.model.hidden_size, self.model.output_size
    self.target_dqn = DQN(*sizes).to('cpu')
    self.target_dqn.load_state_dict(self.dqn.state_dict())
    self.target_dqn.eval()

    self.act_set = {act: i for i, act in enumerate(ontology.acts)}
    self.slot_set = {slot: j for j, slot in enumerate(ontology.slots)}
    self.value_set = ontology.values

    if old_ont is not None:
      self.old_acts = old_ont['acts']
      self.old_slots = old_ont['slots']
    else:
      self.old_acts = self.act_set
      self.old_slots = self.slot_set
    self.act_cardinality = len(self.old_acts)
    self.slot_cardinality = len(self.old_slots)

    self.feasible_agent_actions = ontology.feasible_agent_actions
    self.num_actions = len(self.feasible_agent_actions)

    self.user_planning = world_sim
    self.movie_dict = movie_dict
    self.belief_tracker = world_sim.nlu_model
    self.text_generator = world_sim.nlg_model

    # this should be move into Learner, but not yet because NLU and NLG are messy
    self.init_optimizer(self.dqn.parameters())
    if self.use_existing:
      self.optimizer.load_state_dict(model.existing_checkpoint['optimizer'])

  def initialize_episode(self):
    """ Initialize a new episode. This function is called every time a new episode is run. """
    self.current_slot_id = 0
    self.phase = 0
    self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

  def state_to_action(self, state):
    """ DQN: Input state, output action
      Get argmax prediction from the model, in a no_grad mode
    """
    # self.state['turn_count'] += 2
    self.representation = self.prepare_state_representation(state)
    action_id = self.run_policy(self.representation)

    # CHECK BELOW, maybe the fix is action_id[0]
    act_slot_response = copy.deepcopy(self.feasible_agent_actions[action_id])
    # if self.warm_start == 1:
    # else:
    #   act_slot_response = copy.deepcopy(self.feasible_agent_actions[action_id[0]])

    return {'slot_action': act_slot_response, 'action_id': action_id}

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
    act_slot_response = {'dialogue_act': None, 'inform_slots': {}, 'request_slots': {}}

    if self.current_slot_id < len(self.request_set):
      slot = self.request_set[self.current_slot_id]
      self.current_slot_id += 1
      act_slot_response['dialogue_act'] = "request"
      act_slot_response['request_slots'] = {slot: 'UNK'}
    elif self.phase == 0:
      act_slot_response['dialogue_act'] = "inform"
      act_slot_response['inform_slots'] = {'taskcomplete': "PLACEHOLDER"}
      self.phase += 1
    elif self.phase == 1:
      act_slot_response['dialogue_act'] = "thanks"

    return self.action_index(act_slot_response)

  def DQN_policy(self, state_representation):
    """ Return action from DQN"""
    with torch.no_grad():
      action = self.dqn.predict(tensor(state_representation))
    return action

  def action_index(self, act_slot_response):
    """ Return the index of action """

    for (i, action) in enumerate(self.feasible_agent_actions):
      if act_slot_response == action:
        return i

    print('feasible_agent_actions', self.feasible_agent_actions)
    print('act_slot_response', act_slot_response)
    print("action index not found")
    pdb.set_trace()
    return None

  def prepare_user_intent(self, user_action):
    if self.args.task == 'end_to_end':
      user_representations = []
      act_mapper = dialog_constants.lexicon['act_mapper']
      slot_mapper = dialog_constants.lexicon['slot_mapper']
      val_mapper = dialog_constants.lexicon['val_mapper']

      for slot in self.slot_set:
        vals = self.value_set[slot]
        partial_user_rep = np.zeros((1, len(vals) ))

        if slot == 'act':
          converted = act_mapper[user_action['dialogue_act']]
          if converted != 'skip':
            partial_user_rep[0, vals.index(converted)] = 1.0
        elif slot == 'request':
          for req_slot in user_action['request_slots'].keys():
            if req_slot not in vals:
              req_slot = 'other'
            partial_user_rep[0, vals.index(req_slot)] = 1.0
        else:    # slot is some type of inform
          for inf_slot, inf_val in user_action['inform_slots'].items():
            if slot == inf_slot:
              if inf_val in val_mapper.keys():
                inf_val = val_mapper[inf_val]
              try:
                partial_user_rep[0, vals.index(inf_val)] = 1.0
              except(ValueError):
                print(len(vals))
                print("val", inf_val)
                print("slot", inf_slot)
                pdb.set_trace()
                sys.exit()

        user_representations.append(partial_user_rep)

      return user_representations

    else:   # for discrete user intents
      user_act_rep = np.zeros((1, self.act_cardinality))
      user_act_rep[0, self.act_set[user_action['dialogue_act']]] = 1.0

      user_inform_rep = np.zeros((1, self.slot_cardinality))
      for slot in user_action['inform_slots'].keys():
        user_inform_rep[0, self.slot_set[slot]] = 1.0

      user_request_rep = np.zeros((1, self.slot_cardinality))
      for slot in user_action['request_slots'].keys():
        user_request_rep[0, self.slot_set[slot]] = 1.0

      return [user_act_rep, user_inform_rep, user_request_rep]


  def prepare_user_rep(self, user_action, state_type):
    if state_type == 'belief':     # for continuous user beliefs
      user_representations = []

      for slot in self.slot_set:
        vals = self.value_set[slot]
        partial_user_rep = np.zeros((1, len(vals) ))

        for val, confidence_score in user_action[f'{slot}_slots'].items():
          partial_user_rep[0, vals.index(val)] = confidence_score
        user_representations.append(partial_user_rep)

      return user_representations

    elif state_type == 'intent':
      return self.prepare_user_intent(user_action)
    else:
      raise(Exception('missing a user state type!'))

  def prepare_frame_rep(self, current, state_type):
    if state_type == 'belief':
      inform_slots = copy.deepcopy(self.slot_set)
      del inform_slots['act']
      del inform_slots['request']

      frame_rep = np.zeros((1, len(inform_slots) ))  # (1, 8)
      for inf_slot, inf_value in current['inform_slots'].items():
        # if inf_value in self.value_set[slot]:
        idx = self.value_set[slot].index(inf_value)
        frame_rep[0, inform_slots[inf_slot]] = idx

    elif state_type == 'intent':
      frame_rep = np.zeros((1, self.slot_cardinality))
      for inf_slot in current['inform_slots']:
        frame_rep[0, self.old_slots[inf_slot]] = 1.0

    return frame_rep

  def prepare_agent_rep(self, agent_action):
    agent_act_rep = np.zeros((1, self.act_cardinality))
    agent_inform_rep = np.zeros((1, self.slot_cardinality))
    agent_request_rep = np.zeros((1, self.slot_cardinality))

    if agent_action:
      agent_act_rep[0, self.old_acts[agent_action['dialogue_act']]] = 1.0
      for slot in agent_action['inform_slots'].keys():
        agent_inform_rep[0, self.old_slots[slot]] = 1.0
      for slot in agent_action['request_slots'].keys():
        agent_request_rep[0, self.old_slots[slot]] = 1.0

    return agent_act_rep, agent_inform_rep, agent_request_rep

  def prepare_turns_and_kb(self, turn_count, kb_results, state_type):
    if state_type == 'belief' or self.args.task == 'end_to_end':
      turn_rep = np.zeros((1, self.max_turn + 5))
      turn_rep[0, turn_count] = 1.0
      turn_rep[0, -1] = turn_count

      kb_size = len(kb_results)
      kb_count = np.array([[kb_size]])
      return [turn_rep, kb_count]
    elif state_type == 'intent':
      # One-hot representation of the turn count
      turn_rep = np.zeros((1, 1))
      turn_onehot_rep = np.zeros((1, self.max_turn + 5))
      turn_onehot_rep[0, turn_count] = 1.0
      # Representation of KB results (binary)
      kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))
      kb_count_rep = np.zeros((1, self.slot_cardinality + 1))

      return [turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep]

  def totals(self, rep):
    counter = 0
    for thing in rep:
      counter += thing.shape[1]
    # print(counter)
    return counter

  def prepare_state_representation(self, state):
    """ Create the representation for each state """
    state_type = state['user_action']['type']
    state_representation = []

    # Encode last user dialogue act, inform and request
    user_rep = self.prepare_user_rep(state['user_action'], state_type)
    state_representation.extend(user_rep)
    self.totals(state_representation)
    # Encode last agent dialogue act, inform and request
    agent_representations = self.prepare_agent_rep(state['agent_action'])
    state_representation.extend(agent_representations)
    self.totals(state_representation)
    # Create bag of filled_in slots based on the frame
    frame_rep = self.prepare_frame_rep(state['current_slots'], state_type)
    state_representation.append(frame_rep)
    self.totals(state_representation)
    # Get representations of the turn count and knowledge base
    turn_kb_rep = self.prepare_turns_and_kb(state['turn_count'], state['kb_results_dict'], state_type)
    state_representation.extend(turn_kb_rep)
    yup = self.totals(state_representation)

    return np.hstack(state_representation)

  def store_experience(self, current_state, action, reward, next_state, episode_over):
    current_rep = self.prepare_state_representation(current_state)
    next_rep = self.prepare_state_representation(next_state)
    training_example = (current_rep, action, reward, next_rep, episode_over)

    if self.predict_mode == False:  # Training Mode
      if self.warm_start == 1:
        self.experience_replay_pool.append(training_example)
    else:  # Prediction Mode
      if self.use_world_model:
        self.experience_replay_pool_from_model.append(training_example)
      else:  # we directly append since replay pool is now a deque
        self.experience_replay_pool.append(training_example)

  def sample_from_buffer(self, batch_size):
    """Sample batch size examples from experience buffer and convert it to torch readable format"""
    # type: (int, ) -> Transition

    batch = [random.choice(self.running_pool) for i in range(batch_size)]
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
    erp = list(self.experience_replay_pool)
    erpfm = list(self.experience_replay_pool_from_model)
    self.running_pool = erp + erpfm

    for iter_batch in range(num_batches):
      for _ in range(round(len(self.running_pool) / (batch_size))):
        self.optimizer.zero_grad()
        batch = self.sample_from_buffer(batch_size)

        raw_state_value = self.dqn(tensor(batch.state))
        state_value = raw_state_value.gather(1, torch.LongTensor(batch.action))

        next_state_value, _ = self.target_dqn(tensor(batch.next_state)).max(1)
        next_state_value = next_state_value.unsqueeze(1)
        term = np.asarray(batch.term, dtype=np.float32)

        current_reward = tensor(batch.reward)
        future_reward = self.gamma * next_state_value * (1 - tensor(term))
        expected_value = current_reward + future_reward

        loss = F.mse_loss(state_value, expected_value)
        loss.backward()
        self.optimizer.step()
        self.cur_bellman_err += loss.item()

    if verbose and len(self.experience_replay_pool) != 0:
      print("cur bellman error %.2f, experience replay pool %s, model replay pool %s" % (
          float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
          len(self.experience_replay_pool), len(self.experience_replay_pool_from_model)))

  def reward_function(self, dialog_status):
    # Reward Function 1: a reward function based on the dialog_status
    if dialog_status == dialog_constants.FAILED_DIALOG:
      reward = -self.max_turn                     # -40
    elif dialog_status == dialog_constants.SUCCESS_DIALOG:
      reward = 2 * self.max_turn                  # +80
    else:  # for per turn
      reward = -1                                 # -20 over time
    return reward

  def reset_dqn_target(self):
    self.target_dqn.load_state_dict(self.dqn.state_dict())

"""
  # TODO: merge from DialogManager Later
  def __init__(self, args, model, kb, ontology):
    self.state = DialogueState(kb, ontology)
    self.user = CommandLineUser(args, ontology) if args.user == "command" else UserSimulator(args, ontology)

  def prepare_distributed_state(self, state):
    if self.belief_state_type == 'discrete':
      return self.prepare_discrete_state(state)
    elif self.belief_state_type == 'distributed':
      return self.prepare_distributed_state(state)


    # Create the representation for each state
    user_action = state['user_action']
    frame = state['current_slots']
    agent_last = state['agent_action']
    kb_results_dict = state['kb_results_dict']

    # Create one-hot of acts to represent the current user action
    user_act_rep[0, self.act_set[user_action['dialogue_act']]] = 1.0
    for slot in user_action['inform_slots'].keys():
      user_inform_slots_rep[0, self.slot_set[slot]] = 1.0
    # Create bag of request slots representation from user action
    for slot in user_action['request_slots'].keys():
      user_request_slots_rep[0, self.slot_set[slot]] = 1.0

    # Create bag of filled_in slots based on the frame
    frame_rep = np.zeros((1, self.slot_cardinality))
    for slot in frame['inform_slots']:
      frame_rep[0, self.slot_set[slot]] = 1.0

    # Encode last agent dialogue act, inform and request
    agent_act_rep = np.zeros((1, self.act_cardinality))
    agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
    agent_request_slots_rep = np.zeros((1, self.slot_cardinality))

    if agent_last:
      agent_act_rep[0, self.act_set[agent_last['dialogue_act']]] = 1.0
      for slot in agent_last['inform_slots'].keys():
        agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0
      for slot in agent_last['request_slots'].keys():
        agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

    #  One-hot scalar representation of the turn count
    turn_rep = np.zeros((1,1)) + state['turn_count'] / 10.

    #   Representation of KB results (binary)
    kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

    final_representation = np.hstack(
      [user_act_rep, user_inform_slots_rep, user_request_slots_rep, frame_rep,
        agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep,
        turn_rep, kb_binary_rep])
    return final_representation



"""
