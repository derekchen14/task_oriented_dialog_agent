import os, pdb, sys
import math
import numpy as np
import json

import torch
import torch.nn as nn
from torch import optim
from collections import defaultdict
from objects.components import get_saves

class BaseModule(object):
  def __init__(self, args, model):
    self.args = args
    self.model = model

    self.batch_size = args.batch_size
    self.save_model = args.save_model

    self.opt = args.optimizer
    self.lr = args.learning_rate
    self.reg = args.weight_decay

  def init_optimizer(self, parameters=None):
    model_params = self.model.parameters() if parameters is None else parameters

    if self.opt == 'sgd':
      self.optimizer = optim.SGD(model_params, self.lr, weight_decay=self.reg)
    elif self.opt == 'adam':
      # warmup = step_num * math.pow(4000, -1.5)   -- or -- lr = 0.0158
      # self.lr = (1 / math.sqrt(d)) * min(math.pow(step_num, -0.5), warmup)
      self.optimizer = optim.Adam(model_params, self.lr)
    elif self.opt == 'rmsprop':
      self.optimizer = optim.RMSprop(model_params, self.lr, weight_decay=self.reg)

  def save_config(self, args, save_directory):
    fname = '{}/config.json'.format(save_directory)
    with open(fname, 'wt') as save_file:
      print('Saving config to {}'.format(fname))
      json.dump(vars(args), save_file, indent=2)


  def save_experience_replay_to_file(self, path):
    pickle.dump(self.experience_replay_pool, open(path, "wb"))
    print('saved model in %s' % (path,))

  # Move into LOADER
  @classmethod
  def load_config(cls, fname, ontology, **kwargs):
    with open(fname) as f:
      print('Loading config from {}'.format(fname))
      args = object()
      for k, v in json.load(f):
        setattr(args, k, kwargs.get(k, v))
    return cls(args, ontology)

  def load_experience_replay_from_file(self, path):
    self.experience_replay_pool = pickle.load(open(path, 'rb'), encoding='latin1')

  def save_checkpoint(self, monitor):
    filepath = os.path.join(self.model.save_dir, monitor.unique_id)
    state = {
      'args': vars(self.args),
      'model': self.model.state_dict(),
      'summary': monitor.summary,
      'optimizer': self.optimizer.state_dict(),
    }
    if self.save_model:
      torch.save(state, filepath)
      print("Saved model at {}".format(filepath))

  def prune_saves(self, n_keep=5):
    scores_and_files = get_saves(self.model.save_dir, self.args.early_stop)
    if len(scores_and_files) > n_keep:
      for score, fname in scores_and_files[n_keep:]:
        os.remove(fname)

class BaseBeliefTracker(BaseModule):
  def __init__(self, args, model):
    super().__init__(args, model)

  def quant_report(self, data):
    predictions = self.run_glad_inference(data)
    return data.evaluate_preds(predictions)

  def qual_report(self, samples, preds, confidence, vals):
    num_samples = len(preds)
    corrects = {'inform': [], 'request': [], 'act': []}
    joint_goal = []
    self.lines = []

    idx = 0
    pred_state = {}
    for sample in samples:
      if idx >= num_samples: break
      possible = set()

      gold = {'inform': set(), 'request': set(), 'act': set()}
      for slot, values in sample.user_intent:
        possible.add(slot)
        if slot in ['request', 'act']:
          gold[slot].add((slot, values))
        else:  # dialogue_act == inform
          gold['inform'].add((slot, values))

      pred = {'inform': set(), 'request': set(), 'act': set()}
      for slot, values in preds[idx]:
        if slot in ['request', 'act']:
          pred[slot].add((slot, values))
        else:  # dialogue_act == inform
          pred['inform'].add((slot, values))

      all_correct = True
      for category in ['inform', 'request', 'act']:
        is_correct = gold[category] == pred[category]
        corrects[category].append(is_correct)
        if not is_correct:
          all_correct = False
      joint_goal.append(all_correct)

      if not all_correct:
        utt = sample.utterance
        self.lines.append(utt if isinstance(utt, str) else " ".join(utt))
        self.lines.append(f'Actual: {sample.user_intent}')
        self.lines.append(f'Predicted: {preds[idx]}')
        self.process_confidence(list(possible), confidence, vals, idx)
        self.lines.append('----------------')

      idx += 1

    for category, scores in corrects.items():
      self.lines.append(f'avg_{category}: {np.mean(scores)}')
    self.lines.append(f'joint_goal: {np.mean(joint_goal)}')
    return self.lines

  def process_confidence(self, possible_slots, confidence, vals, idx):
    for slot in possible_slots:
      conf = []
      for jdx, score in enumerate(confidence[slot][idx]):
        if score > 0.01:
          conf.append(vals[slot][jdx])
        conf.append(round(score,3))

      self.lines.append("{} confidence: {}".format(slot, conf))

  def w2i(self, word):
    try:
      index = self.model.vocab.word2index(word)
    except:
      index = self.model.vocab.word2index('<unk>')
    return index

class BasePolicyManager(BaseModule):
  """ Prototype for all agent classes, defining the interface they must uphold """
  def __init__(self, args, model):
    super().__init__(args, model)
    self.debug = args.debug
    self.verbose = args.verbose
    self.max_turn = args.max_turn
    self.batch_size = args.batch_size

    self.belief_tracker = None
    self.text_generator = None

  def initialize_episode(self, sim=False):
    self.simulation_mode = sim
    self.episode_reward = 0
    self.episode_over = False

    self.state.initialize_episode()
    self.user.initialize_episode()
    self.agent.initialize_episode()

    self.user_action = self.user.user_action
    self.state.update(user_action=self.user_action)
    if self.verbose and self.user.do_print:
      print("New episode, user goal:")
      print(self.user.goal)
      self.print_function(self.user_action, "user")


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

  def utterance_to_intent(self, user_utterance):
    """ Returns the predicted intent state given the raw utterance """
    predicted_intent = self.belief_tracker.track(user_utterance)
    agent_action['slot_action']['nl'] = agent_response

  def state_to_action(self, state, available_actions):
    """ Take the current intent state and return an action according to policy

    We define the agents flexibly so that they can either operate on act_slot representations or act_slot_value representations.
    We also define the responses flexibly, returning a dictionary with keys [act_slot_response, act_slot_value_response]. This way the command-line agent can continue to operate with values

    Arguments:
    state      --   A tuple of (history, kb_results) where history is a sequence of previous actions and kb_results contains information on the number of results matching the current constraints.
    user_action         --   A legacy representation used to run the command line agent. We should remove this ASAP but not just yet
    available_actions   --   A list of the allowable actions in the current state

    Returns:
    act_slot_action         --   An action consisting of one act and >= 0 slots as well as which slots are informed vs requested.
    act_slot_value_action   --   An action consisting of acts slots and values in the legacy format. This can be used in the future for training agents that take value into account and interact directly with the database
    """
    act_slot_response = None
    act_slot_value_response = None
    return {"slot_action": act_slot_response, "slot_value_action": act_slot_value_response}

  def action_to_nl(self, agent_action):
    """ Add natural language capabilities (NL) to Agent Dialogue Act """
    if agent_action['slot_action']:
      chosen_action = agent_action['slot_action']
      agent_response = self.text_generator.generate(chosen_action, 'agt')
      agent_action['slot_action']['nl'] = agent_response
    elif agent_action['slot_value_action']:
      agent_action['slot_value_action']['nl'] = ""
      chosen_action = agent_action['slot_value_action']
      agent_response = self.text_generator.generate(chosen_action, 'agt')
      agent_action['slot_action']['nl'] = agent_response

class BaseTextGenerator(BaseModule):
  def __init__(self, args, model):
    super().__init__(args, model)



