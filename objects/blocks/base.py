import os, pdb, sys
import re
import math
import json

import torch
import torch.nn as nn
from torch import optim
from collections import defaultdict



class BaseModule(object):
  def __init__(self, args, model):
    self.args = args
    self.model = model

  def init_optimizer(self, parameters=None):
    model_params = self.parameters() if parameters is None else parameters

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

  @classmethod
  def load_config(cls, fname, ontology, **kwargs):
    with open(fname) as f:
      print('Loading config from {}'.format(fname))
      args = object()
      for k, v in json.load(f):
        setattr(args, k, kwargs.get(k, v))
    return cls(args, ontology)

  def save(self, summary, identifier):
    fname = '{}/{}.pt'.format(self.save_dir, identifier)
    print('saving model to {}.pt'.format(identifier))
    state = {
      'args': vars(self.args),
      'model': self.model.state_dict(),
      'summary': summary,
      'optimizer': self.optimizer.state_dict(),
    }
    torch.save(state, fname)

  def load(self, fname):
    print('loading model from {}'.format(fname))
    state = torch.load(fname)
    self.model.load_state_dict(state['model'])
    self.init_optimizer()
    self.optimizer.load_state_dict(state['optimizer'])

  def get_saves(self, directory=None):
    if directory is None:
      directory = self.save_dir
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    scores = []
    for fname in files:
      re_str = r'dev_{}=([0-9\.]+)'.format(self.args.early_stop)
      dev_acc = re.findall(re_str, fname)
      if dev_acc:
        score = float(dev_acc[0].strip('.'))
        scores.append((score, os.path.join(directory, fname)))
    if not scores:
      raise Exception('No files found!')
    scores.sort(key=lambda tup: tup[0], reverse=True)
    return scores

  def prune_saves(self, n_keep=5):
    scores_and_files = self.get_saves()
    if len(scores_and_files) > n_keep:
      for score, fname in scores_and_files[n_keep:]:
        os.remove(fname)

  def load_best_save(self, directory):
    scores_and_files = self.get_saves(directory=directory)
    if scores_and_files:
      assert scores_and_files, 'no saves exist at {}'.format(directory)
      score, fname = scores_and_files[0]
      self.load(fname)


class BaseBeliefTracker(BaseModule):
  def __init__(self, args, model):
    super().__init__(args, model)

  def quant_report(self, data):
    predictions = self.run_glad_inference(data)
    return data.evaluate_preds(predictions)

  def qual_report(self, data):
    self.eval()
    one_batch = next(dev_data.batch(self.batch_size, shuffle=True))
    loss, scores = self.forward(one_batch)
    predictions = self.extract_predictions(scores)
    return data.run_report(one_batch, predictions, scores)

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


  def state_to_action(self, state, available_actions):
    """ Take the current state and return an action according to the current exploration/exploitation policy

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

  def action_to_nl(self, agent_action):
    """ Add natural language capabilities (NL) to Agent Dialogue Act """
    if agent_action['slot_action']:
      chosen_action = agent_action['slot_action']
      user_response = self.text_generator.generate(chosen_action, 'agt')
      agent_action['slot_action']['nl'] = user_response
    elif agent_action['slot_value_action']:
      agent_action['slot_value_action']['nl'] = ""
      chosen_action = agent_action['slot_value_action']
      user_response = self.text_generator.generate(chosen_action, 'agt')
      agent_action['slot_action']['nl'] = user_response

class BaseTextGenerator(BaseModule):
  def __init__(self, args, model):
    super().__init__(args, model)



