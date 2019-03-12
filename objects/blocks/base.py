import os, pdb, sys
import re
import math
import json

import torch
import torch.nn as nn
from torch import optim
from collections import defaultdict

class BaseBeliefTracker(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.opt = args.optimizer
    self.lr = args.learning_rate
    self.reg = args.weight_decay
    self.batch_size = args.batch_size

    self.dhid = args.hidden_dim
    self.demb = args.embedding_size
    self.n_layers = args.num_layers

  def init_optimizer(self):
    if self.opt == 'sgd':
      self.optimizer = optim.SGD(self.parameters(), self.lr, self.reg)
    elif self.opt == 'adam':
      # warmup = step_num * math.pow(4000, -1.5)   -- or -- lr = 0.0158
      # self.lr = (1 / math.sqrt(d)) * min(math.pow(step_num, -0.5), warmup)
      self.optimizer = optim.Adam(self.parameters(), self.lr)
    elif self.opt == 'rmsprop':
      self.optimizer = optim.RMSprop(self.parameters(), self.lr, self.reg)

  def learn(self, args, datasets):
    train_data, dev_data = datasets['train'], datasets['val']
    track = defaultdict(list)
    iteration = 0
    best = {}
    logger = self.get_train_logger()
    self.init_optimizer()

    for epoch in range(args.epochs):
      logger.info('starting epoch {}'.format(epoch))

      # train and update parameters
      self.train()
      for batch in train_data.batch(batch_size=args.batch_size, shuffle=True):
        iteration += 1
        self.zero_grad()
        loss, scores = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        track['loss'].append(loss.item())

      # evalute on train and dev
      summary = {'iteration': iteration, 'epoch': epoch}
      for k, v in track.items():
        summary[k] = sum(v) / len(v)
      summary.update({'eval_train_{}'.format(k): v for k, v in self.quant_report(train_data).items()})
      summary.update({'eval_dev_{}'.format(k): v for k, v in self.quant_report(dev_data).items()})

      # do early stopping saves
      stop_key = 'eval_dev_{}'.format(args.stop_early)
      train_key = 'eval_train_{}'.format(args.stop_early)
      if best.get(stop_key, 0) <= summary[stop_key]:
        best_dev = '{:f}'.format(summary[stop_key])
        best_train = '{:f}'.format(summary[train_key])
        best.update(summary)
        self.save(best,
          identifier='epoch={epoch},iter={iteration},train_{key}={train},dev_{key}={dev}'.format(
            epoch=epoch, iteration=iteration, train=best_train, dev=best_dev, key=args.stop_early,
          )
        )
        self.prune_saves()
        dev_data.record_preds(
          preds=self.run_glad_inference(dev_data),
          to_file=os.path.join(self.save_dir, 'dev.pred.json'),
        )
      summary.update({'best_{}'.format(k): v for k, v in best.items()})
      logger.info(pformat(summary))
      track.clear()


  def extract_predictions(self, scores, threshold=0.5):
    batch_size = len(list(scores.values())[0])
    predictions = [set() for i in range(batch_size)]
    for s in self.ontology.slots:
      for i, p in enumerate(scores[s]):
        triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > threshold]
        if s == 'request':
          # we can have multiple requests predictions
          predictions[i] |= set([(s, v) for s, v, p_v in triggered])
        elif triggered:
          # only extract the top inform prediction
          sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
          predictions[i].add((sort[0][0], sort[0][1]))
    return predictions

  def run_glad_inference(self, data):
    self.eval()
    predictions = []
    for batch in data.batch(self.batch_size):
      loss, scores = self.forward(batch)
      predictions += self.extract_predictions(scores)
    return predictions

  def quant_report(self, data):
    predictions = self.run_glad_inference(data)
    return data.evaluate_preds(predictions)

  def qual_report(self, data):
    self.eval()
    one_batch = next(dev_data.batch(self.batch_size, shuffle=True))
    loss, scores = self.forward(one_batch)
    predictions = self.extract_predictions(scores)
    return data.run_report(one_batch, predictions, scores)

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
      'model': self.state_dict(),
      'summary': summary,
      'optimizer': self.optimizer.state_dict(),
    }
    torch.save(state, fname)

  def load(self, fname):
    print('loading model from {}'.format(fname))
    state = torch.load(fname)
    self.load_state_dict(state['model'])
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


class BasePolicyManager:
  """ Prototype for all agent classes, defining the interface they must uphold """

  def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
    """ Constructor for the Agent class

    Arguments:
    movie_dict      --  This is here now but doesn't belong - the agent doesn't know about movies
    act_set         --  The set of acts. #### Shouldn't this be more abstract? Don't we want our agent to be more broadly usable?
    slot_set        --  The set of available slots
    """
    self.movie_dict = movie_dict
    self.act_set = act_set
    self.slot_set = slot_set
    self.act_cardinality = len(act_set.keys())
    self.slot_cardinality = len(slot_set.keys())

    self.epsilon = params['epsilon']
    self.agent_run_mode = params['agent_run_mode']
    self.agent_act_level = params['agent_act_level']

    self.nlg_model = None
    self.nlu_model = None

  def initialize_episode(self):
    """ Initialize a new episode. This function is called every time a new episode is run. """
    self.current_action = {}                    #   TODO Changed this variable's name to current_action
    self.current_action['diaact'] = None        #   TODO Does it make sense to call it a state if it has an act? Which act? The Most recent?
    self.current_action['inform_slots'] = {}
    self.current_action['request_slots'] = {}
    self.current_action['turn'] = 0

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


  def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
    """  Register feedback from the environment, to be stored as future training data

    Arguments:
    s_t                 --  The state in which the last action was taken
    a_t                 --  The previous agent action
    reward              --  The reward received immediately following the action
    s_tplus1            --  The state transition following the latest action
    episode_over        --  A boolean value representing whether the this is the final action.

    Returns:
    None
    """
    pass

  def add_nl_to_action(self, agent_action):
    """ Add NL to Agent Dia_Act """

    if agent_action['slot_action']:
      agent_action['slot_action']['nl'] = ""
      #TODO
      user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['slot_action'], 'agt') #self.nlg_model.translate_diaact(agent_action['slot_action']) # NLG
      agent_action['slot_action']['nl'] = user_nlg_sentence
    elif agent_action['slot_value_action']:
      agent_action['slot_value_action']['nl'] = ""
      user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['slot_value_action'], 'agt') #self.nlg_model.translate_diaact(agent_action['act_slot_value_response']) # NLG
      agent_action['slot_action']['nl'] = user_nlg_sentence