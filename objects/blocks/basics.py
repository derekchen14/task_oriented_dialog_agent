import re
import math
import json
import logging
import numbers
import numpy as np
import os, pdb, sys  # set_trace
from pprint import pformat
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from objects.components import var, device
from utils.external.reader import get_glove_name


class ModelTemplate(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.opt = args.optimizer
    self.lr = args.learning_rate
    self.reg = args.weight_decay

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

  def get_train_logger(self):
    logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
    formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    file_handler = logging.FileHandler(os.path.join(self.save_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

  def learn(self, datasets, args):
    train, dev = datasets['train'], datasets['val']
    track = defaultdict(list)
    iteration = 0
    best = {}
    logger = self.get_train_logger()
    self.init_optimizer()

    for epoch in range(args.epochs):
      logger.info('starting epoch {}'.format(epoch))

      # train and update parameters
      self.train()
      for batch in train.batch(batch_size=args.batch_size, shuffle=True):
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
      summary.update({'eval_train_{}'.format(k): v for k, v in self.quant_report(train, args).items()})
      summary.update({'eval_dev_{}'.format(k): v for k, v in self.quant_report(dev, args).items()})

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
        dev.record_preds(
          preds=self.run_pred(dev, self.args),
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

  def run_pred(self, data, args):
    self.eval()
    predictions = []
    for batch in data.batch(batch_size=args.batch_size):
      loss, scores = self.forward(batch)
      predictions += self.extract_predictions(scores)
    return predictions

  def quant_report(self, data, args):
    predictions = self.run_pred(data, args)
    return data.evaluate_preds(predictions)

  def qual_report(self, data, args):
    self.eval()
    one_batch = next(dev.batch(args.batch_size, shuffle=True))
    loss, scores = self.forward(one_batch)
    predictions = self.extract_predictions(scores)
    return data.run_report(one_batch, predictions, scores)

  def save_config(self, save_directory):
    fname = '{}/config.json'.format(save_directory)
    with open(fname, 'wt') as f:
      logging.info('Saving config to {}'.format(fname))
      json.dump(vars(self.args), f, indent=2)

  @classmethod
  def load_config(cls, fname, ontology, **kwargs):
    with open(fname) as f:
      logging.info('Loading config from {}'.format(fname))
      args = object()
      for k, v in json.load(f):
        setattr(args, k, kwargs.get(k, v))
    return cls(args, ontology)

  def save(self, summary, identifier):
    fname = '{}/{}.pt'.format(self.save_dir, identifier)
    logging.info('saving model to {}.pt'.format(identifier))
    state = {
      'args': vars(self.args),
      'model': self.state_dict(),
      'summary': summary,
      'optimizer': self.optimizer.state_dict(),
    }
    torch.save(state, fname)

  def load(self, fname):
    logging.info('loading model from {}'.format(fname))
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
      re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop_early)
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

class BasicClassifer(ModelTemplate):
  def __init__(self, encoder, ff_network, args):
    super().__init__(args)
    self.encoder = encoder.to(device)
    self.decoder = ff_network.to(device)
    self.type = "basic"
  def forward(self, sources, hidden):
    self.encoder.rnn.flatten_parameters()
    encoder_outputs, hidden = self.encoder(sources, hidden)
    return self.decoder(encoder_outputs[0])

class PretrainedEmbeddingsModel(nn.Module):
  """
  Base class that allows pretrained embedding to be loaded from
  a vocabulary. Assumes the wrapper class will initialize the embeddings
  """
  def load_embeddings(self, vocab):
    if self.opt.pt != "none":
      # Get glove vector file name
      name = get_glove_name(self.opt, "tokens")
      print("Loading embeddings from {}".format(name))

      # Load glove vectors
      with open(name, "r") as f:
        token_words = pickle.load(f)

      # Assign glove vectors to correct word in vocab
      for i, word in vocab.iteritems():

        # If vocab word is meaning less, set embedding to 0
        if word in ["<unk>", "<start>", "<end>", "<pad>"]:
          self.embeddings.weight.data[i].zero_()
          continue

        if self.is_cuda:
          vec = torch.cuda.FloatTensor(token_words[word])
        else:
          vec = torch.FloatTensor(token_words[word])

        # Set embedding in embedding module
        self.embeddings.weight.data[i] = vec

class WeightedBOW(PretrainedEmbeddingsModel):
  """
  Indexes a set of word embeddings for a sequence of words it receives
  as input. Weighs each of these embeddings by a learned mask
  and returns the sum

  Initialization Args:
    opt.vSize: number of embeddings to initialize
    opt.hSize: size of embeddings to initialize
    opt.dpt: dropout probability after the embedding layer (default = 0)
    max_size: maximum number of masking functions

  Input:
    input: 3-dimensional tensor of batch_size x seq_len x embed_size

  Output:
    2-dimensional tensor of size batch_size x embed_size

  """
  def __init__(self, opt, max_size):
    super(WeightedBOW, self).__init__()
    self.embeddings = nn.Embedding(opt.vSize, opt.hSize, padding_idx=0)

    self.weights = nn.Parameter(
      torch.FloatTensor(max_size, opt.hSize).fill_(1))

    self.dropout = nn.Dropout(opt.dpt)

    self.is_cuda = False
    self.max_size = max_size
    self.opt = opt

  def forward(self, input):
    batch_size = input.size(0)
    length = input.size(1)

    embed = self.embeddings(input)
    dropped = self.dropout(embed)

    # Multiply dropped embeddings by mask weights
    w_dropped = dropped * self.weights[:length, :].unsqueeze(0).repeat(
      batch_size, 1, 1)

    return torch.sum(w_dropped, 1).view(batch_size, self.opt.hSize), None

  def cuda(self, device_id):
    super(WeightedBOW, self).cuda(device_id)
    self.weights.cuda(device_id)
    self.is_cuda = True