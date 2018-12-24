import numpy as np
import os, pdb, sys  # set_trace
import logging
import numbers
import math
import json
import re

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from objects.components import var, device
from collections import defaultdict
from pprint import pformat

class SelfAttention(nn.Module):
  """
  scores each element of the sequence with a linear layer and uses
  the normalized scores to compute a context over the sequence.
  """
  def __init__(self, d_hid, dropout=0.):
    super().__init__()
    self.scorer = nn.Linear(d_hid, 1)
    self.dropout = nn.Dropout(dropout)

  def forward(self, inp, lens):
    batch_size, seq_len, hidden_dim = inp.size()
    inp = self.dropout(inp)
    raw_scores = self.scorer(inp.contiguous().view(-1, hidden_dim))
    scores = raw_scores.view(batch_size, seq_len)
    max_len = max(lens)
    for i, l in enumerate(lens):
      if l < max_len:
        scores.data[i, l:] = -np.inf
    scores = F.softmax(scores, dim=1)
    context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
    return context


class Attention(nn.Module):
  def __init__(self, method, hidden_size):
    super(Attention, self).__init__()
    self.attn_method = method
    self.tanh = nn.Tanh()
    # the "_a" stands for the "attention" weight matrix
    if self.attn_method == 'luong':                # h(Wh)
      self.W_a = nn.Linear(hidden_size, hidden_size)
    elif self.attn_method == 'vinyals':            # v_a tanh(W[h_i;h_j])
      self.W_a =  nn.Linear(hidden_size * 2, hidden_size)
      self.v_a = nn.Parameter(var(1, hidden_size))
    elif self.attn_method == 'dot':                 # h_j x h_i
      self.W_a = torch.eye(hidden_size) # identity since no extra matrix is needed

  def forward(self, decoder_hidden, encoder_outputs):
    # Create variable to store attention scores           # seq_len = batch_size
    seq_len = len(encoder_outputs)
    attn_scores = var(torch.zeros(seq_len))    # B (batch_size)
    # Calculate scores for each encoder output
    for i in range(seq_len):           # h_j            h_i
        attn_scores[i] = self.score(decoder_hidden, encoder_outputs[i]).squeeze(0)
    # Normalize scores into weights in range 0 to 1, resize to 1 x 1 x B
    attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(0).unsqueeze(0)
    return attn_weights

  def score(self, h_dec, h_enc):
    W = self.W_a
    if self.attn_method == 'luong':                # h(Wh)
      return h_dec.matmul( W(h_enc).transpose(0,1) )
    elif self.attn_method == 'vinyals':            # v_a tanh(W[h_i;h_j])
      hiddens = torch.cat((h_enc, h_dec), dim=1)
      # Note that W_a[h_i; h_j] is the same as W_1a(h_i) + W_2a(h_j) since
      # W_a is just (W_1a concat W_2a)             (nx2n) = [(nxn);(nxn)]
      return self.v_a.matmul(self.tanh( W(hiddens).transpose(0,1) ))
    elif self.attn_method == 'dot':                # h_j x h_i
      return h_dec.matmul(h_enc.transpose(0,1))

class Transformer(nn.Module):
  def __init__(self, vocab_size, hidden_size, n_layers, masked=False, max_len=30):
    super(Transformer, self).__init__()
    self.hidden_size = hidden_size
    self.scale_factor = math.sqrt(hidden_size)
    self.num_attention_heads = 8  # hardcoded since it won't change
    self.num_layers = n_layers   # defaults to 6 to follow the paper
    self.positions = positional_encoding(hidden_size, max_len+1)

    self.dropout = nn.Dropout(0.2)
    self.masked = masked

    for head_idx in range(self.num_attention_heads):
      for vector_type in ['query', 'key', 'value']:
        head_name = "{0}_head_{1}".format(vector_type, head_idx)
        mask_name = "{0}_mask_{1}".format(vector_type, head_idx)
        head_in = self.hidden_size
        head_out = int(self.hidden_size / self.num_attention_heads)
        setattr(self, head_name, nn.Linear(head_in, head_out))
        if masked:
          setattr(self, mask_name, nn.Linear(head_in, head_out))

    self.pw_ffn_1 = nn.Linear(self.hidden_size, hidden_size)
    self.pw_ffn_2 = nn.Linear(self.hidden_size, hidden_size)
    try:
      self.layernorm = nn.LayerNorm(hidden_size, affine=False)
    except(AttributeError):
      self.layernorm = nn.BatchNorm1d(hidden_size, affine=False)

  def forward(self, inputs, encoder_outputs=None, di=None):
    # inputs will be seq_len, batch_size, hidden dim.  However, our batch_size
    # is always one so we squeeze it out to keep calculations simpler
    position_emb = torch.tensor(self.positions[:len(inputs), :], requires_grad=False)
    # if batch_size > 1, self.positions[:len(inputs), :1, :inputs.size(2)].expand_as(inputs)
    transformer_input = inputs.squeeze() + position_emb
    k_v_input = self.dropout(transformer_input)

    for layer_idx in range(self.num_layers):
      if layer_idx > 0:
        transformer_input = self.dropout(transformer_output)

      if self.masked:
        masked_input = self.apply_mask(transformer_input, di)
        k_v_input = encoder_outputs

        mask_attn_heads = []
        for j in range(self.num_attention_heads):
          Q = getattr(self, "query_mask_{}".format(j))(masked_input)
          K = getattr(self, "key_mask_{}".format(j))(k_v_input)
          V = getattr(self, "value_mask_{}".format(j))(k_v_input)
          mask_attn_heads.append(self.scaled_dot_product_attention(Q, K, V))
        residual_connection = masked_input + torch.cat(mask_attn_heads, dim=1)
        masked_output = self.layernorm(residual_connection)
        transformer_input = self.dropout(masked_output)

      attn_heads = []  # don't create a new variable since it messes with the graph
      for idx in range(self.num_attention_heads):
        Q = getattr(self, "query_head_{}".format(idx))(transformer_input)
        K = getattr(self, "key_head_{}".format(idx))(k_v_input)
        V = getattr(self, "value_head_{}".format(idx))(k_v_input)
        attn_heads.append(self.scaled_dot_product_attention(Q, K, V))
      residual_connection = transformer_input + torch.cat(attn_heads, dim=1)
      multihead_output = self.layernorm(residual_connection)

      pw_ffn_output = self.positionwise_ffn(multihead_output)
      transformer_output = self.layernorm(multihead_output + pw_ffn_output)

    return transformer_output

  def apply_mask(self, decoder_inputs, decoder_idx):
    mask = var(torch.zeros(( decoder_inputs.shape )), "variable")
    mask[:decoder_idx+1, :] = 1
    return decoder_inputs * mask
    # updated code for dealing with batch_size >1; lengths is a list,
    # where each element is an integer representing the number of tokens
    # for each sentence in the batch
    # batch_size = lengths.numel()
    # max_len = max_len or lengths.max()
    # return (torch.arange(0, max_len)
    #         .type_as(lengths)
    #         .repeat(batch_size, 1)
    #         .lt(lengths.unsqueeze(1)))  # less than

  def positionwise_ffn(self, multihead_output):
    ffn_1_output = F.relu(self.pw_ffn_1(multihead_output))
    ffn_2_output = self.pw_ffn_2(ffn_1_output)
    return ffn_2_output

  def scaled_dot_product_attention(self, Q, K, V):
    # K should be seq_len, hidden_dim / 8 to start with, but will be transposed
    scaled_matmul = Q.matmul(K.transpose(0,1)) / self.scale_factor   # batch_size x seq_len
    # (batch_size, hidden_dim) x (hidden_dim, seq_len) / broadcast integer
    attn_weights = F.softmax(scaled_matmul, dim=1)
    attn_context = attn_weights.matmul(V)             # batch_size, hidden_dim
    return attn_context

  def positional_encoding(dim, max_len=5000):
    # Implementation based on "Attention Is All You Need"
    pe = torch.arange(0, max_len).unsqueeze(1).expand(max_len, dim)
    div_term = 1 / torch.pow(10000, torch.arange(0, dim * 2, 2) / dim)
    pe = pe * div_term.expand_as(pe)
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe  # .unsqueeze(1)

def attend(seq, cond, lens):
  """
  attend over the sequences `seq` using the condition `cond`.
  `seq` are usually the hidden states of an RNN as a matrix
  `cond` is usually the vector of the current decoder state
  """
  scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
  max_len = max(lens)
  for i, l in enumerate(lens):
    if l < max_len:
      scores.data[i, l:] = -np.inf
  scores = F.softmax(scores, dim=1)
  context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
  return context, scores

class ModelTemplate(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.opt = args.optimizer
    self.lr = args.learning_rate
    self.reg = args.weight_decay

    self.dhid = args.hidden_size
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
      summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
      summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})

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

# the GLAD model described in https://arxiv.org/abs/1805.09655.
class GlobalLocalModel(ModelTemplate):
  def __init__(self, args, ontology, vocab, Eword, GLADEncoder):
    super().__init__(args)
    self.optimizer = None

    self.vocab = vocab
    self.ontology = ontology
    if args.pretrained:
      self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(Eword))
    else:
      self.embedding = nn.Embedding(len(vocab), self.demb)  # (num_embeddings, embedding_dim)

    self.demb = args.embedding_size    # aka embedding dimension
    self.dhid = args.hidden_size      # aka hidden state dimension
    dropout = {key: args.drop_prob for key in ["emb", "local", "global"]}

    self.utt_encoder = GLADEncoder(self.demb, self.dhid, ontology.slots, dropout)
    self.act_encoder = GLADEncoder(self.demb, self.dhid, ontology.slots, dropout)
    self.ont_encoder = GLADEncoder(self.demb, self.dhid, ontology.slots, dropout)
    self.utt_scorer = nn.Linear(2 * self.dhid, 1)
    self.score_weight = nn.Parameter(torch.Tensor([0.5]))

  def forward(self, batch):
    # convert to variables and look up embeddings
    eos = self.vocab.word2index('<eos>')
    utterance, utterance_len = self.pad([e.num['utterance'] for e in batch], self.embedding, pad=eos)
    acts = [self.pad(e.num['agent_actions'], self.embedding, pad=eos) for e in batch]
    ontology = {s: self.pad(v, self.embedding, pad=eos) for s, v in self.ontology.num.items()}

    ys = {}
    for s in self.ontology.slots:
      # for each slot, compute the scores for each value
      H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s)
      # H_utt: torch.Size([50, 30, 400])  batch_size x seq_len x embed_dim
      # c_utt: torch.Size([50, 400])
      _, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))
      _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s)
      # C_acts is list of length 50, a single c_act is size([1, 400])
      # C_vals is list of length 7, a single c_val is size([400])

      # compute the utterance score
      y_utts = []
      q_utts = []
      for i, c_val in enumerate(C_vals):
        q_utt, _ = attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance_len)
        q_utts.append(q_utt)   # torch.Size([50, 400])
      y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)

      # compute the previous action score
      q_acts = []
      for j, C_act in enumerate(C_acts):
        q_act, _ = attend(C_act.unsqueeze(0), c_utt[j].unsqueeze(0), lens=[C_act.size(0)])
        q_acts.append(q_act)  # torch.Size([1, 400])
      # (50x7) =         (50, 400)       x    (400, 7)
      y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

      # combine the scores
      # y_acts: torch.Size([50, 7])  batch size, num of values for slot 's'
      # y_utts: torch.Size([50, 7])  for slot==area, there are 7 values
      ys[s] = torch.sigmoid(y_utts + self.score_weight * y_acts)
      # ys[s] = torch.sigmoid(c_utt.mm(C_vals.transpose(0, 1)))

    if self.training:
      # create label variable and compute loss
      labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
      for i, e in enumerate(batch):
        for s, v in e.user_intent:
          labels[s][i][self.ontology.values[s].index(v)] = 1
      labels = {s: torch.Tensor(m).to(device) for s, m in labels.items()}

      loss = 0
      for s in self.ontology.slots:
        loss += F.binary_cross_entropy(ys[s], labels[s])
    else:
      loss = torch.Tensor([0]).to(device)
    return loss, {s: v.data.tolist() for s, v in ys.items()}

  def pad(self, seqs, emb, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens
    # out = F.dropout(emb(padded.to(device)), drop_rate)
    # return out, lens

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

