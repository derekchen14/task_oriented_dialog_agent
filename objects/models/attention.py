import numpy as np
import os, pdb, sys  # set_trace
import logging

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from objects.components import var, device
from collections import defaultdict
from pprint import pformat


class Attention(nn.Module):
  """
  Attend to a sequence of tensors using an attention distribution
  Input:
    Sequence: usually the hidden states of an RNN as a matrix
      shape = batch_size x seq_len x hidden_dim
    Condition: usually the vector of the current decoder state
      shape = batch_size x hidden_dim
    Lengths: list of integers, where each integer is the number of tokens
      in each sequence, the length of list should equal batch_size

    weights: attention distribution
      do_reduce: indicator variable indicating whether we compute
                 the weighted average of the attended vectors OR
                 just return them scaled by their attention weight

  Output:
      attention distribution over hidden states
      activations for each hidden state

  """
  def __init__(self, hidden_dim=None, method='dot', act='tanh'):
    super(Attention, self).__init__()
    act_options = { "softmax": nn.Softmax(dim=1),
                    "sigmoid": nn.Sigmoid(),
                    "tanh": nn.Tanh() }
    self.activation = act_options[act]

    if method == 'linear':                # h(Wh)
      self.W_a = nn.Linear(hidden_dim, 1)
    elif method == 'concat':            # v_a tanh(W[h_i;h_j])
      self.W_a =  nn.Linear(hidden_dim * 2, hidden_dim)
      self.v_a = torch.tensor(1, hidden_dim)
    # if attention method is 'dot' no extra matrix is needed
    self.attn_method = method

  def forward(self, sequence, condition, lengths=None):
    """
    Compute context weights for a given type of scoring mechanim
    return the scores along with the weights
    """
    if lengths is None:
      self.masking = False
    else:
      assert len(lengths) == sequence.shape[0] # equal to batch size
      self.lengths = lengths
      self.masking = True

    scores = self.score(sequence, condition)
    # Normalize scores into weights --> batch_size x seq_len
    self.scores = F.softmax(scores, dim=1)
    # batch_size x seq_len --> batch_size x seq_len x hidden_dim
    expanded = self.scores.unsqueeze(2).expand_as(sequence)
    # context weights shape is batch_size x hidden_dim
    context_weights = expanded.mul(sequence).sum(1)

    return context_weights

  def score(self, sequence, condition):
    """
    Calculate activation score over the sequence using the condition.
    Output shape = batch_size x seq_len
    """
    if self.attn_method == 'linear':                # h(Wh)
      batch_size, seq_len, hidden_dim = sequence.size()
      # batch_size, seq_len, hidden_dim --> batch_size x seq_len, 1
      reshaped = condition.contiguous().view(-1, hidden_dim)
      # batch_size x seq_len, 1 --> batch_size, seq_len
      scores = self.W_a(reshaped).view(batch_size, seq_len)
    elif self.attn_method == 'concat':            # v_a tanh(W[h_i;h_j])
      joined = torch.cat((input_2, input_1), dim=1)
      # Note that W_a[h_i; h_j] is the same as W_1a(h_i) + W_2a(h_j) since
      # W_a is just (W_1a concat W_2a)             (nx2n) = [(nxn);(nxn)]
      logit = self.W_a(joined).transpose(0,1)
      scores = self.v_a.matmul(self.tanh(logit))
    elif self.attn_method == 'dot':
      # batch_size x hidden_dim --> batch_size x seq_len x hidden_dim
      expanded = condition.unsqueeze(1).expand_as(sequence)
      # final scores shape is batch_size x seq_len
      scores = expanded.mul(sequence).sum(2)

    if self.masking:
      max_len = max(self.lengths)
      for i, l in enumerate(self.lengths):
        if l < max_len:
          scores.data[i, l:] = -np.inf
    return scores


class Attender(nn.Module):
    """
    Attend to a set of vectors using an attention distribution

    Input:
        input: 3-dimensional tensor of size
                batch_size x seq_len x hidden_state
                These are the hidden states to compare to
        weights: attention distribuution
        do_reduce: indicator variable indicating whether we compute
                   the weighted average of the attended vectors OR
                   just return them scaled by their attention weight

    Output:
        attention distribution over hidden states
        activations for each hidden state

    """
    def __init__(self):
        super(Attender, self).__init__()

    # input is batch_size * num_agenda * input_embed
    # weights is batch_size * num_agenda
    def forward(self, input, weights, do_reduce=True):
        if do_reduce:
            out = torch.bmm(weights.unsqueeze(1), input)
            return out.view(out.size(0), out.size(2))
        else:
            out = weights.unsqueeze(2).repeat(1, 1, input.size(2)) * input
            return out


class DoubleDotAttention(nn.Module):
    """
    Compute a dot product attention between two vectors

    Initialization Args:
        act: which non-linearity to use to compute attention
            (sigmoid for independent attention,
             softmax for joint attention)

    Input:
        input_1: 3-dimensional tensor of batch_size x seq_len x hidden_state
                These are the hidden states to compare to
        input_2: 3-dimensional tensor of batch_size x seq_len x hidden_state
                These are the second hidden states to compare to
        hidden: vector used to compute attentions

    Output:
        attention distribution over hidden states
        activations for each hidden state

    """
    def __init__(self, act="softmax"):
        super(DoubleDotAttention, self).__init__()

        if act == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()

    # Input should be bs x num_agenda x input_size
    def forward(self, input_1, input_2, hidden):
        batch_size = input_1.size(0)
        num_items = input_1.size(1)
        input_size = input_1.size(2)

        assert input_1.size(2) == hidden.size(1)
        assert input_1.size(0) == hidden.size(0)
        assert input_2.size(2) == hidden.size(1)
        assert input_2.size(0) == hidden.size(0)
        hidden_ = hidden.view(batch_size, 1, input_size).expand(
            batch_size, num_items, input_size).contiguous().view(
            -1, input_size)
        inputs_1 = input_1.view(-1, input_size)
        inputs_2 = input_2.view(-1, input_size)

        activation_1 = torch.sum(inputs_1 * hidden_, 1).view(
            batch_size, num_items)

        activation_2 = torch.sum(inputs_2 * hidden_, 1).view(
            batch_size, num_items)

        activation = activation_1 + activation_2

        return self.act(activation), activation