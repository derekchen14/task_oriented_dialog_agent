import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
  def __init__(self, hidden=None, method='dot', act='tanh'):
    super(Attention, self).__init__()
    act_options = { "softmax": nn.Softmax(dim=1),
                    "sigmoid": nn.Sigmoid(),
                    "tanh": nn.Tanh() }
    self.activation = act_options[act]

    if method == 'linear':                # h(Wh)
      self.W_a = nn.Linear(hidden, hidden)
    elif method == 'concat':            # v_a tanh(W[h_i;h_j])
      self.W_a =  nn.Linear(2 * hidden, hidden)
      self.v_a = nn.Linear(hidden, 1, bias=False)
    elif method == 'double':            # v_a tanh(W[h_i;h_j])
      self.W_a =  nn.Linear(2 * hidden, hidden)
      self.v_a = nn.Linear(hidden, 1, bias=False)
    elif method == 'self':
      self.W_a = nn.Linear(hidden, 1)
      self.drop = nn.Dropout(act)  # semi hack to use the param
    # if attention method is 'dot' no extra matrix is needed
    self.attn_method = method

  def forward(self, sequence, condition, lengths=None):
    """
    Compute context weights for a given type of scoring mechanism
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
    # scores unsqueezed is batch_size x 1 x seq_len
    # sequence shape is batch_size x seq_len x hidden_dim
    weights = torch.bmm(self.scores.unsqueeze(1), sequence).squeeze(1)
    # this is equivalent to doing a dot product and summing
    # weights = scores.unsqueeze(2).expand_as(sequence).mul(sequence).sum(1)

    return weights

  def score(self, sequence, condition):
    """
    Calculate activation score over the sequence using the condition.
    Output shape = batch_size x seq_len
    """
    if self.attn_method == 'self':                 # W h
      batch_size, seq_len, hidden_dim = sequence.size()
      # start out with some drop out
      sequence = self.drop(sequence)
      # batch_size, seq_len, hidden_dim --> batch_size x seq_len, 1
      reshaped = sequence.contiguous().view(-1, hidden_dim)
      # batch_size x seq_len, 1 --> batch_size, seq_len
      scores = self.W_a(reshaped).view(batch_size, seq_len)

    elif self.attn_method == 'linear':                # p(Wq)
      # after affine, sequence keeps the same batch_size x seq_len x hidden_dim
      # after unsqueeze, condition shape becomes batch_size x hidden_dim x 1
      product = torch.bmm(self.W_a(sequence), condition.unsqueeze(2))
      # final score shape is just batch_size x seq_len as expected
      scores = product.squeeze(2)

    elif self.attn_method == 'concat':            # v_a tanh(W[h_i;h_j])
      # batch_size x hidden_dim --> batch_size x seq_len x hidden_dim
      expanded = condition.unsqueeze(1).expand_as(sequence)
      # joined shape becomes batch_size x seq_len x (2 * hidden_dim)
      # Note that W_a[h_i; h_j] is the same as W_1a(h_i) + W_2a(h_j) since
      # W_a is just (W_1a concat W_2a)             (nx2n) = [(nxn);(nxn)]
      joined = torch.cat([sequence, expanded], dim=2)
      # logit is now brought back to original batch_size x seq_len x hidden_dim
      logit = self.activation(self.W_a(joined))
      # v_a brings to batch_size x seq_len x 1, and squeeze completes the job
      scores = self.v_a(logit).squeeze(2)

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


class DoubleAttention(nn.Module):
    """
    Compute a double dot product attention. While the typical attention
    takes in a 3-dim sequence and 2-dim condition this module takes in two
    inputs that are both 3-dim and a hidden state as the 2-dim condition

    Initialization Args:
        act: which non-linearity to use to compute attention
          (sigmoid for independent attention, softmax for joint attention)

    Input:
        hidden: vector used to compute attentions which acts as a gate
          shape is batch_size x hidden_dim
        slots: 3-dimensional tensor of batch_size x num_items x hidden_dim
        keys: 3-dimensional tensor of batch_size x num_items x hidden_dim
          Note that the slot and key shapes are inter-changeable

    Output:
        weights: weighted context representing the chosen slots
        normalized: attention distribution over memory slots
        scores: raw activation scores based on hidden states
    """
    def __init__(self, act="softmax"):
      super(DoubleDotAttention, self).__init__()
      if act == "softmax":
        self.activation = nn.Softmax(dim=1)
      elif act == "sigmoid":
        self.activation = nn.Sigmoid()

    def forward(hidden, slots, keys):
      scores = self.score(hidden, slots, keys)
      activated = self.activation(scores, dim=1)
      weights = torch.bmm(activated.unsqueeze(1), slots).squeeze(1)
      return weights, activated, scores

    def score(self, hidden, input_1, input_2):
      hidden_ = hidden.unsqueeze(1).expand_as(input_1)
      logit_1 = (input_1 * hidden_).sum(2)
      logit_2 = (input_2 * hidden_).sum(2)
      return logit_1 + logit_2