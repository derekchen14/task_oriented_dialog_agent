import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb  # set_trace

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
      self.v_a = nn.Parameter(torch.FloatTensor(1, hidden_size))
    elif self.attn_method == 'dot':                 # h_j x h_i
      self.W_a = torch.eye(hidden_size) # identity since no extra matrix is needed

  def forward(self, decoder_hidden, encoder_outputs):
    # Create variable to store attention scores           # seq_len = batch_size
    seq_len = len(encoder_outputs)
    attn_scores = smart_variable(torch.zeros(seq_len))    # B (batch_size)
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
  def __init__(self, hidden_size, n_layers, masked=False):
    super(Transformer, self).__init__()
    self.max_length = 30
    self.num_attention_heads = 8  # hardcoded since it won't change
    self.num_layers = n_layers   # defaults to 6 to follow the paper
    self.positional_embedding = nn.Embedding(self.max_length, hidden_size)
    self.positional_embedding.weight.requires_grad = False
    self.dropout = nn.Dropout(0.2)
    self.masked = masked

    for head_idx in range(self.num_attention_heads):
      for vector_type in ['query', 'key', 'value']:
        head_name = vector_type + "_head_" + head_idx
        mask_name = vector_type + "_mask_" + head_idx
        head_in = self.hidden_size
        head_out = self.hidden_size / self.num_attention_heads
        setattr(self, head_name, nn.Linear(head_in, head_out))
        if masked:
          setattr(self, mask_name, nn.Linear(head_in, head_out))

    self.pw_ffn_1 = nn.Linear(self.input_size, vocab_size)
    self.pw_ffn_2 = nn.Linear(self.input_size, vocab_size)
    self.layernorm = nn.LayerNorm(100)

  def forward(self, transformer_input, encoder_outputs=None, di=None):
    positions = self.positional_embedding[:len(transformer_input), :]
    transformer_input += positions

    for layer_idx in range(self.num_layers):
      if layer_idx > 0:
        transformer_input = self.dropout(transformer_output)

      if self.masked:
        masked_input = self.apply_mask(transformer_input, decoder_idx)
        k_v_inputs = encoder_outputs

        mask_attn_heads = []
        for j in range(self.num_attention_heads):
          Q = getattr(self, "query_mask_{}".format(j))(masked_input)
          K = getattr(self, "key_mask_{}".format(j))(k_v_inputs)
          V = getattr(self, "value_mask_{}".format(j))(k_v_inputs)
          mask_attn_heads.append(self.scaled_dot_product_attention(Q, K, V))
        residual_connection = embedded + torch.stack(mask_attn_heads, dim=1)
        masked_output = self.layernorm(residual_connection)
        transformer_input = self.dropout(masked_output)

      attn_heads = []  # don't create a new variable since it messes with the graph
      for idx in range(self.num_attention_heads):
        Q = getattr(self, "query_head_{}".format(idx))(transformer_input)
        K = getattr(self, "key_head_{}".format(idx))(k_v_inputs)
        V = getattr(self, "value_head_{}".format(idx))(k_v_inputs)
        attn_heads.append(self.scaled_dot_product_attention(Q, K, V))
      residual_connection = embedded + torch.stack(attn_heads, dim=1)
      multihead_output = self.layernorm(residual_connection)

      pw_ffn_output = self.position_wise_ffn(multihead_output)
      transformer_output = self.layernorm(multihead_output + pw_ffn_output)

    return transformer_output

  def apply_mask(self, decoder_inputs, decoder_idx):
    mask = torch.zeros(( decoder_inputs.size() ))
    mask[:, :decoder_idx+1] = 1
    return decoder_inputs * mask

  def positionwise_ffn(self, multihead_output):
    ffn_1_output = F.relu(self.pw_ffn_1(multihead_output))
    ffn_2_output = self.pw_ffn_2(ffn_1_output)
    return ffn_2_output

  def scaled_dot_product_attention(self, Q, K, V):
    # K should be seq_len, hidden_dim to start with, but will be transposed
    scaled_matmul = Q.matmul(K.T) / self.scale_factor   # batch_size x seq_len
    # (batch_size, hidden_dim) x (hidden_dim, seq_len) / broadcast integer
    attn_weights = F.softmax(scaled_matmul, axis=1)
    attn_context = attn_weights.matmul(V)             # batch_size, hidden_dim
    return attn_context

