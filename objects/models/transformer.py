import numpy as np
from torch import nn

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

class TransformerXL(nn.Module):
  '''
  Able to handle long-term recurrent connections
  '''

  def __init__(self, vocab_size):
    pass