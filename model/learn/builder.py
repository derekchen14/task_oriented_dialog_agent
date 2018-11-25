import torch
from torch.nn.parameter import Parameter
from torch import optim

from model.components import device
from model.learn import encoders as enc
from model.learn import decoders as dec
from model.learn import modules

class Builder(object):
  def __init__(self, args):
    self.model_type = args.model_type
    self.hidden_size = args.hidden_size
    self.method = args.attn_method
    self.n_layers = args.n_layers
    self.drop_prob = args.drop_prob
    self.optimizer = args.optimizer
    self.weight_decay = args.weight_decay
    self.lr = args.learning_rate

  def build_model(self, vocab_size, output_size=None, max_length=25):
    if output_size is None:
      output_size = vocab_size # word generation rather than classification

    if self.model_type == "basic":
      encoder = enc.RNN_Encoder
      decoder = dec.RNN_Decoder
    elif self.model_type == "gru":
      encoder = enc.GRU_Encoder(vocab_size, self.hidden_size, self.n_layers)
      decoder = dec.GRU_Decoder(output_size, self.hidden_size, self.n_layers)
    elif self.model_type == "lstm":
      encoder = enc.LSTM_Encoder(vocab_size, self.hidden_size, self.n_layers)
      decoder = dec.FF_Network(self.hidden_size, output_size)
    elif self.model_type == "attention":
      encoder = enc.GRU_Encoder(vocab_size, self.hidden_size, self.n_layers)
      decoder = dec.Attn_Decoder(output_size, self.hidden_size, self.method, self.drop_prob)
    elif self.model_type == "bidirectional":
      encoder = enc.Bid_Encoder(vocab_size, self.hidden_size)
      decoder = dec.Bid_Decoder(output_size, self.hidden_size, self.method, self.drop_prob)
    elif self.model_type == "copy":
      encoder = enc.Match_Encoder(vocab_size, self.hidden_size)
      decoder = dec.Copy_Without_Attn_Decoder(output_size, self.hidden_size,
                    self.method, self.drop_prob)
      zeros_tensor = torch.zeros(vocab_size, max_length)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
    elif self.model_type == "combined":
      encoder = enc.Match_Encoder(vocab_size, self.hidden_size)
      decoder = dec.Copy_Decoder(output_size, self.hidden_size, self.method,
                    self.drop_prob, max_length)
      zeros_tensor = torch.zeros(vocab_size, max_length)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
    elif self.model_type == "transformer":
      encoder = enc.Transformer_Encoder(vocab_size, self.hidden_size, self.n_layers)
      decoder = dec.Transformer_Decoder(output_size, self.hidden_size, self.n_layers)
    elif self.model_type == "replica":
      encoder = enc.Replica_Encoder(vocab_size, self.hidden_size)
      decoder = dec.Replica_Decoder(output_size, self.hidden_size, self.method,
                    self.drop_prob, max_length)
      zeros_tensor = torch.zeros(vocab_size, max_length)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))

    return encoder.to(device), decoder.to(device)

  def init_optimizers(self, encoder, decoder):
    enc_params, dec_params = encoder.parameters(), decoder.parameters()

    if self.optimizer == 'SGD':
      enc_optimizer = optim.SGD(enc_params, self.lr, self.weight_decay)
      dec_optimizer = optim.SGD(dec_params, self.lr, self.weight_decay)
    elif self.optimizer == 'Adam':
      # warmup = step_num * math.pow(4000, -1.5)
      # self.lr = (1 / math.sqrt(d)) * min(math.pow(step_num, -0.5), warmup)
      self.lr = 0.0158
      enc_optimizer = optim.Adam(enc_params, self.lr, betas=(0.9, 0.98), eps=1e-9)
      dec_optimizer = optim.Adam(dec_params, self.lr, betas=(0.9, 0.98), eps=1e-9)
      # encoder_optimizer = optim.Adam(enc_params, self.lr * 0.01, weight_decay=weight_decay)
      # decoder_optimizer = optim.Adam(dec_params, self.lr * 0.01, weight_decay=weight_decay)
    else:
      enc_optimizer = optim.RMSprop(enc_params, self.lr, self.weight_decay)
      dec_optimizer = optim.RMSprop(dec_params, self.lr, self.weight_decay)

    return enc_optimizer, dec_optimizer