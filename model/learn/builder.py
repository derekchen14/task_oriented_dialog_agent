import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim

from model.components import device
from model.learn import encoders as enc
from model.learn import decoders as dec
from model.learn import modules
# from model.learn.belief_tracker import BeliefTracker
# from model.learn.policy_manager import PolicyManager
# from model.learn.text_generator import TextGenerator

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

  def create_model(self, vocab_size, output_size=None, max_length=25):
    if output_size is None:
      output_size = vocab_size # word generation rather than classification

    elif self.model_type in ["basic", "dual", "per_slot"]:
      encoder = enc.LSTM_Encoder(vocab_size, self.hidden_size, self.n_layers)
      ff_network = dec.FF_Network(self.hidden_size, output_size)
      return BasicClassifer(encoder, ff_network)
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

    return Seq2Seq(encoder, decoder)

  def init_optimizers(self, model):
    enc_params, dec_params = model.encoder.parameters(), model.decoder.parameters()

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

  def make_system(self, input_size, *output_size):
    models = [self.create_model(input_size, outs) for outs in output_size]
    return models[0] if len(models) == 1 else models


class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder.to(device)
    self.decoder = decoder.to(device)
    self.type = "seq2seq"
    self.arguments_size = decoder.arguments_size

  def flatten_parameters(self):
    self.encoder.rnn.flatten_parameters()
    self.decoder.rnn.flatten_parameters()

  def forward(self, sources, targets, enc_hidden, enc_length, dec_length, track):
    loss, predictions, visual, teach_ratio = track
    self.flatten_parameters()
    encoder_outputs, enc_hidden = self.encoder(sources, enc_hidden)

    dec_hidden = enc_hidden
    decoder_input = var([[vocab.SOS_token]], "long")
    decoder_context = var(torch.zeros(1, 1, decoder.hidden_size))

    for di in range(dec_length):
      use_teacher_forcing = random.random() < teach_ratio
      if self.arguments_size == "large":
        dec_output, dec_context, dec_hidden, attn_weights = self.decoder(
          decoder_input, dec_context, dec_hidden, encoder_outputs,
          sources, targets, di, use_teacher_forcing)
      elif self.arguments_size == "medium":
        dec_output, dec_context, dec_hidden, attn_weights = self.decoder(
            decoder_input, dec_context, dec_hidden, encoder_outputs)
      elif self.arguments_size == "small":
        dec_output, dec_context = self.decoder(decoder_input, dec_context)
        attn_weights, visual = False, False

      # visual[:, di] = attn_weights.squeeze(0).squeeze(0).cpu().data
      loss += criterion(dec_output, targets[di])

      if use_teacher_forcing:
        decoder_input = targets[di]
      else:       # Use the predicted word as the next input
        topv, topi = dec_output.data.topk(1)
        ni = topi[0][0]
        predictions.append(ni)
        if ni == vocab.EOS_token:
          break
        decoder_input = var([[ni]], "long")

    return loss, predictions, visual


class BasicClassifer(nn.Module):
  def __init__(self, encoder, ff_network):
    super(BasicClassifer, self).__init__()
    self.encoder = encoder.to(device)
    self.decoder = ff_network.to(device)
    self.type = "basic"
  def forward(self, sources, hidden):
    self.encoder.rnn.flatten_parameters()
    encoder_outputs, hidden = self.encoder(sources, hidden)
    return self.decoder(encoder_outputs[0])
