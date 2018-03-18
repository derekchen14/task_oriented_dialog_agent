from torch import optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import NLLLoss, parameter

import utils.internal.vocabulary as vocab
import utils.internal.data_io as data_io

import torch
import random
import numpy as np
import pdb

use_cuda = cuda.is_available()

def starting_checkpoint(iteration):
  if iteration == 1:
    if use_cuda:
      print("Starting to train on GPUs ... ")
    else:
      print("Start local CPU training ... ")

def init_optimizers(optimizer_type, weight_decay, enc_params, dec_params, lr):
  if optimizer_type == 'SGD':
    encoder_optimizer = optim.SGD(enc_params, lr, weight_decay)
    decoder_optimizer = optim.SGD(dec_params, lr, weight_decay)
  elif optimizer_type == 'Adam':
    # warmup = step_num * math.pow(4000, -1.5)
    # lr = (1 / math.sqrt(d)) * min(math.pow(step_num, -0.5), warmup)
    lr = 0.0158
    encoder_optimizer = optim.Adam(enc_params, lr, betas=(0.9, 0.98), eps=1e-9)
    decoder_optimizer = optim.Adam(dec_params, lr, betas=(0.9, 0.98), eps=1e-9)
    # encoder_optimizer = optim.Adam(enc_params, lr * 0.01, weight_decay=weight_decay)
    # decoder_optimizer = optim.Adam(dec_params, lr * 0.01, weight_decay=weight_decay)
  else:
    encoder_optimizer = optim.RMSprop(enc_params, lr, weight_decay)
    decoder_optimizer = optim.RMSprop(dec_params, lr, weight_decay)
  return encoder_optimizer, decoder_optimizer

def basic_variable(data, dtype="long"):
    if dtype == "long":
        tensor = torch.LongTensor(data)
    elif dtype == "float":
        tensor = torch.FloatTensor(data)
    return Variable(tensor)

def smart_variable(data, dtype="tensor"):
    if dtype == "list":
        result = basic_variable(data)
    elif dtype == "tensor":
        result = Variable(data)
    elif dtype == "var":
        result = data

    return result.cuda() if use_cuda else result

def clip_gradient(models, clip):
  '''
  models: a list, such as [encoder, decoder]
  clip: amount to clip the gradients by
  '''
  if clip is None:
    return
  for model in models:
    clip_grad_norm(model.parameters(), clip)

def choose_model(model_type, vocab_size, hidden_size, method, n_layers, drop_prob, max_length):
  if model_type == "basic":
    from model.encoders import RNN_Encoder
    from model.decoders import RNN_Decoder
    encoder = RNN_Decoder
    decoder = RNN_Decoder
  elif model_type == "gru":
    from model.encoders import GRU_Encoder
    from model.decoders import GRU_Decoder
    encoder = GRU_Encoder(vocab_size, hidden_size, n_layers)
    decoder = GRU_Decoder(vocab_size, hidden_size, n_layers)
  elif model_type == "attention":
    from model.encoders import GRU_Encoder
    from model.decoders import Attn_Decoder
    encoder = Attn_Encoder(vocab_size, hidden_size, n_layers)
    decoder = Attn_Decoder(vocab_size, hidden_size, method, drop_prob)
  elif model_type == "bidirectional":
    from model.encoders import Bid_Encoder
    from model.decoders import Bid_Decoder
    encoder = Bid_Encoder(vocab_size, hidden_size)
    decoder = Bid_Decoder(vocab_size, hidden_size, method, drop_prob)
  elif model_type == "match":
    from model.encoders import Match_Encoder
    from model.decoders import Match_Decoder
    encoder = Match_Encoder(vocab_size, hidden_size)
    decoder = Match_Decoder(vocab_size, hidden_size, method, drop_prob)
    decoder.embedding.weight = encoder.embedding.weight
  elif model_type == "copy":
    from model.encoders import Match_Encoder
    from model.decoders import Copy_Decoder
    encoder = Match_Encoder(vocab_size, hidden_size)
    decoder = Copy_Decoder(vocab_size, hidden_size, method, drop_prob, max_length)
    zeros_tensor = torch.zeros(vocab_size, max_length)
    copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
    decoder.embedding.weight = parameter.Parameter(torch.cat(copy_tensor, dim=1))
  elif model_type == "transformer":
    from model.encoders import Transformer_Encoder
    from model.decoders import Transformer_Decoder
    encoder = Transformer_Encoder(vocab_size, hidden_size, n_layers)
    decoder = Transformer_Decoder(vocab_size, hidden_size, n_layers)
  elif model_type == "memory":
    from model.encoders import Memory_Encoder
    from model.decoders import Memory_Decoder
    encoder = Memory_Encoder
    decoder = Memory_Decoder
    decoder.embedding.weight = encoder.embedding.weight

  return encoder, decoder

def run_inference(encoder, decoder, sources, targets, criterion, teach_ratio):
  if decoder.arguments_size == "transformer":
    return transformer_inference(encoder, decoder, sources, targets, criterion)
  loss = 0
  encoder_hidden = encoder.initHidden()
  encoder_length = sources.size()[0]
  encoder_outputs, encoder_hidden = encoder(sources, encoder_hidden)

  decoder_hidden = encoder_hidden
  decoder_length = targets.size()[0]
  decoder_input = smart_variable([[vocab.SOS_token]], "list")
  decoder_context = smart_variable(torch.zeros(1, 1, decoder.hidden_size))

  visual = torch.zeros(encoder_length, decoder_length)
  predictions = []
  for di in range(decoder_length):
    use_teacher_forcing = random.random() < teach_ratio
    if decoder.arguments_size == "large":
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs,
        sources, targets, di, use_teacher_forcing)
    elif decoder.arguments_size == "medium":
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
          decoder_input, decoder_context, decoder_hidden, encoder_outputs)
    elif decoder.arguments_size == "small":
      decoder_output, decoder_context = decoder(decoder_input, decoder_context)
      attn_weights, visual = False, False

    if attn_weights:
      visual[:, di] = attn_weights.squeeze(0).squeeze(0).cpu().data

    loss += criterion(decoder_output, targets[di])

    if use_teacher_forcing:
      decoder_input = targets[di]
    else:       # Use the predicted word as the next input
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      predictions.append(ni)
      if ni == vocab.EOS_token:
        break
      decoder_input = smart_variable(torch.LongTensor([[ni]]))

  return loss, predictions, visual

def transformer_inference(encoder, decoder, sources, targets, criterion):
  loss = 0
  predictions = []
  encoder_outputs = encoder(smart_variable(sources, dtype="var"))
  decoder_start = smart_variable([[vocab.SOS_token]], "list")
  decoder_tokens = smart_variable(targets, dtype="var")
  decoder_inputs = torch.cat([decoder_start, decoder_tokens], dim=0)

  for di in range(targets.size()[0]):
    decoder_output = decoder(decoder_inputs, encoder_outputs, di)
    # we need to index into the output now since output is (seq_len, vocab)
    loss += criterion(decoder_output[di].view(1,-1), decoder_tokens[di])

    topv, topi = decoder_output.data.topk(1)
    ni = topi[0][0]
    predictions.append(ni)
    if ni == vocab.EOS_token:
      break

  return loss, predictions, None

def grab_attention(val_data, encoder, decoder, task, vis_count):
  encoder.eval()
  decoder.eval()
  dialogues = data_io.select_consecutive_pairs(val_data, vis_count)

  visualizations = []
  for dialog in dialogues:
    for turn in dialog:
      input_variable, output_variable = turn
      _, responses, visual = run_inference(encoder, decoder, input_variable, \
                      output_variable, criterion=NLLLoss(), teach_ratio=0)
      queries = input_variable.data.tolist()
      query_tokens = [vocab.index_to_word(q[0], task) for q in queries]
      response_tokens = [vocab.index_to_word(r, task) for r in responses]

      visualizations.append((visual, query_tokens, response_tokens))

  return visualizations

class LossTracker(object):
  def __init__(self, threshold):
    self.train_steps = []
    self.train_losses = []
    self.train_epoch = 0

    self.val_steps = []
    self.val_losses = []
    self.val_epoch = 0

    self.completed_training = True
    # Minimum loss we are willing to accept for calculating absolute loss
    self.threshold = threshold
    self.absolute_range = 4
    # Trailing average storage for calculating relative loss
    self.trailing_average = []
    self.epochs_per_avg = 3
    self.lookback_range = 2

  def update_loss(self, loss, split):
    if split == "train":
      self.train_losses.append(loss)
      self.train_epoch += 1
    elif split == "val":
      self.val_losses.append(loss)
      self.val_epoch += 1

  def should_early_stop(self):
    if self.threshold < 0:  # we turn off early stopping
      return False

    trail_idx = self.val_epoch - self.epochs_per_avg
    if trail_idx >= 0:
      avg = np.average(self.val_losses[trail_idx : self.val_epoch])
      self.trailing_average.append(float(avg))

      if self._check_absolute(avg) or self._check_relative(avg, trail_idx):
        return True
    # if nothing causes an alarm, then we should continue
    return False

  def _check_absolute(self, current_avg):
    if self.val_epoch == (10 - self.absolute_range):
      if current_avg > (self.threshold * 1.5):
        return True
    elif self.val_epoch == 10:
      if current_avg > self.threshold:
        return True
    elif self.val_epoch == (10 + self.absolute_range):
      if current_avg > (self.threshold * 0.9):
        return True
    return False

  def _check_relative(self, current_avg, trail_idx):
    if self.val_epoch >= (self.epochs_per_avg + self.lookback_range):
      lookback_avg = self.trailing_average[trail_idx - self.lookback_range]
      if (current_avg / lookback_avg) > 1.1:
        return True
    return False
