from torch import optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import utils.internal.vocabulary as vocab

import torch
import random

use_cuda = cuda.is_available()

def starting_checkpoint(iteration):
  if iteration == 1:
    if use_cuda:
      print("Starting to train on GPUs ... ")
    else:
      print("Start local CPU training ... ")

def init_optimizers(optimizer_type, enc_params, dec_params, lr, weight_decay):
  if optimizer_type == 'SGD':
    encoder_optimizer = optim.SGD(enc_params, lr, weight_decay)
    decoder_optimizer = optim.SGD(dec_params, lr, weight_decay)
  elif optimizer_type == 'Adam':
    encoder_optimizer = optim.Adam(enc_params, lr * 0.01, weight_decay)
    decoder_optimizer = optim.Adam(dec_params, lr * 0.01, weight_decay)
  else:
    encoder_optimizer = optim.RMSprop(enc_params, lr, weight_decay)
    decoder_optimizer = optim.RMSprop(dec_params, lr, weight_decay)
  return encoder_optimizer, decoder_optimizer

def smart_variable(tensor):
  result = Variable(tensor)
  if use_cuda:
    return result.cuda()
  else:
    return result

def clip_gradient(models, clip):
  '''
  models: a list, such as [encoder, decoder]
  clip: amount to clip the gradients by
  '''
  if clip is None:
    return
  for model in models:
    clip_grad_norm(model.parameters(), clip)

def run_inference(encoder, decoder, sources, targets, criterion, teach_ratio):
  loss = 0
  encoder_hidden = encoder.initHidden()
  encoder_outputs, encoder_hidden = encoder(sources, encoder_hidden)

  decoder_hidden = encoder_hidden
  decoder_input = smart_variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_context = smart_variable(torch.zeros(1, 1, decoder.hidden_size))

  predictions = []
  for di in range(targets.size()[0]):
    decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs)
    loss += criterion(decoder_output, targets[di])

    if random.random() < teach_ratio:   # Use teacher forcing
      decoder_input = targets[di]
    else:       # Use the predicted word as the next input
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      predictions.append(ni)
      if ni == vocab.EOS_token:
        break
      decoder_input = smart_variable(torch.LongTensor([[ni]]))

  return loss, predictions