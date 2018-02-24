from torch import optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import NLLLoss, parameter

import utils.internal.vocabulary as vocab
import utils.internal.data_io as data_io

import torch
import random
import pdb

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
    encoder_optimizer = optim.Adam(enc_params, lr * 0.01, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(dec_params, lr * 0.01, weight_decay=weight_decay)
  else:
    encoder_optimizer = optim.RMSprop(enc_params, lr, weight_decay)
    decoder_optimizer = optim.RMSprop(dec_params, lr, weight_decay)
  return encoder_optimizer, decoder_optimizer

def smart_variable(tensor, is_var=False):
  # If the "tensor" is actually already a variable, then no need to update
  result = tensor if is_var else Variable(tensor)
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

def choose_model(model_type, vocab_size, hidden_size, method, n_layers, drop_prob, max_length):
  if model_type == "basic":
    from model.encoders import RNN_Encoder
    from model.decoders import RNN_Decoder
    encoder = RNN_Decoder
    decoder = RNN_Decoder
  elif model_type == "gru":
    from model.encoders import GRU_Encoder
    from model.decoders import GRU_Decoder
    encoder = GRU_Encoder
    decoder = GRU_Decoder
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
  elif model_type == "memory":
    from model.encoders import Memory_Encoder
    from model.decoders import Memory_Decoder
    encoder = Memory_Encoder
    decoder = Memory_Decoder
    decoder.embedding.weight = encoder.embedding.weight

  return encoder, decoder

def run_inference(encoder, decoder, sources, targets, criterion, teach_ratio):
  loss = 0
  encoder_hidden = encoder.initHidden()
  encoder_length = sources.size()[0]
  encoder_outputs, encoder_hidden = encoder(sources, encoder_hidden)

  decoder_hidden = encoder_hidden
  decoder_length = targets.size()[0]
  decoder_input = smart_variable(torch.LongTensor([[vocab.SOS_token]]))
  decoder_context = smart_variable(torch.zeros(1, 1, decoder.hidden_size))

  visual = torch.zeros(encoder_length, decoder_length)
  predictions = []
  for di in range(decoder_length):
    use_teacher_forcing = random.random() < teach_ratio
    if decoder.expand_params:
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs,
        sources, targets, di, use_teacher_forcing)
    else:
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
          decoder_input, decoder_context, decoder_hidden, encoder_outputs)
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
      # pdb.set_trace()
      query_tokens = [vocab.index_to_word(q[0], task) for q in queries]
      response_tokens = [vocab.index_to_word(r, task) for r in responses]

      visualizations.append((visual, query_tokens, response_tokens))

  return visualizations