from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import rnn as rnn_utils
from torch.nn import NLLLoss, parameter

import utils.internal.vocabulary as vocab
import utils.internal.initialization as data_io
from utils.external.bleu import BLEU

import torch
import numpy as np
import os, pdb, sys
import re
from tqdm import tqdm as progress_bar

use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

def var(data, dtype="float"):
  if dtype == "float":
    result = torch.Tensor(data)
  elif dtype == "long":
    result = torch.LongTensor(data)
  elif dtype == "variable":
    result = data
  return result.to(device)

def clip_gradient(model, clip):
  if clip is None: return
  try:
    clip_grad_norm_(model.encoder.parameters(), clip)
    clip_grad_norm_(model.decoder.parameters(), clip)
  except(AttributeError):
    pass

def get_saves(director, early_stop):
  files = [f for f in os.listdir(directory) if f.endswith('.pt')]
  scores = []
  for fname in files:
    re_str = r'dev_{}=([0-9\.]+)'.format(early_stop)
    dev_acc = re.findall(re_str, fname)
    if dev_acc:
      score = float(dev_acc[0].strip('.'))
      scores.append((score, os.path.join(directory, fname)))
  if not scores:
    raise Exception('No files found!')
  scores.sort(key=lambda tup: tup[0], reverse=True)
  return scores

def run_inference(model, batch):
  if model.model_type in ["basic", "dual", "multi"]:
    return basic_inference(model, sources, targets)
  elif model.model_type == "transformer":
    return transformer_inference(model, sources, targets)
  elif model.model_type == "glad":
    return model.forward(batch)
  else:
    assert(model.type == "seq2seq")

  sources = var(sources, dtype="variable")
  targets = var(targets, dtype="variable")
  enc_hidden = model.encoder.initHidden()
  enc_length = sources.shape[0]
  dec_length = targets.shape[0]

  loss = 0
  predictions = []
  visual = torch.zeros(enc_length, dec_length)
  track = loss, predictions, visual, teach_ratio

  return model(sources, targets, enc_hidden, enc_length, dec_length, track)

def basic_inference(model, batch):
  hidden = model.encoder.initHidden()
  output = model(sources, hidden)
  topv, topi = output.data.topk(2)
  pred = topi[0]  # returns full list of predictions as a tensor
  # pred = topi[0][0] # instead to select just the first
  loss = model.criterion(output, targets)
  return loss, pred

def transformer_inference(model, sources, targets, criterion):
  loss = 0
  predictions = []
  encoder_outputs = model.encoder(var(sources, dtype="variable"))
  decoder_start = var([[vocab.SOS_token]], "list")
  decoder_tokens = var(targets, dtype="variable")
  decoder_inputs = torch.cat([decoder_start, decoder_tokens], dim=0)

  for di in range(targets.size()[0]):
    decoder_output = model.decoder(decoder_inputs, encoder_outputs, di)
    # we need to index into the output now since output is (seq_len, vocab)
    loss += criterion(decoder_output[di].view(1,-1), decoder_tokens[di])

    topv, topi = decoder_output[di].data.topk(1)
    pdb.set_trace()
    ni = topi[0][0]
    predictions.append(ni)
    if ni == vocab.EOS_token:
      break

  return loss, predictions, None

def run_rnn(rnn, inputs, lens):
    # sort by lens, argsort gives smallest to largest, [::-1] gives largest to smallest
    order = np.argsort(lens)[::-1].tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = rnn_utils.pack_padded_sequence(reindexed, reindexed_lens, batch_first=True)
    # print("packed: {}".format(packed.shape))
    outputs, _ = rnn(packed)
    # print("outputs: {}".format(outputs.shape))
    # pdb.set_trace()
    padded, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True, padding_value=0.)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
    # reindexed_lens = [lens[i] for i in order]
    # recovered_lens = [reindexed_lens[i] for i in reverse_order]
    # assert recovered_lens == lens
    return recovered

def unique_identifier(summary, epoch, iteration, early_stop_metric):
  uid = 'epoch={epoch},iter={iter},train_{key}={train:.4f},dev_{key}={dev:.4f}'.format(
          epoch=epoch, iter=iteration, key=early_stop_metric,
          train=summary.get(f'train_{early_stop_metric}', 0.0),
          dev=summary[f'best_{early_stop_metric}'] )
  return uid

def show_dialogues(val_data, encoder, decoder, task):
  encoder.eval()
  decoder.eval()
  dialogues = data_io.select_consecutive_pairs(val_data, 5)

  for i, dialog in enumerate(dialogues):
    print("Dialogue Sample {} ------------".format(i))
    for j, turn in enumerate(dialog):
      input_variable, output_variable = turn
      _, predictions, _ = run_inference(encoder, decoder, input_variable, \
                      output_variable, criterion=NLLLoss(), teach_ratio=0)
      sources = input_variable.data.tolist()
      targets = output_variable.data.tolist()

      source_tokens = [vocab.index_to_word(s[0], task) for s in sources]
      target_tokens = [vocab.index_to_word(t[0], task) for t in targets]
      pred_tokens = [vocab.index_to_word(p, task) for p in predictions]

      source = " ".join(source_tokens[:-1]) # Remove the <EOS>
      target = " ".join(target_tokens[:-1])
      pred = " ".join(pred_tokens[:-1])
      print("User Query: {0}".format(source))
      print("Target Response: {0}".format(target))
      print("Predicted Response: {0}".format(pred))
    print('')

def match_embedding(vocab_size, hidden_size):
  match_tensor = torch.load('datasets/restaurants/match_features.pt')
  embed = torch.nn.Embedding(vocab_size, hidden_size)
  # Extract just the tensor inside the Embedding
  embed_tensor = embed.weight.data
  extended_tensor = torch.cat([embed_tensor, match_tensor], dim=1)
  # Set the weight of original embedding matrix with the new Parameter
  embed.weight = torch.nn.parameter.Parameter(extended_tensor)
  return embed
