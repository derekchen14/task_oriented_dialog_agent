from torch.nn.utils import clip_grad_norm_
from torch.nn.utils import rnn as rnn_utils
from torch.nn import NLLLoss, parameter

import utils.internal.vocabulary as vocab
import utils.internal.data_io as data_io
from utils.external.bleu import BLEU

import torch
import numpy as np
import pdb, sys
from tqdm import tqdm as progress_bar

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
  clip_grad_norm_(model.encoder.parameters(), clip)
  clip_grad_norm_(model.decoder.parameters(), clip)

def run_inference(model, sources, targets, criterion, teach_ratio):
  if model.type in ["basic", "dual", "multi"]:
    return basic_inference(model, sources, targets, criterion)
  elif model.type == "transformer":
    return transformer_inference(model, sources, targets, criterion)
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

def basic_inference(model, sources, targets, criterion):
  hidden = model.encoder.initHidden()
  output = model(sources, hidden)
  topv, topi = output.data.topk(2)
  pred = topi[0]  # returns full list of predictions as a tensor
  # pred = topi[0][0] # instead to select just the first

  if criterion is None:
    loss = 0
  else:
    try:
      loss = criterion(output, targets)
    except(RuntimeError):
      print(output)
      print(targets)
      pdb.set_trace()
  # loss = 0 if criterion is None else criterion(output, targets)
  return loss, pred, None

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