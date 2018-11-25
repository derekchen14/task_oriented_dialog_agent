from torch.nn.utils import clip_grad_norm_
from torch.nn import NLLLoss, parameter

import utils.internal.vocabulary as vocab
import utils.internal.data_io as data_io
from utils.external.bleu import BLEU

import torch
import random
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

def clip_gradient(models, clip):
  '''
  models: a list, such as [encoder, decoder]
  clip: amount to clip the gradients by
  '''
  if clip is None:
    return
  for model in models:
    clip_grad_norm_(model.parameters(), clip)

def run_inference(encoder, decoder, sources, targets, criterion, teach_ratio):
  if decoder.arguments_size == "extra_large":
    return transformer_inference(encoder, decoder, sources, targets, criterion)
  if decoder.arguments_size == "tiny":
    return basic_inference(encoder, decoder, sources, targets, criterion)

  loss = 0
  encoder_hidden = encoder.initHidden()
  encoder_length = sources.size()[0]
  cuda_sources = smart_variable(sources, dtype="var")
  encoder_outputs, encoder_hidden = encoder(cuda_sources, encoder_hidden)

  decoder_hidden = encoder_hidden
  decoder_length = targets.size()[0]
  targets = smart_variable(targets, dtype="var")
  decoder_input = smart_variable([[vocab.SOS_token]], "list")
  decoder_context = smart_variable(torch.zeros(1, 1, decoder.hidden_size))

  visual = torch.zeros(encoder_length, decoder_length)
  predictions = []
  for di in range(decoder_length):
    use_teacher_forcing = random.random() < teach_ratio
    if decoder.arguments_size == "large":
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
        decoder_input, decoder_context, decoder_hidden, encoder_outputs,
        cuda_sources, targets, di, use_teacher_forcing)
    elif decoder.arguments_size == "medium":
      decoder_output, decoder_context, decoder_hidden, attn_weights = decoder(
          decoder_input, decoder_context, decoder_hidden, encoder_outputs)
    elif decoder.arguments_size == "small":
      decoder_output, decoder_context = decoder(decoder_input, decoder_context)
      attn_weights, visual = False, False

    # visual[:, di] = attn_weights.squeeze(0).squeeze(0).cpu().data
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

def basic_inference(encoder, decoder, sources, targets, criterion):
  encoder_hidden = encoder.initHidden()
  encoder_outputs, encoder_hidden = encoder(sources, encoder_hidden)

  decoder_output = decoder(encoder_outputs[0])
  topv, topi = decoder_output.data.topk(1)
  pred = topi[0][0]

  loss = criterion(decoder_output, targets)
  return loss, pred, None

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

    topv, topi = decoder_output[di].data.topk(1)
    pdb.set_trace()
    ni = topi[0][0]
    predictions.append(ni)
    if ni == vocab.EOS_token:
      break

  return loss, predictions, None

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
