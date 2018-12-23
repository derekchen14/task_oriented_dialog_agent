import os
import random
import logging
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from objects.components import device
from objects.learn import encoders as enc
from objects.learn import decoders as dec
from objects.learn.modules import GlobalLocalModel, BasicClassifer
# from objects.learn.belief_tracker import BeliefTracker
# from objects.learn.policy_manager import PolicyManager
# from objects.learn.text_generator import TextGenerator

class Builder(object):
  def __init__(self, args, loader=None):
    self.args = args
    self.use_existing = args.use_existing
    self.model_type = args.model
    self.dhid = args.hidden_size
    self.test_mode = args.test_mode

    self.loader = loader
    self.data_dir = loader.data_dir
    self.embeddings = loader.embeddings if args.pretrained else None

  def get_model(self, vocab_size, output_size=None, task=None):
    if output_size is None:
      output_size = vocab_size # word generation rather than classification
    self.prepare_directory(self.args, task)

    model = self.create_model(vocab_size, output_size)
    if self.test_mode:
      logging.info("Loading model at {} for testing".format(self.dir))
      if self.model_type == "glad":
        model.load_best_save(directory=self.dir)
      else:
        model = self.load_model(self.dir, model)
    elif self.use_existing:
      logging.info("Resuming model at {} for training".format(self.dir))
      model = self.load_model(self.dir, model)
    else:
      logging.info("Building model at {}".format(self.dir))
      model.save_dir = self.dir

    return model.to(device)

  def prepare_directory(self, args, task):
    if task is None:
      self.model_path = args.prefix + args.model + args.suffix
    else:
      self.model_path = task + "_" + args.prefix + args.model + args.suffix
    self.dir = os.path.join("results", args.task, args.dataset, self.model_path)
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)

  def load_model(self, directory, model):
    filename = "epoch=12_success=25.4042_recall@two=41.3395"
    state = torch.load("{0}/{1}.pt".format(directory, filename))
    model.load_state_dict(state['model'])
    model.eval()
    if self.use_existing:
      model.init_optimizer()
      model.optimizer.load_state_dict(state['optimizer'])
    return model

  def create_model(self, vocab_size, output_size):
    if self.model_type == "basic":
      encoder = enc.RNN_Encoder(vocab_size, self.args)
      ff_network = dec.FF_Network(self.dhid, output_size, self.model_type)
      return BasicClassifer(encoder, ff_network, self.args)
    if self.model_type in ["bilstm", "dual", "per_slot"]:
      encoder = enc.BiLSTM_Encoder(vocab_size, self.embeddings, self.args)
      ff_network = dec.FF_Network(self.dhid, output_size, self.model_type)
      return BasicClassifer(encoder, ff_network, self.args)
    elif self.model_type == "glad":
      glad_model = GlobalLocalModel(self.args, self.loader.ontology,
                    self.loader.vocab, self.embeddings, enc.GLADEncoder)
      glad_model.save_config(self.dir)
      return glad_model.to(device)
    elif self.model_type == "attention":
      encoder = enc.GRU_Encoder(vocab_size, self.args)
      decoder = dec.Attn_Decoder(output_size, self.args)
    elif self.model_type == "bidirectional":
      encoder = enc.Bid_Encoder(vocab_size, self.dhid)
      decoder = dec.Bid_Decoder(output_size, self.args)
    elif self.model_type == "copy":
      encoder = enc.Match_Encoder(vocab_size, self.dhid)
      decoder = dec.Copy_Without_Attn_Decoder(output_size, self.args)
      zeros_tensor = torch.zeros(vocab_size, max_length=25)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
    elif self.model_type == "combined":
      encoder = enc.Match_Encoder(vocab_size, self.dhid)
      decoder = dec.Copy_Decoder(output_size, self.args)
      zeros_tensor = torch.zeros(vocab_size, max_length=25)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
    elif self.model_type == "transformer":
      encoder = enc.Transformer_Encoder(vocab_size, self.args)
      decoder = dec.Transformer_Decoder(output_size, self.args)

    return Seq2Seq(encoder, decoder)


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