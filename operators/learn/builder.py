import os
import random
import logging
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from objects.models import encoders as enc
from objects.models import decoders as dec
from objects.models import Transformer, EntNet, NPN , GLAD, Seq2Seq
from objects.modules import *
from objects.components import device

class Builder(object):
  def __init__(self, args, data_loader):
    self.args = args
    self.use_existing = args.use_existing
    self.model_type = args.model
    self.dhid = args.hidden_dim
    self.test_mode = args.test_mode

    self.data_loader = data_loader
    self.data_dir = data_loader.data_dir
    self.embeddings = data_loader.embeddings if args.pretrained else None

  def set_module(self, args):
    pass

  def get_model(self, processor, task=None):
    input_size, output_size = processor.input_output_cardinality()
    self.prepare_directory(self.args, task)

    model = self.create_model(input_size, output_size)
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

  def create_model(self, input_size, output_size):
    if self.model_type == "basic":
      encoder = enc.RNN_Encoder(input_size, self.args)
      ff_network = dec.FF_Network(self.dhid, output_size, self.model_type)
      return BasicClassifer(encoder, ff_network, self.args)
    if self.model_type in ["bilstm", "dual", "per_slot"]:
      encoder = enc.BiLSTM_Encoder(input_size, self.embeddings, self.args)
      ff_network = dec.FF_Network(self.dhid, output_size, self.model_type)
      return BasicClassifer(encoder, ff_network, self.args)
    elif self.model_type == "glad":
      glad_model = GLAD(self.args, self.data_loader.ontology,
                    self.data_loader.vocab, self.embeddings, enc.GLAD_Encoder)
      glad_model.save_config(self.dir)
      return glad_model.to(device)
    elif self.model_type == "rulebased":
      return InformPolicy(self.args)
    elif self.model_type == "attention":
      encoder = enc.GRU_Encoder(input_size, self.args)
      decoder = dec.Attn_Decoder(output_size, self.args)
    elif self.model_type == "copy":
      encoder = enc.Match_Encoder(input_size, self.dhid)
      decoder = dec.Copy_Without_Attn_Decoder(output_size, self.args)
      zeros_tensor = torch.zeros(input_size, max_length=25)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
    elif self.model_type == "combined":
      encoder = enc.Match_Encoder(input_size, self.dhid)
      decoder = dec.Copy_Decoder(output_size, self.args)
      zeros_tensor = torch.zeros(input_size, max_length=25)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
    elif self.model_type == "transformer":
      encoder = enc.Transformer_Encoder(input_size, self.args)
      decoder = dec.Transformer_Decoder(output_size, self.args)

    return Seq2Seq(encoder, decoder)

  def configure_module(self, args, model):
    return model