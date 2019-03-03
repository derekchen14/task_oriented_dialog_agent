import os, pdb, sys  # set_trace
import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from objects.components import device
from objects.models import encoders as enc
from objects.models import decoders as dec
from objects.models import *
from objects.modules import *

class Builder(object):
  def __init__(self, args, loader):
    self.args = args
    self.use_existing = args.use_existing
    self.model_type = args.model
    self.dhid = args.hidden_dim
    self.test_mode = args.test_mode

    self.loader = loader
    self.data_dir = loader.data_dir
    self.embeddings = loader.embeddings if args.pretrained else None

  def get_model(self, processor, monitor, model_type=None):
    self.prepare_directory()
    model = self.create_model(processor, model_type)
    monitor.build_logger(self.dir)

    if self.test_mode:
      monitor.logger.info("Loading model at {} for testing".format(self.dir))
      # if self.model_type == "glad":
      #   model.load_best_save(directory=self.dir)
      model = self.load_model(self.dir, model)
    elif self.use_existing:
      monitor.logger.info("Resuming model at {} for training".format(self.dir))
      model = self.load_model(self.dir, model)
    else:
      monitor.logger.info("Building model at {}".format(self.dir))
      model.save_dir = self.dir
    return model

  def add_loss_function(self, model, function_type):
    if function_type == "negative_log_likelihood":
      model.criterion = nn.NLLLoss()
    elif function_type == "binary_cross_entropy":
      model.criterion = nn.BCELoss()
    else:
      raise(ValueError, "loss function is not valid")
    return model

  def prepare_directory(self):
    model_path = self.args.prefix + self.args.model + self.args.suffix
    self.dir = os.path.join("results", self.args.task, self.args.dataset, model_path)
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

  def create_model(self, processor, model_type):
    input_size, output_size = processor.input_output_cardinality()
    if model_type is None:
      model_type = self.model_type

    if model_type == "basic":
      encoder = enc.RNN_Encoder(input_size, self.args)
      ff_network = dec.FF_Network(self.dhid, output_size, model_type)
      model = BasicClassifer(encoder, ff_network, self.args)
      model = self.add_loss_function(model, "negative_log_likelihood")
    if model_type in ["bilstm", "dual", "per_slot"]:
      encoder = enc.BiLSTM_Encoder(input_size, self.embeddings, self.args)
      ff_network = dec.FF_Network(self.dhid, output_size, model_type)
      model = BasicClassifer(encoder, ff_network, self.args)
      model = self.add_loss_function(model, "negative_log_likelihood")
    elif model_type == "glad":
      model = GLAD(self.args, self.loader.ontology,
                    self.loader.vocab, self.embeddings, enc.GLAD_Encoder)
      model = self.add_loss_function(model, "binary_cross_entropy")
    elif model_type == "rulebased":
      model = HackPolicy(processor.ontology)
      mmodel_type = model_type
    elif model_type == "attention":
      encoder = enc.GRU_Encoder(input_size, self.args)
      decoder = dec.Attn_Decoder(output_size, self.args)
      model = Seq2Seq(encoder, decoder)
    elif model_type == "copy":
      encoder = enc.Match_Encoder(input_size, self.dhid)
      decoder = dec.Copy_Without_Attn_Decoder(output_size, self.args)
      zeros_tensor = torch.zeros(input_size, max_length=25)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
      model = Seq2Seq(encoder, decoder)
    elif model_type == "combined":
      encoder = enc.Match_Encoder(input_size, self.dhid)
      decoder = dec.Copy_Decoder(output_size, self.args)
      zeros_tensor = torch.zeros(input_size, max_length=25)
      copy_tensor = [zeros_tensor, encoder.embedding.weight.data]
      decoder.embedding.weight = Parameter(torch.cat(copy_tensor, dim=1))
      model = Seq2Seq(encoder, decoder)
    elif model_type == "transformer":
      encoder = enc.Transformer_Encoder(input_size, self.args)
      decoder = dec.Transformer_Decoder(output_size, self.args)
      model = Seq2Seq(encoder, decoder)
    elif model_type == 'belief_tracker':
      results_dir = os.path.join("results", self.args.task, self.args.dataset)
      model = NLU(self.loader, results_dir)
      model.load_nlu_model('nlu_1468447442')
      model.module_type = model_type
    elif model_type == 'policy_manager':
      movie_kb = self.loader.json_data('movie_kb.1k')
      act_set = self.loader.text_data('dia_acts')
      slot_set = self.loader.text_data('slot_set')
      goal_set = self.loader.json_data('goal_set')
      self.ontology_sets = (act_set, slot_set, goal_set, movie_kb)
      model = AgentDQN(vars(self.args), movie_kb, act_set, slot_set)
      model.module_type = model_type
    elif model_type == 'text_generator':
      results_dir = os.path.join("results", self.args.task, self.args.dataset)
      model = NLG(self.loader, results_dir)
      model.load_nlg_model('nlg_1468202263')
      nl_pairs = self.loader.json_data('dia_act_nl_pairs.v6')
      model.load_predefine_act_nl_pairs(nl_pairs)
      model.module_type = model_type

    return model

  def configure_module(self, args, model):
    if model.module_type == 'policy_manager':
      kb, ontology = self.loader.kb, self.loader.ontology
      if args.model == 'rulebased':
        module = RulebasedPolicyManager(args, model, kb, ontology)
        results_dir = os.path.join('results', args.dataset, 'models')
        module.user.text_generator = TextGenerator.from_pretrained(args)
        module.user.text_generator.set_templates(args.dataset)
      elif args.model == 'ddq':
        movie_dictionary = self.loader.json_data('dicts.v3')
        act_set, slot_set, goal_set, movie_kb = self.ontology_sets
        user_sim = RuleSimulator(vars(args), movie_dictionary, act_set, slot_set, goal_set)
        world_model = ModelBasedSimulator(vars(args),
              movie_dictionary, act_set, slot_set, goal_set)
        model.set_user_planning(world_model)
        module = DialogManager(model, user_sim, world_model,
              act_set, slot_set, movie_kb)
        # module = NeuralPolicyManager(args, model, kb, ontology)
    elif model.module_type == 'belief_tracker':
      module = model
    elif model.module_type  == 'text_generator':
      module = model
    else:
      module = model.to(device)

    module.dir = self.dir
    return module

  def create_agent(self, belief_tracker, policy_manager, text_generator):
    agent = policy_manager

    agent.model.set_nlu_model(belief_tracker)
    agent.user_sim.set_nlu_model(belief_tracker)
    agent.world_model.set_nlu_model(belief_tracker)

    agent.model.set_nlg_model(text_generator)
    agent.user_sim.set_nlg_model(text_generator)
    agent.world_model.set_nlg_model(text_generator)

    return agent