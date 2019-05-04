import os, pdb, sys  # set_trace
import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from objects.components import device, get_saves
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
    self.use_old_nlu = args.use_old_nlu

    self.loader = loader
    self.data_dir = loader.data_dir
    self.embeddings = loader.embeddings if args.pretrained else None

  def get_model(self, processor, monitor, model_type=None):
    self.prepare_directory()
    model = self.create_model(processor, model_type)
    monitor.build_logger(self.dir)

    model_str = "model" if model_type is None else model_type
    if self.test_mode:
      print("Loading {} at {} for testing".format(model_str, self.dir))
      model = self.load_best_model(self.dir, model, model_type)
    elif self.use_existing:
      print("Resuming {} at {} for training".format(model_str, self.dir))
      model = self.load_best_model(self.dir, model, model_type)
    else:
      print("Building {} at {}".format(model_str, self.dir))

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
      print("Created directory at {}".format(self.dir))

  def load_best_model(self, directory, model, model_type):
    scores_and_files = get_saves(directory, self.args.early_stop)
    if scores_and_files:
      assert scores_and_files, 'no saves exist at {}'.format(directory)
      score, filepath = scores_and_files[0]
    elif model.module_type == 'intent_tracker':
      model.load_nlu_model('nlu_1468447442')
      return model
    elif model.module_type == 'text_generator':
      model.load_nlg_model('nlg_1468202263')
      return model
    else:
      filename = "epoch=12_success=25.4042_recall@two=41.3395"
      filepath = os.path.join(self.save_dir, filename)

    checkpoint = self.loader.restore_checkpoint(filepath)
    model.load_state_dict(checkpoint['model'])
    model.existing_checkpoint = checkpoint
    model.train() if self.use_existing else model.eval()

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
    elif model_type == "rulebased":
      model = HackPolicy(processor.ontology)
      mmodel_type = model_type
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
    elif model_type == "glad":
      model = GLAD(self.args, self.loader.ontology,
                    self.loader.vocab, self.embeddings, enc.GLAD_Encoder)
      model = self.add_loss_function(model, "binary_cross_entropy")
      model.module_type = 'belief_tracker'
    elif model_type == 'ddq':
      model = DQN(input_size, self.dhid, output_size)
      model.module_type = 'policy_manager'
      return model.to(torch.device('cpu'))
    elif model_type == "attention":
      encoder = enc.GRU_Encoder(input_size, self.args)
      decoder = dec.Attn_Decoder(output_size, self.args)
      model = Seq2Seq(encoder, decoder)
      model.module_type = 'text_generator'
    elif model_type == 'nlg_model':
      model = NLG(self.loader, 'results/end_to_end/ddq/movies/')
      model.load_nlg_model('nlg_1468202263')
      model.load_natural_langauge_templates('nl_templates')
      model.module_type = 'text_generator'
      return model
    elif model_type == 'nlu_model':
      model = NLU(self.loader, 'results/end_to_end/ddq/movies/')
      model.load_nlu_model('nlu_1468447442')
      model.module_type = 'intent_tracker'
      return model

    return model.to(device)

  def configure_module(self, args, model):
    if model.module_type == 'belief_tracker':
      module = NeuralBeliefTracker(args, model)

    elif model.module_type == 'policy_manager':
      kb, ontology = self.loader.kb, self.loader.ontology
      if args.model == 'rulebased':
        module = RulebasedPolicyManager(args, model, kb, ontology)
        results_dir = os.path.join('results', args.dataset, 'models')
        module.user.text_generator = RuleTextGenerator.from_pretrained(args)
        module.user.text_generator.set_templates(args.dataset)
      elif args.model == 'ddq':
        pm_model = NeuralPolicyManager(args, model)
        goal_set = self.loader.json_data('goals')

        user_sim = RuleSimulator(args, ontology, goal_set)
        world_sim = NeuralSimulator(args, ontology, goal_set)
        real_user = CommandLineUser(args, ontology, goal_set)
        turk_user = MechanicalTurkUser(args, ontology, goal_set)
        users = (user_sim, world_sim, real_user, turk_user)

        results_dir = os.path.join("results", self.args.task, self.args.dataset)
        nlu_model = NLU(self.loader, results_dir)
        nlu_model.load_nlu_model('nlu_1468447442')

        nlg_model = NLG(self.loader, results_dir)
        nlg_model.load_nlg_model('nlg_1468202263')
        nlg_model.load_natural_langauge_templates('dia_act_nl_pairs.v6')

        user_sim.nlu_model = nlu_model
        world_sim.nlu_model = nlu_model
        user_sim.nlg_model = nlg_model
        world_sim.nlg_model = nlg_model

        pm_model.configure_settings(device, world_sim, ontology, kb)
        module = DialogManager(args, pm_model, users, ontology, kb)

    elif model.module_type == 'text_generator':
      module = model

    module.dir = self.dir
    return module

  def create_agent(self, bt_model, pm_model, tg_model):
    kb, goals = self.loader.kb, self.loader.goals
    ontology = self.loader.ontology

    user_sim = RuleSimulator(self.args, ontology, goals)
    world_sim = NeuralSimulator(self.args, ontology, goals)
    users = (user_sim, world_sim)

    if self.use_old_nlu:
      belief_tracker = bt_model
    else:
      belief_tracker = NeuralBeliefTracker(self.args, bt_model)

    policy_manager = NeuralPolicyManager(self.args, pm_model)
    text_generator = NeuralTextGenerator(self.args, tg_model)

    user_sim.nlu_model = belief_tracker
    world_sim.nlu_model = belief_tracker
    user_sim.nlg_model = text_generator
    world_sim.nlg_model = text_generator

    policy_manager.configure_settings(device, world_sim, ontology, kb)

    return DialogManager(self.args, policy_manager, users, ontology, kb)
