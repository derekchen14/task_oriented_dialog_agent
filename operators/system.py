from operators.learn import Learner
from operators.evaluate import LossTracker, RewardTracker
from utils.internal.vocabulary import Vocabulary
from utils.internal.ontology import Ontology

class SingleSystem(object):
  def __init__(self, args, loader, builder, processor, evaluator):
    self.args = args
    self.metrics = args.metrics
    self.task = args.task
    self.evaluator = evaluator

    if self.task == "policy":
      tracker = RewardTracker(args)
      processor.ontology = Ontology(args, loader.data_dir)
    else:
      tracker = LossTracker(args)
      processor.vocab = Vocabulary(args, loader.data_dir)

    if args.use_existing:
      builder.module_loader = ModuleLoader(args)
      self.module = builder.set_module(args)
    else:
      model = builder.get_model(processor)
      # model = builder.create_model(processor)
      self.module = builder.configure_module(args, model)

    if not args.test_mode:
      self.learner = Learner(args, self.module, processor, tracker)

  def run_main(self):
    # unpack only the arguments needed, store into params
    params = {}
    params["task"] = self.task
    params["hidden_dim"] = self.args.hidden_dim
    datasets = self.learner.processor.datasets

    if self.task == "glad":
      self.module.learn(datasets, self.args)
      # self.learner.supervise(params, self.module, datasets)
    elif self.task == "policy":
      self.learner.reinforce(params, self.module, datasets)
    else:
      self.learner.rulebased(params, self.module, datasets)

  def evaluate(self):
    self.evaluator.model = self.module
    self.evalutor.run_report(self.metrics)


class EndToEndSystem(object):
  def __init__(self, args, loader, builder, tracker, evaluator):
    self.args = args
    self.metrics = args.metrics
    self.task = loader.categories
    vocab = Vocabulary(args, loader.data_dir)
    self.evaluator = evaluator

    belief_tracker = builder.configure_module(args, "belief tracker")
    policy_manager = builder.configure_module(args, "policy manager")
    text_generator = builder.configure_module(args, "text generator")
    modules = [belief_tracker, policy_manager, text_generator]

    self.dialogue_agent = builder.create_agent(args, modules)
    self.processor = PreProcessor(args, vocab, loader)

    if not args.test_mode:
      self.learner = Learner(args, self.dialogue_agent, self.processor, tracker)

  def run_main(self):
    self.learner.learn(self.task)

  def evaluate(self):
    self.evalutor.set_agent(self.dialogue_agent)
    self.evaluator.run_report(self.metrics)