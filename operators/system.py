from operators.learn import Learner
from operators.evaluate import RewardMonitor, LossMonitor

class SingleSystem(object):
  def __init__(self, args, loader, builder, processor, evaluator):
    self.args = args
    self.task = args.task
    self.evaluator = evaluator
    self.monitor = LossMonitor(args.threshold, args.metrics, args.early_stop)

    model = builder.get_model(processor, self.monitor)
    model.save_config(args, builder.dir)
    self.module = builder.configure_module(args, model, loader)
    if not args.test_mode:
      self.learner = Learner(args, self.module, processor, self.monitor)

  def run_main(self):
    self.learner.supervise(self.args)

  def evaluate(self, test_mode):
    self.evaluator.module = self.module
    self.evaluator.monitor = self.monitor
    if self.args.test_mode:
      self.run_test()
    self.evaluator.generate_report()

class EndToEndSystem(object):
  def __init__(self, args, loader, builder, processor, evaluator):
    self.args = args
    self.metrics = args.metrics
    self.evaluator = evaluator
    self.monitor = RewardMonitor(args.metrics, args.threshold)

    nlu_model = builder.get_model(processor, self.monitor, "belief_tracker")
    pm_model = builder.get_model(processor, self.monitor, "policy_manager")
    nlg_model = builder.get_model(processor, self.monitor, "text_generator")

    belief_tracker = builder.configure_module(args, nlu_model)
    policy_manager = builder.configure_module(args, pm_model)
    text_generator = builder.configure_module(args, nlg_model)

    modules = [belief_tracker, policy_manager, text_generator]
    self.dialogue_agent = builder.create_agent(*modules)

    if not args.test_mode:
      self.learner = Learner(args, self.dialogue_agent, processor, self.monitor)

  def run_main(self):
    self.learner.reinforce(self.args)

  def evaluate(self, test_mode):
    self.evaluator.agent = self.dialogue_agent
    self.evaluator.monitor = self.monitor
    self.evaluator.generate_report()