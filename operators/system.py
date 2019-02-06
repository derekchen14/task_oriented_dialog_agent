from operators.learn import Learner
from operators.evaluate import Monitor

class SingleSystem(object):
  def __init__(self, args, loader, builder, processor, evaluator):
    self.args = args
    self.metrics = args.metrics
    self.task = args.task
    self.evaluator = evaluator
    self.monitor = Monitor(args.epochs, args.threshold, self.task)

    model = builder.get_model(processor)
    self.module = builder.configure_module(args, model, loader)
    if not args.test_mode:
      self.learner = Learner(args, self.module, processor, self.monitor)

  def run_main(self):
    if self.task == "glad":
      self.module.learn(self.args, self.learner.processor.datasets)
      # self.learner.supervise(self.args)
    elif self.task == "policy":
      self.learner.reinforce(self.args)

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
    self.task = loader.categories
    self.evaluator = evaluator

    belief_tracker = builder.configure_module(args, "belief monitor")
    policy_manager = builder.configure_module(args, "policy manager")
    text_generator = builder.configure_module(args, "text generator")
    modules = [belief_tracker, policy_manager, text_generator]
    self.dialogue_agent = builder.create_agent(args, modules)

    if not args.test_mode:
      self.learner = Learner(args, self.dialogue_agent, processor)

  def run_main(self):
    self.learner.learn(self.task)

  def evaluate(self):
    self.evalutor.set_agent(self.dialogue_agent)
    self.evaluator.run_report(self.metrics)
