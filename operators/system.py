from operators.learn import Learner
from operators.evaluate import RewardMonitor, LossMonitor

class SingleSystem(object):
  def __init__(self, args, loader, builder, processor, evaluator):
    self.args = args
    self.task = args.task
    self.evaluator = evaluator
    if self.args.task == 'track_intent':
      self.monitor = LossMonitor(args.metrics, args.threshold, args.early_stop)
    elif self.args.task == 'manage_policy':
      self.monitor = RewardMonitor(args.metrics, args.threshold)

    model = builder.get_model(processor, self.monitor)
    self.module = builder.configure_module(args, model)
    self.module.save_config(args, builder.dir)
    if not args.test_mode:
      self.learner = Learner(args, self.module, processor, self.monitor)

  def run_main(self):
    if self.args.task == 'track_intent':
      self.learner.supervise(self.args)
    elif self.args.task == 'manage_policy':
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

    if args.use_existing:
      self.monitor.restore_from_checkpoint(modules)
    if not args.test_mode:
      self.learner = Learner(args, self.dialogue_agent, processor, self.monitor)

  def run_main(self):
    self.learner.end_to_end(self.args)

  def evaluate(self, test_mode):
    self.evaluator.agent = self.dialogue_agent
    self.evaluator.monitor = self.monitor
    if test_mode:
      if self.args.user in ['command', 'simulate']:
        self.evaluator.start_talking(self.args.user, self.args.epochs)
      elif self.args.user == 'turk':
        import os
        root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        self.evaluator.start_server(root_dir)
    else:
      self.evaluator.generate_report()