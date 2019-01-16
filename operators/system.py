from operators.preprocess import PreProcessor
from operators.learn import Learner
from utils.internal.vocabulary import Vocabulary

class SingleSystem(object):
  def __init__(self, args, loader, builder, tracker):
    self.args = args
    self.task = loader.categories
    vocab = Vocabulary(args, loader.data_dir)
    self.model = builder.get_model(vocab.ulary_size(), vocab.label_size())
    # self.module = builder.configure_module(args, self.model)
    self.processor = PreProcessor(args, vocab, loader)
    if not args.test_mode:
      self.learner = Learner(args, self.model, self.processor, tracker)

  def run_main(self):
    if self.task == "glad":
      self.model.learn(self.processor.datasets, self.args)
    else:
      self.learner.learn(self.task)

class MultiSystem(object):
  def __init__(self, args, loader, builder, tracker):
    self.tasks = loader.categories
    self.train_mode = not args.test_mode

    self.models = {}
    self.learners = {}
    for task in self.tasks:
      vocab = Vocabulary(args, loader.data_dir, task)
      model = builder.get_model(vocab.ulary_size(), vocab.label_size(), task)
      self.models[task] = model
      self.processor = PreProcessor(args, vocab, loader, task)
      if self.train_mode:
        learner = Learner(args, model, self.processor, tracker, task)
        self.learners[task] = learner

  def run_main(self):
    for task in self.tasks:
      learner = self.learners[task]
      learner.learn(task)

class EndToEndSystem(object):
  def __init__(self, args, loader, builder, tracker):
    self.args = args
    self.task = loader.categories
    vocab = Vocabulary(args, loader.data_dir)

    belief_tracker = builder.configure_module(args, self.model)
    policy_manager = builder.configure_module(args, self.model)
    text_generator = builder.configure_module(args, self.model)
    modules = [belief_tracker, policy_manager, text_generator]

    self.processor = PreProcessor(args, vocab, loader)
    if not args.test_mode:
      self.learner = Learner(args, modules, self.processor, tracker)

  def run_main(self):
    self.learner.learn(self.task)