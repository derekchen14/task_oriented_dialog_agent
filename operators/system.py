from model.preprocess import PreProcessor
from model.learn import Learner

class SingleSystem(object):
  def __init__(self, args, loader, builder, tracker):
    self.task = loader.categories
    vocab = loader.vocab

    self.model = builder.get_model(vocab.ulary_size(), vocab.label_size())
    self.processor = PreProcessor(args, loader)
    if not args.test_mode:
      self.model.save_dir = builder.dir
      self.learner = Learner(args, self.model, self.processor, tracker)

  def run_main(self):
    self.learner.learn(self.task)

class MultiSystem(object):
  def __init__(self, args, loader, builder, tracker):
    self.tasks = loader.categories
    self.train_mode = not args.test_mode
    vocab = loader.vocab

    self.models = {}
    self.learners = {}
    self.processors = {}
    for task in self.tasks:
      model = builder.get_model(vocab.ulary_size(), vocab.label_size())
      model.save_dir = "{}_{}".format(task, builder.dir)
      self.models[task] = model
      self.processors[task] = PreProcessor(args, loader, task)
      if self.train_mode:
        learner = Learner(args, model, processor, tracker, task)
        self.learners[task] = learner

  def run_main(self):
    for task in self.tasks:
      learner = self.learners[task]
      learner.learn(task)