import utils.internal.per_slot_vocab as vocab
from model.components import var
from utils.internal.data_io import load_dataset, pickle_io
import os, pdb

class PerSlotProcessor(object):
  def __init__(self, args):
    if args.test_mode:
      self.prepare_examples("test", args.task_name)
    else:
      self.prepare_examples("train", args.task_name)
      self.prepare_examples("val", args.task_name)

  def prepare_examples(self, split, task):
    dataset, max_length = load_dataset(task, split)

    variables = []
    for example in dataset:
      input_var = self.prepare_input(example["input_source"])
      output_var, double = self.prepare_output(example["output_target"])
      if double:
        variables.append((input_var, output_var[0]))
        output_var = output_var[1]
      variables.append((input_var, output_var))

    setattr(self, "{}_data".format(split), variables)

  def prepare_input(self, source):
    tokens = [vocab.word_to_index(word) for word in source]
    return var(tokens, "long")
    # tokens = []
    # for word in source["utterance"].split():
    #   tokens.append(vocab.word_to_index(word))
    # return var(tokens, "long")

  def prepare_output(self, target):
    if len(target) == 1:
      indexes = var(vocab.belief_to_index(target[0]), "long")
      return indexes, False
    elif len(target) == 2:
      idx_a = var(vocab.belief_to_index(target[0]), "long")
      idx_b = var(vocab.belief_to_index(target[1]), "long")
      indexes = (idx_a, idx_b)
      return indexes, True