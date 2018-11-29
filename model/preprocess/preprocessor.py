import utils.internal.vocabulary as vocab
from model.components import var
from utils.internal.data_io import load_dataset, pickle_io
import os

class PreProcessor(object):
  def __init__(self, args, kind):
    self.kind = kind
    use_context, task = args.context, args.task_name
    if args.test_mode:
      self.prepare_examples("test", use_context, task)
    elif args.debug:
      self.load_debug_examples(task)
    else: # normal training mode
      self.prepare_examples("train", use_context, task)
      self.prepare_examples("val", use_context, task)
      self.make_cache(task)

  def load_debug_examples(self, task):
    data_path = os.path.join("datasets", task, "debug", "cache.pkl")
    debug_data = pickle_io(data_path, "load")
    self.train_data = debug_data["train"]
    self.val_data = debug_data["val"]

  def prepare_examples(self, split, use_context, task):
    ''' For DSTC2, format is
    Example is dict with keys
      {"input_source": [utterance, context, id]}
      {"output_target": [list of labels]}
    where each label is (high level intent, low level intent, slot, value)
    '''
    dataset, max_length = load_dataset(task, split)

    variables = []
    for example in dataset:
      if task in ["in-car", "babi"]:
        input_output_vars = self.dialog_to_variable(example, task)
      elif task == "dstc2":
        input_var = self.prepare_input(example["input_source"], use_context, task)
        output_var, double = self.prepare_output(example["output_target"])
        if double:
          variables.append((input_var, output_var[0]))  # append extra example
          output_var = output_var[1]      # set output to be the second label
      variables.append((input_var, output_var))

    setattr(self, "{}_data".format(split), variables)

  def prepare_input(self, source, use_context, task):
    tokens = []
    if "turn" in source.keys():
      # vocab is designed so that the first 14 tokens are turn indicators
      tokens.append(source["turn"])
    if use_context:
      for word in source["context"].split():
        tokens.append(var(vocab.word_to_index(word, task), "long"))
      tokens.append(var(vocab.SOS_token, "long"))

    for word in source["utterance"].split():
      tokens.append(vocab.word_to_index(word, task))
    return var(tokens, "long")

  def prepare_output(self, target):
    if len(target) == 1:
      dual_index = vocab.belief_to_index(target[0])
      if self.kind == "intent":
        output_var = var([dual_index[0]], "long")
      elif self.kind == "sv":
        output_var = var([dual_index[1]], "long")
      return output_var, False

    elif len(target) == 2:
      dual_indexes = [vocab.belief_to_index(b) for b in target]
      output_vars = []
      for di in dual_indexes:
        if self.kind == "intent":
          output_vars.append(var([di[0]], "long"))
        elif self.kind == "sv":
          output_vars.append(var([di[1]], "long"))

      return output_vars, True
  '''
  def prepare_output(self, target):
    if len(target) == 1:
      target_index = vocab.belief_to_index(target[0], self.kind)
      output_var = var([target_index], "long")
      return output_var, False
    elif len(target) == 2:
      target_index = vocab.beliefs_to_index(target, self.kind)
      if self.kind == "full_enumeration":
        output_var = var([target_index], "long")
        return output_var, False
      else:
        output_vars = [var([ti], "long") for ti in target_index]
        return output_vars, True
  '''

  def make_cache(self, task):
    data_path = os.path.join("datasets", task, "debug", "cache.pkl")
    if not os.path.exists(data_path):
      cache = { "train": self.train_data[0:14], "val": self.val_data[0:7] }
      pickle_io(data_path, "save", cache)
      print("{} saved successfully!".format(data_path))

  def dialog_to_variable(self, dialog, task):
    # example is list of tuples, where each tuple is utterance and response
    # [(u1, r1), (u2, r2)...]
    dialog_pairs = []
    for t_idx, turn in enumerate(dialog):
      utterance, wop = variable_from_sentence(turn[0], [t_idx+1], task)
      response, dop = variable_from_sentence(turn[1], [], task)
      dialog_pairs.append((utterance, response))

    return dialog_pairs
