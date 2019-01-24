from objects.components import var
from utils.internal.initialization import pickle_io
import os, pdb

class PreProcessor(object):
  def __init__(self, args, vocab, loader, task=None):
    self.vocab = vocab
    self.loader = loader
    self.multitask = loader.multitask
    self.task = task if self.multitask else args.task
    self.datasets = loader.datasets

    if self.task == "glad":
      pass
    elif args.test_mode:
      self.prepare_examples("test", args.context)
    elif args.debug:
      self.load_debug_examples(loader.debug_dir)
    else: # normal training mode
      self.prepare_examples("train", args.context)
      self.prepare_examples("val", args.context)
      self.make_cache(loader.debug_dir)

  def prepare_examples(self, split, use_context):
    dataset = self.datasets[split]

    variables = []
    for example in dataset:
      input_var = self.prepare_input(example["input_source"], use_context)
      output_var, double = self.prepare_output(example["output_target"])
      if double:
        variables.append((input_var, output_var[0]))  # append extra example
        output_var = output_var[1]      # set output to be the second label
      variables.append((input_var, output_var))

    self.datasets[split] = variables

  def prepare_input(self, source, use_context):
    tokens = [self.vocab.word_to_index(word) for word in source]
    return var(tokens, "long")

  def prepare_output(self, target):
    target = self._multi_prep(target) if self.multitask else self._single_prep(target)
    if len(target) == 1:
      target_index = self.vocab.label_to_index(target[0])
      output_var = var([target_index], "long")
      return output_var, False
    elif len(target) == 2:
      target_indexes = [self.vocab.label_to_index(t) for t in target]
      output_vars = [var([ti], "long") for ti in target_indexes]
      return output_vars, True

  def load_debug_examples(self, debug_dir):
    data_path = os.path.join(debug_dir, "cache.pkl")
    debug_data = pickle_io(data_path, "load")
    self.train_data = debug_data["train"]
    self.val_data = debug_data["val"]

  def make_cache(self, debug_dir):
    data_path = os.path.join(debug_dir, "cache.pkl")
    if not os.path.exists(data_path):
      cache = { "train": self.train_data[0:14], "val": self.val_data[0:7] }
      pickle_io(data_path, "save", cache)
      print("{} saved successfully!".format(data_path))

  def dialog_to_variable(self, dialog, dataset):
    # example is list of tuples, where each tuple is utterance and response
    # [(u1, r1), (u2, r2)...]
    dialog_pairs = []
    for t_idx, turn in enumerate(dialog):
      utterance, wop = variable_from_sentence(turn[0], [t_idx+1], dataset)
      response, dop = variable_from_sentence(turn[1], [], dataset)
      dialog_pairs.append((utterance, response))

    return dialog_pairs

  def _single_prep(self, target):
    if len(target) ==  1:
      act, slot, value = target[0]
      return ["{}={}".format(slot, value)]
    else:
      intents = [self._single_prep([label])[0] for label in target]
      if self.task == "full_enumeration":
        return ["+".join(intents)]
      else:
        return intents

  def _multi_prep(self, intents):
    if self.task == "slot":
      return [slot for act, slot, value in intents]
    elif self.task == "value":
      return [value for act, slot, value in intents]
    elif self.task in ["area", "food", "price", "request"]:
      targets = []
      for act, slot, value in intents:
        if self.task == slot or self.task == act:
          targets.append(value)
        else:
          targets.append("<NONE>")
      return targets

'''
tokens = []
if "turn" in source.keys():
  # vocab is designed so that the first 14 tokens are turn indicators
  tokens.append(source["turn"])
if use_context:
  for word in source["context"].split():
    tokens.append(var(vocab.word_to_index(word, dataset), "long"))
  tokens.append(var(vocab.SOS_token, "long"))

for word in source["utterance"].split():
  tokens.append(vocab.word_to_index(word, dataset))
return var(tokens, "long")

For DSTC2, format is
Example is dict with keys
  {"input_source": [utterance, context, id]}
  {"output_target": [list of labels]}
where each label is (high level intent, low level intent, slot, value)
'''