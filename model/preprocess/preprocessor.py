import utils.internal.vocabulary as vocab
from model.components import var
from utils.internal.data_io import load_dataset

class PreProcessor(object):
  def __init__(self, args):
    use_context, debug, task = args.context, args.debug, args.task_name
    if args.test_mode:
      self.prepare_examples("test", use_context, debug, task)
    else:
      self.prepare_examples("train", use_context, debug, task)
      self.prepare_examples("val", use_context, debug, task)

  def prepare_examples(self, split, use_context, debug, task):
    ''' For DSTC2, format is
    Example is dict with keys
      {"input_source": [utterance, context, id]}
      {"output_target": [list of labels]}
    where each label is (high level intent, low level intent, slot, value)
    '''
    dataset, max_length = load_dataset(task, split, debug)

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
    kind = "ordered_values" # "full_enumeration", "possible_only"

    if len(target) == 1:
      target_index = vocab.belief_to_index(target[0], kind)
      output_var = var([target_index], "long")
      return output_var, False
    elif len(target) == 2:
      target_index = vocab.beliefs_to_index(target, kind)
      if kind == "full_enumeration":
        output_var = var([target_index], "long")
        return output_var, False
      else:
        output_vars = [var([ti], "long") for ti in target_index]
        return output_vars, True

  def dialog_to_variable(self, dialog, task):
    # example is list of tuples, where each tuple is utterance and response
    # [(u1, r1), (u2, r2)...]
    dialog_pairs = []
    for t_idx, turn in enumerate(dialog):
      utterance, wop = variable_from_sentence(turn[0], [t_idx+1], task)
      response, dop = variable_from_sentence(turn[1], [], task)
      dialog_pairs.append((utterance, response))
    return dialog_pairs
