import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import os, pdb, sys

from utils.external.bleu import BLEU
import utils.internal.initialization as data_io
from objects.components import var, run_inference

class Evaluator(object):
  def __init__(self, args, processor):
    self.config = args
    self.metrics = args.metrics
    self.method = args.attn_method
    self.verbose = args.verbose

    self.save_dir = os.path.join("results", args.task, args.dataset)
    self.vocab = processor.vocab
    if args.test_mode:
      self.data = processor.datasets['test']
    else:
      self.data = processor.datasets['val']

    if args.report_results:
      self.quantitative_report()
      self.qualitative_report()
    if args.visualize > 0:
      self.visual_report(args.visualize)
    if args.plot_results:
      self.plot("Training curve")

  def plot(title):
    tracker = getattr(system.learner, '{}_tracker'.format(self.tasks[0]))
    xs = tracker.train_epochs   # tracker.val_epochs
    ys = tracker.train_losses   # tracker.val_losses

    xlabel = "Iterations"
    ylabel = "Loss"
    assert len(xs) == len(ys)
    for i in range(len(xs)):
      plt.plot(xs[i], ys[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(["Train", "Validation"])
    plt.show()

  def set_agent(agent):
    """
    semantics is that the agent contain multiple modules, rather than just one
    """
    self.multitask = True
    self.agent = agent

  def set_module(module):
    """
    a single module is either the intent prediction, policy management or
    text generation.  Belief tracking includes the KB Operator and Dialog
    State Tracking, but those are deterministic and are not evaluated
    """
    self.multitask = False
    self.module = module

  def run_report(self, metrics):
    if self.tasks[0] == "glad":
      self.model.quant_report(self.data, self.config)  # qual_report for qualitative analysis
      sys.exit()
    scores = {"inform": 0, "request": 0, "exact": 0, "rank": 0}
    display = []
    use_display = False
    for example in self.data:
      utterance, target = example

      if random.random() < -1:
        display.append(" ----- ")
        input_text = " ".join([self.vocab.index_to_word(token) for token in utterance])
        display.append(input_text)
        use_display = True
        target_words = ["Target: "]
        pred_words = ["Prediction: "]

      corrects = {"inform": True, "request": True, "exact": True, "rank": True}
      for idx, task in enumerate(self.tasks):
        model = self.models[task] if self.multitask else self.model
        hidden_state = model.encoder.initHidden()

        output = model(utterance, hidden_state)
        _, top1 = output.data.topk(1)
        exact_pred = top1[0][0]
        _, top2 = output.data.topk(2)
        rank_preds = top2[0]

        target_word = self.vocab.index_to_label(target)
        if exact_pred != target:
          corrects["exact"] = False

        if target not in rank_preds:
          corrects["rank"] = False
          if task in ["area", "food", "price"]:
            corrects["inform"] = False
          elif task == "request":
            corrects["request"] = False
          elif task == "slot" and idx < 3:  # indexes 0, 1, 2 refer to area, food, price
            corrects["inform"] = False
          elif task == "slot" and idx == 3:  # index 3 refers to question
            corrects["request"] = False
          elif task == "value" and target < 87:  # indexes 1 to 86 refer values of area, food, price
            corrects["inform"] = False
          elif task == "value" and target >= 87:  # index 3 refers to values of request
            corrects["request"] = False
          elif task in ["full_enumeration", "possible_only", "ordered_values"]:
            slot_type = target_word.split("=")
            if slot_type[0] in ["area", "food", "price"]:
              corrects["inform"] = False
            else:
              corrects["request"] = False

        if use_display and target != 0:
          target_words.append(target_word)
        if use_display and exact_pred != 0:
          pred_words.append(self.vocab.index_to_word(exact_pred))

      for success, status in corrects.items():
        if status:
          scores[success] += 1
      if use_display:
        display.append(" ".join(target_words))
        display.append(" ".join(pred_words))
        use_display = False

    for success, score in scores.items():
      accuracy = float(score) / len(self.data)
      print("{} accuracy: {:.4f}".format(success, accuracy))
    for line in display:
      print(line)

  # Qualitative evalution of model performance
  def qualitative_report(self):
    qual_report_path = "{}/qual.txt".format(self.save_dir)
    samples = [random.choice(self.data) for i in range(10)]
    with open(qual_report_path, "w") as file:
      for sample in samples:
        source, target = sample
        _, pred, _ = run_inference(self.model, source, target, None, 0)
        human_readable = self.vocab.index_to_word(pred)

        input_text = " ".join([self.vocab.index_to_word(token) for token in source])
        file.write("Input: {}\n".format(input_text))
        file.write("Predicted: {}\n".format(human_readable))
    print('Qualitative examples saved to {}'.format(qual_report_path))

  # Quantitative evalution of model performance
  def quantitative_report(self):
    quant_report_path = "{}/quant.txt".format(self.save_dir)
    train_s, train_l = self.tracker.train_steps, self.tracker.train_losses
    val_s, val_l = self.tracker.val_steps, self.tracker.val_losses
    bleu, accuracy = self.tracker.bleu_scores, self.tracker.accuracy

    df_train = pd.DataFrame(data={'train_steps':train_s, 'train_loss':train_l})
    df_val = pd.DataFrame(data={'validation_steps':val_s, 'validation_loss': val_l,
                      'bleu_score': bleu, 'per_turn_accuracy': accuracy})
    df_params = pd.DataFrame(data={
        "Params": ['hidden-size', 'learning-rate', 'drop-prob', 'model-type', \
                  'weight-decay', 'decay-times', 'attention-method'],
        "Values": [self.config.hidden_size, self.config.learning_rate, \
                  self.config.drop_prob, self.config.model_type, \
                  self.config.weight_decay, self.config.decay_times, \
                  self.config.attn_method]})
    loss_history = pd.concat([df_train, df_val, df_params], axis=1)
    loss_history.to_csv(quant_report_path, index=False)
    print('Loss, BLEU and accuracy saved to {}'.format(quant_report_path))

  def visual_report(self, vis_count):
    self.model.eval()
    dialogues = data_io.select_consecutive_pairs(self.data, vis_count)

    visualizations = []
    for dialog in dialogues:
      for turn in dialog:
        input_variable, output_variable = turn
        _, responses, visual = run_inference(self.model, input_variable, \
                        output_variable, criterion=NLLLoss(), teach_ratio=0)
        queries = input_variable.data.tolist()
        query_tokens = [self.vocab.index_to_word(q[0]) for q in queries]
        response_tokens = [self.vocab.index_to_word(r) for r in responses]

        visualizations.append((visual, query_tokens, response_tokens))
    self.show_save_attention(visualizations)


  def show_save_attention(self, visualizations):
    for i, viz in enumerate(self.visualizations):
      visual, query_tokens, response_tokens = viz
      visual[-1,:] = 0
      # Set up figure with colorbar
      fig = plt.figure()
      ax = fig.add_subplot(111)
      cax = ax.matshow(visual.numpy(), cmap='bone')
      fig.colorbar(cax)
      # Set up axes
      ax.set_yticklabels([''] + query_tokens)
      ax.set_xticklabels([''] + response_tokens, rotation=90)
      ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
      ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
      # Set up labels
      plt.ylabel("User Query")
      plt.xlabel("Agent Response")
      # pdb.set_trace()
      plt.savefig("results/visualize_{0}{1}.png".format(self.method, i))
      if self.verbose:
        plt.show()
      plt.close()



"""
  def dual_report(self, slot_learner, value_learner):
    datapoint_count = len(slot_learner.processor.val_data)
    slot_model = slot_learner.model
    value_model = value_learner.model

    success, rank_success =  0, 0
    display = []
    for idx in range(datapoint_count):
      utterance = slot_learner.processor.val_data[idx][0]
      slot_target = slot_learner.processor.val_data[idx][1]
      value_target = value_learner.processor.val_data[idx][1]

      intent_hidden = slot_model.encoder.initHidden()
      intent_output = slot_model(utterance, intent_hidden)
      _, i_top = intent_output.data.topk(1)
      i_pred = i_top[0][0]
      _, i_rank = intent_output.data.topk(2)
      i_preds = i_rank[0]

      slot_value_hidden = value_model.encoder.initHidden()
      slot_value_output = value_model(utterance, slot_value_hidden)
      _, slot_value_top = slot_value_output.data.topk(1)
      sv_pred = slot_value_top[0][0]
      _, sv_rank = slot_value_output.data.topk(2)
      sv_preds = sv_rank[0]

      if (i_pred == slot_target) and (sv_pred == value_target):
        success += 1
      if (slot_target in i_preds) and (value_target in sv_preds):
        rank_success += 1
      if random.random() < 0.01:
        input_text = " ".join([vocab.index_to_word(token, self.task) for token in utterance])
        post_slot_target = vocab.index_to_word(slot_target, "slot")
        post_value_target = vocab.index_to_word(value_target, "value")
        post_slot_pred = vocab.index_to_word(i_pred, "slot")
        post_value_pred = vocab.index_to_word(sv_pred, "value")

        display.append(input_text)
        display.append("Target: {0}({1})".format(post_slot_target, post_value_target))
        display.append("Predicted: {0}({1})".format(post_slot_pred, post_value_pred))
        display.append(" ----- ")

    rank_accuracy = rank_success / float(datapoint_count)
    print("Exact accuracy: {:.4f}, rank accuracy {:.4f}".format(
      success / float(datapoint_count), rank_accuracy) )
    for line in display:
      print(line)
"""
