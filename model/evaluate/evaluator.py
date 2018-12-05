import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import sys, pdb

from utils.external.bleu import BLEU
# import utils.internal.vocabulary as vocab
# import utils.internal.dual_vocab as vocab
import utils.internal.per_slot_vocab as vocab
import utils.internal.data_io as data_io
from model.components import var, run_inference

class Evaluator(object):
  def __init__(self, args, kind):
    self.qual_report_path = "results/{0}_{1}_qual.txt".format(args.report_path, args.suffix)
    self.quant_report_path = "results/{0}_{1}_quant.csv".format(args.report_path, args.suffix)

    self.method = args.attn_method
    self.verbose = args.verbose
    self.config = args
    self.kind = kind

  def plot(xs, ys, title="Training Curve"):
  # if args.plot_results:
  #   evaluator.plot([strain, sval], [ltrain, lval])
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

  def dual_report(self, intent_learner, sv_learner):
    datapoint_count = len(intent_learner.processor.val_data)
    i_model = intent_learner.model
    sv_model = sv_learner.model

    success, rank_success =  0, 0
    display = []
    for idx in range(datapoint_count):
      utterance = intent_learner.processor.val_data[idx][0]
      intent_target = intent_learner.processor.val_data[idx][1]
      sv_target = sv_learner.processor.val_data[idx][1]

      intent_hidden = i_model.encoder.initHidden()
      intent_output = i_model(utterance, intent_hidden)
      _, i_top = intent_output.data.topk(1)
      i_pred = i_top[0][0]
      _, i_rank = intent_output.data.topk(2)
      i_preds = i_rank[0]

      slot_value_hidden = sv_model.encoder.initHidden()
      slot_value_output = sv_model(utterance, slot_value_hidden)
      _, slot_value_top = slot_value_output.data.topk(1)
      sv_pred = slot_value_top[0][0]
      _, sv_rank = slot_value_output.data.topk(2)
      sv_preds = sv_rank[0]

      if (i_pred == intent_target) and (sv_pred == sv_target):
        success += 1
      if (intent_target in i_preds) and (sv_target in sv_preds):
        rank_success += 1
      if random.random() < 0.01:
        input_text = " ".join([vocab.index_to_word(token, "dstc2") for token in utterance])
        post_intent_target = vocab.index_to_word(intent_target, "intent")
        post_sv_target = vocab.index_to_word(sv_target, "sv")
        post_i_pred = vocab.index_to_word(i_pred, "intent")
        post_sv_pred = vocab.index_to_word(sv_pred, "sv")

        display.append(input_text)
        display.append("Target: {0}({1})".format(post_intent_target, post_sv_target))
        display.append("Predicted: {0}({1})".format(post_i_pred, post_sv_pred))
        display.append(" ----- ")

    rank_accuracy = rank_success / float(datapoint_count)
    print("Overall accuracy: {:.4f}, rank accuracy {:.4f}".format(
      success / float(datapoint_count), rank_accuracy) )
    for line in display:
      print(line)


  def per_slot_report(self, learners):
    data = learners[0].processor.val_data
    exact_success, rank_success = 0, 0
    display = []
    use_display = False
    for example in data:
      utterance, targets = example

      if random.random() < 0.01:
        display.append(" ----- ")
        input_text = " ".join([vocab.index_to_word(token, "dstc2") for token in utterance])
        display.append(input_text)
        use_display = True
        target_words = ["Target: "]
        pred_words = ["Prediction: "]

      exact_correct = True
      rank_correct = True
      for idx, learner in enumerate(learners):
        target = targets[idx]
        category = vocab.categories[idx]

        hidden_state = learner.model.encoder.initHidden()
        output = learner.model(utterance, hidden_state)
        _, top1 = output.data.topk(1)
        exact_pred = top1[0][0]
        _, top2 = output.data.topk(2)
        rank_preds = top2[0]

        if exact_pred is not target:
          exact_correct = False
        if target not in rank_preds:
          rank_correct = False

        if use_display and target != 0:
          target_words.append(vocab.index_to_word(target, category))
        if use_display and exact_pred != 0:
          pred_words.append(vocab.index_to_word(exact_pred, category))

      if exact_correct:
        exact_success += 1
      if rank_correct:
        rank_success += 1
      if use_display:
        display.append(" ".join(target_words))
        display.append(" ".join(pred_words))
        use_display = False

    exact_accuracy = float(exact_success) / len(data)
    rank_accuracy = float(rank_success) / len(data)
    print("Exact accuracy: {:.4f}, Rank accuracy {:.4f}".format(exact_accuracy, rank_accuracy) )
    for line in display:
      print(line)

  # Qualitative evalution of model performance
  def qual_report(self, model, data):
    samples = [random.choice(data) for i in range(10)]
    task = self.config.task_name
    with open(self.qual_report_path, "w") as file:
      for sample in samples:
        source, target = sample
        _, pred, _ = run_inference(model, source, target, None, 0)
        human_readable = vocab.index_to_word(pred, self.kind)

        input_text = " ".join([vocab.index_to_word(token, task) for token in source])
        file.write("Input: {}\n".format(input_text))
        file.write("Predicted: {}\n".format(human_readable))
    print('Qualitative examples saved to {}'.format(self.qual_report_path))

  # Quantitative evalution of model performance
  def quant_report(self, tracker):
    train_s, train_l = tracker.train_steps, tracker.train_losses
    val_s, val_l = tracker.val_steps, tracker.val_losses
    bleu, accuracy = tracker.bleu_scores, tracker.accuracy

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
    loss_history.to_csv(self.quant_report_path, index=False)
    print('Loss, BLEU and accuracy saved to {}'.format(self.quant_report_path))

  def visual_report(self, val_data, model, task, vis_count):
    model.eval()
    dialogues = data_io.select_consecutive_pairs(val_data, vis_count)

    visualizations = []
    for dialog in dialogues:
      for turn in dialog:
        input_variable, output_variable = turn
        _, responses, visual = run_inference(model, input_variable, \
                        output_variable, criterion=NLLLoss(), teach_ratio=0)
        queries = input_variable.data.tolist()
        query_tokens = [vocab.index_to_word(q[0], task) for q in queries]
        response_tokens = [vocab.index_to_word(r, task) for r in responses]

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