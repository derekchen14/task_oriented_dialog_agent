import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import sys, pdb

from utils.external.bleu import BLEU
import utils.internal.vocabulary as vocab
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