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
from operators.evaluate.server import HTTPServer, Handler, ToyModel

class Evaluator(object):
  def __init__(self, args, processor):
    self.config = args
    self.task = args.task
    self.metrics = args.metrics
    self.method = args.attn_method
    self.verbose = args.verbose
    self.batch_size = args.batch_size

    self.module = None
    self.monitor = None

    self.save_dir = os.path.join("results", args.task, args.dataset)
    self.vocab = processor.vocab
    self.data = processor.datasets['test'] if args.test_mode else processor.datasets['val']
    self.ontology = processor.ontology

  def run_test(self):
    output = self.module.run_glad_inference(self.data)

  def generate_report(self):
    if self.config.report_visual:
      self.visual_report()
    if self.config.report_qual:
      self.qualitative_report()
    if self.config.report_quant:
      self.monitor.summarize_results(verbose=False)
      # if self.task == "glad":
      #   self.model.quant_report(self.data, self.config)
      self.quantitative_report()

  def qualitative_report(self):
    print("starting qualitative_report")
    self.module.model.eval()
    samples = next(self.data.batch(self.batch_size, shuffle=True))
    loss, scores = self.module.model.forward(samples)
    predictions = self.module.extract_predictions(scores)
    vals = self.ontology.values
    lines = self.module.qual_report(samples, predictions, scores, vals)

    save_dir = self.save_dir if self.module is None else self.module.dir
    qual_report_path = os.path.join(save_dir, "qual.txt")
    with open(qual_report_path, "w") as file:
      for line in lines:
        file.write(line + '\n')
    print('Qualitative examples saved to {}'.format(qual_report_path))

  """ Quantitative evalution of model performance, for per step loss
      please view the logging output in the results folder instead """
  def quantitative_report(self):
    datarows = {}
    extras = ["save_model", "prefix", "suffix", "verbose", "gpu", "context", \
              "metrics", "report_visual", "report_qual", "report_quant"]
    for param, value in vars(self.config).items():
      if param not in extras:
        datarows[param] = value

    for metric in self.metrics:
      datarows[metric] = getattr(self.monitor, metric)

    quant = pd.DataFrame(data=datarows, index=["Attributes"])
    self.save_csv_report(quant.T, "quant")

  def visual_report(self, vis_count):
    self.model.eval()
    dialogues = data_io.select_consecutive_pairs(self.data, vis_count)
    train_s, train_l = self.monitor.train_steps, self.monitor.train_losses
    val_s, val_l = self.monitor.val_steps, self.monitor.val_losses

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

  def save_csv_report(self, report, filename):
    report_path = "{}/{}.csv".format(self.save_dir, filename)
    report.to_csv(report_path, index=True)
    print('{} report complete, saved to {}!'.format(filename, report_path))

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

  def start_talking(self, user_type, num_epochs):
    self.episode_counter = 0
    self.max_episodes = num_epochs

    while not self.done_talking(user_type):
      turn_count = 0
      self.agent.initialize_episode(user_type)
      self.agent.start_conversation(user_type)

      episode_over = False
      while not episode_over:
        episode_over, reward = self.agent.next(record_user_data=False)
        turn_count += 1

        if episode_over:
          final_reward = reward - turn_count # lose reward for every turn taken
          result = "succeeded  :)" if final_reward > 0 else "failed  :("
          print(f"this dialogue {result}")
          print("---" * 16)

  def done_talking(self, user_type):
    if self.episode_counter == 0:
      self.episode_counter += 1
      return False
    if user_type == 'simulate':
      if self.episode_counter > self.max_episodes:
        return True
      else:
        self.episode_counter += 1
        return False

    done = input("Are you done talking? ")
    if done in ["yes", "Yes", "y"]:
      print("It was good talking to you, have a nice day!")
      return True
    elif done in ["no", "No", "n"]:
      return False

  def start_server(self, root_dir):
    """ Server will run on port 1414 from index.html
    It will chat with the user until user decides to end.  Then the
    user will be redirected to survey page, which stores records in CSV.
    For more details see operators/evaluate/server.py """
    ip_address = ('0.0.0.0', 1414)
    httpd = HTTPServer(ip_address, Handler)
    httpd.agent = self.agent
    httpd.wd = os.path.relpath(os.path.join(root_dir, 'utils'))
    print("Listening at", ip_address)
    httpd.serve_forever()

