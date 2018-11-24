import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import pandas as pd
import sys, pdb

from utils.external.bleu import BLEU
import utils.internal.vocabulary as vocab
from model.components import var

def plot(xs, ys, title, xlabel, ylabel):
  assert len(xs) == len(ys)
  for i in range(len(xs)):
    plt.plot(xs[i], ys[i])

  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(['Train', "Validation"])
  plt.show()
  print('Performance plotted!')

# Qualitative evalution of model performance
def qual_report(encoder, decoder, data, args):
  samples = [random.choice(data) for i in range(10)]

  # TODO: allow sample sentences to be passed in rather than hard-coded
  # samples = ["i want european food",
  #   "restaurant in the north part of town that serves korean food",
  #   "whats the address and phone number",
  #   "okay thank you good bye",
  #   "i need cheap chinese food",
  #   "i need chinese food",
  #   "is there anything else"]

  results_path = "results/{0}_{1}_qual.txt".format(args.results_path, args.suffix)
  with open(results_path, "w") as file:
    for sample in samples:
      source, target = sample
      encoder_hidden = encoder.initHidden()
      encoder_outputs, _ = encoder(source, encoder_hidden)

      decoder_output = decoder(encoder_outputs[0])
      topv, topi = decoder_output.data.topk(1)
      pred = topi[0][0]
      human_readable = vocab.index_to_word(pred, "full_enumeration")

      input_text = " ".join([vocab.index_to_word(token, "dstc2") for token in source])
      file.write("Input: {}\n".format(input_text))
      file.write("Predicted: {}\n".format(human_readable))
    file.close()

# Quantitative evalution of model performance
def quant_report(results, args):
  learner, bleu, acc = results
  train_s, train_l = learner.train_steps, learner.train_losses
  val_s, val_l = learner.val_steps, learner.val_losses
  df_train = pd.DataFrame(data={'train_steps':train_s, 'train_loss': train_l})
  df_val = pd.DataFrame(data={'validation_steps':val_s, 'validation_loss': val_l,
                                'bleu_score': bleu, 'per_turn_accuracy': acc})
  df_params = pd.DataFrame(data={
      "Params": ['hidden-size', 'learning-rate', 'drop-prob', 'model-type', \
                'weight-decay', 'decay-times', 'attention-method'],
      "Values": [args.hidden_size, args.learning_rate, args.drop_prob, args.model_type, \
                args.weight_decay, args.decay_times, args.attn_method]})
  results_path = "results/{0}_{1}.csv".format(args.results_path, args.suffix)
  loss_history = pd.concat([df_train, df_val, df_params], axis=1)
  loss_history.to_csv(results_path, index=False)
  print('Loss, BLEU and accuracy saved to {}!'.format(results_path))

def batch_processing(batch_val_loss, batch_bleu, batch_success):
  avg_val_loss = sum(batch_val_loss) * 1.0 / len(batch_val_loss)
  avg_bleu = 100 * float(sum(batch_bleu)) / len(batch_bleu)
  avg_success = 100 * float(sum(batch_success)) / len(batch_success)

  print('Validation Loss: {0:2.4f}, BLEU Score: {1:.2f}, Per Turn Accuracy: {2:.2f}'.format(
          avg_val_loss, avg_bleu, avg_success))
  return avg_val_loss, avg_bleu, avg_success

def show_save_attention(visualizations, method, verbose):
  for i, viz in enumerate(visualizations):
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
    plt.savefig("results/visualize_{0}{1}.png".format(method, i))
    if verbose:
      plt.show()
    plt.close()

def test_mode_run(test_pairs, encoder, decoder, task):
  batch_test_loss, batch_bleu, batch_success = [], [], []
  bleu_scores, accuracy = [], []
  learner = LossTracker(-1)

  encoder.eval()
  decoder.eval()

  for test_pair in progress_bar(test_pairs):
    test_input = test_pair[0]
    test_output = test_pair[1]
    loss, predictions, visual = run_inference(encoder, decoder, test_input, \
                      test_output, criterion=NLLLoss(), teach_ratio=0)

    targets = test_output.data.tolist()
    predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
    target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

    test_loss = loss.data[0] / test_output.size()[0]
    bleu_score = BLEU.compute(predicted_tokens, target_tokens)
    turn_success = all([pred == tar[0] for pred, tar in zip(predictions, targets)])

    batch_test_loss.append(test_loss)
    batch_bleu.append(bleu_score)
    batch_success.append(turn_success)

  return batch_processing(batch_test_loss, batch_bleu, batch_success)
