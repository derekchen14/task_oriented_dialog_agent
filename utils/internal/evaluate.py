import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import pdb

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

def create_report(results, args):
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
  loss_history = pd.concat([df_train, df_val, df_params], axis=1)
  loss_history.to_csv(args.results_path, index=False)
  print('Loss, BLEU and accuracy saved to {}!'.format(args.results_path))

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