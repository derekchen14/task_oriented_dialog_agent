import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

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
  train_s, train_l, val_s, val_l, bleu, acc = results
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

def visualize_attention(results):
  output_words, attentions = evaluate("je suis trop froid .")
  plt.matshow(attentions.numpy())

def show_attention(input_sentence, output_words, attentions):
  # Set up figure with colorbar
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.numpy(), cmap='bone')
  fig.colorbar(cax)
  # Set up axes
  ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
  ax.set_yticklabels([''] + output_words)
  # Show label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
  plt.close()

def evaluate_and_show_attention(input_sentence):
  output_words, attentions = evaluate(input_sentence)
  print('input =', input_sentence)
  print('output =', ' '.join(output_words))
  show_attention(input_sentence, output_words, attentions)