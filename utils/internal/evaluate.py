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

def process(results, args):
  train_l, val_l, train_s, val_s = results
  df_train = pd.DataFrame(data={'train_steps':train_s, 'train_loss': train_l})
  df_val = pd.DataFrame(data={'validation_steps':val_s, 'validation_loss': val_l})
  df_params = pd.DataFrame(data={
      "Params": ['hidden-size', 'optimizer', 'drop-prob', \
                'weight-decay', 'decay-times', 'attention-type'],
      "Values": [args.hidden_size, args.optimizer, args.drop_prob, \
                args.weight_decay, args.decay_times, args.attention_type]})
  loss_history = pd.concat([df_train, df_val, df_params], axis=1)
  loss_history.to_csv(args.loss_path, index=False)
  print('Loss history saved!')