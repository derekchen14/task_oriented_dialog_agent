import argparse

def solicit_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
  parser.add_argument('-t', '--task-name', choices=['1','2','3','4','5', \
              'challenge', 'concierge','schedule', 'navigate','weather'],
              help='Choose the task to train on', )
  parser.add_argument('-m', '--model-type', default='match', choices=['basic', \
              'gru', 'attention', 'match', 'copy', 'transformer', 'memory'],
              help='Choose the model type used',)
  parser.add_argument('--debug', default=False, action='store_true',
              help='whether or not to go into debug mode, which is faster')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
              help='whether or not to have verbose prints')

  # --------- TUNING OPTIONS -------------
  parser.add_argument('--early-stopping', default=-1.0, type=float,
              help='Minimum loss value we are willing to accept during epoch 10 \
                    at validation, set to negative value to prevent early stopping')
  parser.add_argument('--trials-per-setting', default=2, type=int,
              help='Number of trials per parameter setting, appends a letter to \
                    the end of each report or checkpoint to differentiate runs')

  # ------ PARAMETER OPTIMIZATION --------
  parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
              help='Learning rate alpha for weight updates')
  parser.add_argument('--hidden-size', default=256, type=int,
              help='Number of hidden units in each LSTM')
  parser.add_argument('--optimizer', default='SGD', type=str,
              help='Optimizer we want to use')
  parser.add_argument('--drop-prob', default=0.2, type=float,
              help='probability of dropping a node')
  parser.add_argument('--teacher-forcing', default=0.6, type=float,
              help='teacher forcing ratio, 0 means no teacher forcing')
  parser.add_argument('-reg', '--weight-decay', default=0.003, type=float,
              help='weight_decay to regularize the weights')
  parser.add_argument('--n-layers', default=1, type=int,
              help='Number of layers in each LSTM')
  parser.add_argument('--n-iters', default=25000, type=int,
              help='iterations to train')
  parser.add_argument('--decay-times', default=3, type=int,
              help='total lr decay times')
  parser.add_argument('--attn-method', default='luong', type=str,
              help='type of attention', choices=['luong', 'dot', 'vinyals'])

  # -------- MODEL CHECKPOINTS ----------------
  parser.add_argument('--save-model', default=False, action='store_true',
              help='when true, save model weights in a checkpoint')
  parser.add_argument('--use-existing', default=False, action='store_true',
              help='when true, we use an existing model rather than training a new one')
  parser.add_argument('--encoder-path', default='results/1_en.pt', type=str,
              help='where to save or load the encoder')
  parser.add_argument('--decoder-path', default='results/1_de.pt', type=str,
              help='where to save or load the decoder')

  # -------- REPORTING RESULTS ----------------
  parser.add_argument('--report-results', default=False, action='store_true',
              help='when true, report the BLEU score, loss history, per dialog \
              and per turn accuracy in the results path file')
  parser.add_argument('--plot-results', default=False, action='store_true',
              help='when true, plot the loss graph in file')
  parser.add_argument('--visualize', default=0, type=int,
              help='randomly select x number of dialogues from validation set, \
              visualize the attention weights and store in results/visualize.png')
  parser.add_argument('--results-path', default='results/model_results1.csv',
              help='where to save error', type=str)

  return parser.parse_args()