import argparse

def solicit_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
  parser.add_argument('-t', '--task-name', help='Choose the task to train on',
              choices=['1','2','3','4','5','challenge','concierge','schedule', \
                  'navigate','weather'] )
  parser.add_argument('--debug', default=False, action='store_true',
              help='whether or not to go into debug mode, which is faster')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
              help='whether or not to have verbose prints')

  # ------ PARAMETER OPTIMIZATION --------
  parser.add_argument('--hidden-size', default=256, type=int,
              help='Number of hidden units in each LSTM')
  parser.add_argument('--optimizer', default='SGD', type=str,
              help='Optimizer we want to use')
  parser.add_argument('--drop-prob', default=0.5, type=float,
              help='probability of dropping a node')
  parser.add_argument('--teacher-forcing', default=0.5, type=float,
              help='teacher forcing ratio, 0 means no teacher forcing')
  parser.add_argument('-w', '--weight-decay', default=0.001, type=float,
              help='weight_decay with default 0.001')
  parser.add_argument('--n-layers', default=1, type=int,
              help='Number of layers in each LSTM')
  parser.add_argument('--n-iters', default=25000, type=int,
              help='iterations to train')
  parser.add_argument('--decay-times', default=2, type=int,
              help='total lr decay times')
  parser.add_argument('--attention-type', default='general', type=str,
              help='type of attention', choices=['general', 'dot', 'special'])

  # -------- SAVING RESULTS ----------------
  parser.add_argument('--save-model', default=False, action='store_true',
              help='when true, save model weights in a checkpoint')
  parser.add_argument('--save-loss', default=False, action='store_true',
              help='when true, save the train and loss curves')
  parser.add_argument('--plot-results', default=False, action='store_true',
              help='when true, plot the loss graph in file')
  parser.add_argument('--encoder-path', default='results/1_en.pt', type=str,
              help='where to save encoder')
  parser.add_argument('--decoder-path', default='results/1_de.pt', type=str,
              help='where to save decoder')
  parser.add_argument('--loss-path', default='results/1_loss_history.csv', type=str,
              help='where to save error')

  return parser.parse_args()