import argparse

def solicit_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
  parser.add_argument('-t', '--task-name', help='Choose the task to train on',
    choices=['1','2','3','4','5','challenge','concierge','schedule','navigate','weather'] )
  parser.add_argument('--hidden-size', default=256, type=int,
    help='Number of hidden units in each LSTM')

  parser.add_argument('-op', '--optimzier', default='SGD', type=str, help='Optimizer we want to use')

  parser.add_argument('--n-layers', default=1, type=int,
                      help='Number of layers in each LSTM')
  parser.add_argument('--drop-prob', default=0.5, type=float,
    help='probability of dropping a node')
  parser.add_argument('-teach', '--teacher-forcing', default=0.5, type=float, help='teacher forcing ratio')
  # 0 means no teacher forcing
  parser.add_argument('-wd', '--weight_decay', default=0.001, type=float, help='weight_decay')


  parser.add_argument('-enp', '--encoder-path', default='results/1_en.pt', type=str,
    help='where to save encoder')
  parser.add_argument('-edp', '--decoder-path', default='results/1_de.pt', type=str,
    help='where to save decoder')
  parser.add_argument('-ep', '--error-path', default='results/1_error.csv', type=str,
                      help='where to save error')
  parser.add_argument('--n_iters', default=25000, type=int,
                      help='iterations to train')
  parser.add_argument('--decay_times', default=2, type=int, help='total lr decay times')
  parser.add_argument('--debug', default=False, action='store_true',
    help='whether or not to go into debug mode, which is faster')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
    help='whether or not to have verbose prints')
  parser.add_argument('-s', '--save-results', default=False, action='store_true',
    help='whether or not we save model weights')
  parser.add_argument('-p', '--plot-results', default=False, action='store_true',
    help='when true, we save results awhether or not to have verbose prints')


  return parser.parse_args()
