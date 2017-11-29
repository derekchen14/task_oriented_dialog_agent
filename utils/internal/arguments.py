import argparse

def solicit_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
  parser.add_argument('-t', '--task-name', help='Choose the task to train on', \
    choices=['1','2','3','4','5','dstc','concierge', 'schedule','navigate','weather'] )
  parser.add_argument('--hidden-size', default=256, type=int, help='Number of hidden units in each LSTM')
  parser.add_argument('-enp', '--encoder-path', default='car_en.pt', type=str,
    help='where to save encoder')
  parser.add_argument('-edp', '--decoder-path', default='car_de.pt', type=str,
    help='where to save decoder')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
    help='whether or not to have verbose prints')
  parser.add_argument('-s', '--save_results', default=False, action='store_true',
    help='whether or not we save model weights')
  parser.add_argument('-p', '--plot_results', default=False, action='store_true',
    help='when true, we save results awhether or not to have verbose prints')
  return parser.parse_args()
