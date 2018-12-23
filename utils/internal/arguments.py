from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def solicit_args():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('--seed', help='Random seed', type=int, default=14)
  parser.add_argument('-t', '--task', default='default', type=str,
              help='overall configuration of operations and objects in system', )
  parser.add_argument('-d', '--dataset', choices=['babi', 'in-car', 'woz2',
              'dstc2', 'frames', 'multiwoz', 'basicwoz'], default='frames',
              help='Choose the data to train on, defines labels', )
  parser.add_argument('-m', '--model', default='match', choices=['basic', \
              'attention', 'transformer', 'match', 'bilstm', 'glad'],
              help='Choose the model type used',)

  parser.add_argument('--debug', default=False, action='store_true',
              help='whether or not to go into debug mode, which is faster')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
              help='whether or not to have verbose prints')
  parser.add_argument('--test-mode', default=False, action='store_true',
              help='if true, then we are in test phase instead of train/val')
  parser.add_argument('--gpu', type=int, help='which GPU to use')

  # --------- TUNING OPTIONS -------------
  parser.add_argument('--context', default=False, action='store_true',
              help='if true, then include context as part of training input')
  parser.add_argument('--attn-method', default='luong', type=str,
              help='type of attention', choices=['luong', 'dot', 'vinyals'])
  parser.add_argument('--early-stop', default=-1.0, type=float,
              help='Minimum loss value we are willing to accept during epoch 10 \
                    at validation, set to negative value to prevent early stopping')
  parser.add_argument('--stop-early', default='joint_goal', type=str,
              help='slot to early stop on, used for GLAD')

  # ------ PARAMETER OPTIMIZATION --------
  parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
              help='Learning rate alpha for weight updates')
  parser.add_argument('--hidden-size', default=256, type=int,
              help='Number of hidden units in each LSTM')
  parser.add_argument('--embedding-size', default=300, type=int,
              help='Word embedding size usually from pretrained')
  parser.add_argument('--optimizer', default='sgd', type=str,
              help='Optimizer to use. Choose from sgd, rmsprop, adam')
  parser.add_argument('--drop-prob', default=0.2, type=float,
              help='probability of dropping a node, opposite of keep prob')
  parser.add_argument('--teacher-forcing', default=0.6, type=float,
              help='teacher forcing ratio, 0 means no teacher forcing')
  parser.add_argument('-reg', '--weight-decay', default=0.003, type=float,
              help='weight_decay to regularize the weights')
  parser.add_argument('--num-layers', default=1, type=int,
              help='Number of layers in each LSTM')
  parser.add_argument('--n-iters', default=30000, type=int,
              help='iterations to train')
  parser.add_argument('--batch_size', default=50, type=int,
              help='batch size for training')
  parser.add_argument('-e', '--epochs', default=14, type=int,
              help='Number of epochs to train, not compatible with n_iters')
  parser.add_argument('--decay-times', default=3, type=int,
              help='total lr decay times')

  # -------- MODEL CHECKPOINTS ----------------
  parser.add_argument('--save-model', default=False, action='store_true',
              help='when true, save model weights in a checkpoint')
  parser.add_argument('--use-existing', default=False, action='store_true',
              help='when true, we use an existing model rather than training a new one')
  parser.add_argument('--pretrained', default=False, action='store_true',
              help='when true, use pretrained word embeddings from data directory')
  parser.add_argument('--prefix', default='', type=str,
              help='prepended string to distinguish among trials, usually date')
  parser.add_argument('--suffix', default='', type=str,
              help='appended string to distinguish among trials, usually count')

  # -------- REPORTING RESULTS ----------------
  parser.add_argument('--report-results', default=False, action='store_true',
              help='when true, report the BLEU score, loss history, per dialog \
              and per turn accuracy in the results path file')
  parser.add_argument('--plot-results', default=False, action='store_true',
              help='when true, plot the loss graph in file')
  parser.add_argument('--visualize', default=0, type=int,
              help='randomly select x number of dialogues from validation set, \
              visualize the attention weights and store in results/visualize.png')
  parser.add_argument('--metrics', nargs='+', default=['accuracy'],
              help='list of evaluation metrics to run, such as accuracy \
              macro_f1, micro_f1, bleu, rouge, or inform_success')

  return parser.parse_args()