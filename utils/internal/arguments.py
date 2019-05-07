from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def solicit_args():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('--seed', help='Random seed', type=int, default=14)
  parser.add_argument('-t', '--task', default='default', type=str,
              help='overall configuration of operations and objects in system', )
  parser.add_argument('-d', '--dataset', choices=['babi', 'woz2', 'dstc2',
              'e2e/movies', 'ddq/movies', 'e2e/restaurants'], default='woz2',
              help='Choose the data to train on, defines labels', )
  parser.add_argument('-m', '--model', default='ddq',
              help='Choose the model type used',)
  parser.add_argument('--use-old-nlu', default=False, action='store_true',
              help='Use the old nlg model for the Belief Tracker module')

  parser.add_argument('--debug', default=False, action='store_true',
              help='whether or not to go into debug mode, which is faster')
  parser.add_argument('-v', '--verbose', default=False, action='store_true',
              help='whether or not to have verbose prints')
  parser.add_argument('--test-mode', default=False, action='store_true',
              help='if true, then we are in test phase instead of train/val')
  parser.add_argument('--use-existing', default=False, action='store_true',
              help='continue training a model rather than creating a new one')
  parser.add_argument('--gpu', type=int, help='which GPU to use')

  # --------- TUNING OPTIONS -------------
  parser.add_argument('--context', default=False, action='store_true',
              help='if true, then include context as part of training input')
  parser.add_argument('--attn-method', default='luong', type=str,
              help='type of attention', choices=['luong', 'dot', 'vinyals'])
  parser.add_argument('--optimizer', default='sgd', type=str,
              help='Optimizer to use. Choose from sgd, rmsprop, adam')

  # ------ PARAMETER OPTIMIZATION --------
  param_group = parser.add_argument_group(title='hyperparameters')
  param_group.add_argument('-lr', '--learning-rate', default=0.01, type=float,
              help='Learning rate alpha for weight updates')
  param_group.add_argument('--hidden-dim', default=256, type=int,
              help='Number of hidden units, size of hidden dimension')
  param_group.add_argument('--embedding-size', default=300, type=int,
              help='Word embedding size usually from pretrained')
  param_group.add_argument('--drop-prob', default=0.2, type=float,
              help='probability of dropping a node, opposite of keep prob')
  param_group.add_argument('--teacher-forcing', default=0.6, type=float,
              help='teacher forcing ratio, 0 means no teacher forcing')
  param_group.add_argument('-reg', '--weight-decay', default=0.003, type=float,
              help='weight_decay to regularize the weights')
  param_group.add_argument('--num-layers', default=1, type=int,
              help='Number of layers in each LSTM')
  param_group.add_argument('--batch-size', default=50, type=int,
              help='batch size for training')
  param_group.add_argument('-e', '--epochs', default=14, type=int,
              help='Number of epochs or episodes to train')
  param_group.add_argument('--decay-times', default=3, type=int,
              help='total lr decay times')

  # --------- LIMITS AND THRESHOLDS -----------
  parser.add_argument('--max-turn', default=20, type=int,
              help='max allowed turns in dialogue before declaring failure')
  parser.add_argument('--max-seq-len', default=15, type=int,
              help='max number of tokens allowed to generate or to use as input')
  parser.add_argument('--threshold', default=0.8, type=float,
              help='minimum confidence level to keep, minimum success rate for \
              experience replay, minimum loss value we are willing to accept \
              for early stopping (with -1 to turn off), or other threshold')
  parser.add_argument('--early-stop', default='joint_goal', type=str,
              help='slot to report metrics on, used by monitor')

  # --------- REINFORCEMENT LEARNING ----------
  parser.add_argument('--discount-rate', default=0.9, type=float,
              help='discount rate for value, commonly known as gamma')
  parser.add_argument('--pool-size', default=1000, type=int,
              help='number of examples to hold in experience replay pool')
  parser.add_argument('--epsilon', default=0.1, type=float,
              help='Amount to start looking around in epsilon-greedy exploration')
  parser.add_argument('--warm-start', default=False, action='store_true',
              help='when true, agent has warm start phase for training')
  parser.add_argument('--user', default='simulate', type=str,
              help='type of user to talk to', choices=['simulate', 'command', 'turk'])
  parser.add_argument('--belief', default='discrete', type=str,
              help='type of belief state representation',
              choices=['discrete', 'memory', 'distributed'])

  # -------- MODEL CHECKPOINTS ----------------
  parser.add_argument('--save-model', default=False, action='store_true',
              help='when true, save model weights in a checkpoint')
  parser.add_argument('--pretrained', default=False, action='store_true',
              help='when true, use pretrained word embeddings from data directory')
  parser.add_argument('--prefix', default='', type=str,
              help='prepended string to distinguish among trials, usually date')
  parser.add_argument('--suffix', default='', type=str,
              help='appended string to distinguish among trials, usually count')

  # -------- REPORTING RESULTS ----------------
  parser.add_argument('--report-visual', default=False, action='store_true',
              help='when true, plot the train and val loss graph to file')
  parser.add_argument('--report-qual', default=False, action='store_true',
              help='when true, 50 random samples of inputs and outputs will be \
              printed out in human readable form for qualitative evaluation')
  parser.add_argument('--report-quant', default=False, action='store_true',
              help='when true, the values selected by the metrics argument will \
              be calculated, displayed and stored into a results.log file')
  parser.add_argument('--metrics', nargs='+', default=['accuracy'],
              choices=['bleu', 'rouge', 'meteor', 'accuracy', 'val_loss', \
              'macro_f1', 'micro_f1', 'avg_reward', 'avg_turn', 'success_rate'],
              help='list of evaluation metrics, each metric is a single float')

  return parser.parse_args()