import time as tm
import pdb, sys
import random
import logging

from torch.nn import NLLLoss as NegLL_Loss
from torch.optim.lr_scheduler import StepLR

from operators.evaluate import RewardMonitor
from utils.external.bleu import BLEU
from utils.internal.clock import *
from objects.components import *

class Learner(object):
  def __init__(self, args, module, processor, monitor, task=None):
    self.verbose = args.verbose
    self.debug = args.debug
    self.epochs = args.epochs
    self.decay_times = args.decay_times
    self.teach_ratio = args.teacher_forcing

    self.processor = processor
    self.monitor = monitor
    self.vocab = processor.vocab
    self.module = module
    self.task = task

  def train(self, input_var, output_var):
    self.model.train()   # affects the performance of dropout
    self.model.zero_grad()

    loss, _, _ = run_inference(self.model, input_var, output_var, \
                          self.criterion, self.teach_ratio)
    loss.backward()
    clip_gradient(self.model, clip=10)
    self.model.optimizer.step()

    return loss.item() / output_var.shape[0]

  def validate(self, input_var, output_var, task):
    self.model.eval()  # val period has no training, so teach ratio is 0
    loss, predictions, visual = run_inference(self.model, input_var, \
                      output_var, self.criterion, teach_ratio=0)

    queries = input_var.data.tolist()
    targets = output_var.data.tolist()

    # when task is not specified, it defaults to index_to_label
    predicted_tokens = [self.vocab.index_to_label(predictions[0])]
    query_tokens = [self.vocab.index_to_word(y) for y in queries]
    target_tokens = [self.vocab.index_to_label(z) for z in targets]

    avg_loss = loss.item() / output_var.shape[0]
    # bleu_score = 1 BLEU.compute(predicted_tokens, target_tokens)
    exact_success = (predictions[0].item() == targets[0])
    rank_success = targets[0] in predictions
    # return avg_loss, bleu_score, turn_success
    return avg_loss, exact_success, rank_success

  ''' Modified since predictions are now single classes rather than sentences
  predicted_tokens = [vocab.index_to_word(x, task) for x in predictions]
  query_tokens = [vocab.index_to_word(y[0], task) for y in queries]
  target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

  turn_success = [pred == tar[0] for pred, tar in zip(predictions, targets)]
  return avg_loss, bleu_score, all(turn_success)
  '''
  def supervise(self, params):
    supervise_start_time = tm.time()
    logging.info('Starting supervised learning ...')
    self.model.init_optimizer()
    self.criterion = NegLL_Loss()

    train_data = self.processor.datasets['train']
    val_data = self.processor.datasets['val']
    self.run_epochs(train_data, val_data)

    logging.info("Done training {}".format(params.task))
    time_past(supervise_start_time)

  def run_epochs(self, train_data, val_data):
    """ main methods are run_epochs, train, validate, predict, and inference """
    print_every, plot_every, val_every = print_frequency(self.verbose, self.debug)
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for epoch in range(self.epochs):
      start = tm.time()
      starting_checkpoint(epoch, self.epochs, use_cuda)
      for i, train_pair in enumerate(self.processor.datasets['train']):
        input_var, output_var = train_pair
        loss = self.train(input_var, output_var)
        print_loss_total += loss
        plot_loss_total += loss

        if i > 0 and i % print_every == 0:
          self.monitor.train_steps.append(i + 1)
          print_loss_avg = print_loss_total / print_every
          print_loss_total = 0  # reset the print loss
          print('{1:3.1f}% complete {2}, Train Loss: {0:.4f}'.format(
              print_loss_avg, (i/n_iters * 100.0), timeSince(start, i/n_iters )))
          self.monitor.update_loss(print_loss_avg, "train")

        if i > 0 and i % val_every == 0:
          self.monitor.val_steps.append(i + 1)
          batch_val_loss, batch_bleu, batch_success = [], [], []
          for val_input, val_output in self.processor.datasets['val']:
            val_loss, bs, ts = self.validate(val_input, val_output, task)
            batch_val_loss.append(val_loss)
            batch_bleu.append(bs)
            batch_success.append(ts)

          self.monitor.batch_processing(batch_val_loss, batch_bleu, batch_success)
          if self.monitor.should_early_stop(i):
            print("Early stopped at val epoch {}".format(self.monitor.val_epoch))
            self.monitor.completed_training = False
            break
      if self.monitor.best_so_far() and not self.debug:
        summary = self.monitor.generate_summary()
        identifier = "epoch={0}_success={1:.4f}_recall@two={2:.4f}".format(
              epoch, summary["accuracy"], summary["recall@k=2"])
        self.model.save(summary, identifier)


  """ Warm_Start Simulation (by Rule Policy) """
  def warm_start_simulation(self):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0
    for episode in range(warm_start_epochs):
      dialog_manager.initialize_episode()
      episode_over = False
      while(not episode_over):
        episode_over, reward = dialog_manager.next(collect_data=True)
        cumulative_reward += reward
        if episode_over:
          if reward > 0:
            successes += 1
            print ("warm_start simulation episode %s: Success" % (episode))
          else: print ("warm_start simulation episode %s: Fail" % (episode))
          cumulative_turns += dialog_manager.state_tracker.turn_count

      warm_start_run_epochs += 1

      if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
        break

    agent.warm_start = 2
    res['success_rate'] = float(successes)/warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward)/warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns)/warm_start_run_epochs
    print ("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (episode+1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    print ("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))

  def reinforce(self, params):
    """ main methods are run_episodes, store_experience, and next """
    reinforce_start_time = tm.time()
    logging.info('Starting reinforcement learning ...')
    self.success_rate_threshold = params.threshold

    if params.warm_start:  #  TODO: check that a pretrained model doesn't already exist
      warm_start_simulation()

    self.module.user.goal_sets = self.processor.datasets
    self.module.user.learning_phase = "train"
    self.run_episodes(params.epochs)

    logging.info("Done training {}".format(params.task))
    time_past(reinforce_start_time)

  def run_one_episode(self, monitor, collect_data=False):
    monitor.start_episode()
    self.module.initialize_episode()   # module is policy_manager

    episode_over = False
    while not episode_over:
      episode_over, reward = self.module.next(collect_data)
      monitor.status["episode_reward"] += reward
      monitor.status["turn_count"] += 1

      if episode_over:
        if monitor.status["episode_reward"] > 0:
          monitor.status["success"] = True
        monitor.end_episode()

    return monitor

  def run_episodes(self, num_episodes):
    print("Running {} training episodes".format(num_episodes))
    for episode in progress_bar(range(num_episodes)):
      self.monitor = self.run_one_episode(self.monitor)

      # run simulation to generate experiences that are stored in replay buffer
      num_simulations = 3 if self.debug else 100
      if self.verbose: print("Running {} simulations".format(num_simulations))
      sim_monitor = RewardMonitor(num_simulations)
      for sim_episode in range(num_simulations):
        sim_monitor = self.run_one_episode(sim_monitor, collect_data=True)
      sim_monitor.summarize_results(self.verbose)

      if self.monitor.best_so_far(sim_monitor.success_rate):
        self.module.save_checkpoint(sim_monitor, episode)

        # best_model['model'] = copy.deepcopy(agent)
        # best_res['success_rate'] = simulation_res['success_rate']
        # best_res['ave_reward'] = simulation_res['ave_reward']
        # best_res['ave_turns'] = simulation_res['ave_turns']
        # best_res['epoch'] = episode
      # agent.clone_dqn = copy.deepcopy(agent.dqn)
      # agent.train(batch_size, 1)
      # agent.predict_mode = False
    self.monitor.summarize_results(True)
    # self.verbose


"""
    n_iters = 600 if self.debug else len(train_data)
    step_size = n_iters/(self.decay_times+1)
    enc_scheduler = StepLR(enc_optimizer, step_size=step_size, gamma=0.2)
    dec_scheduler = StepLR(dec_optimizer, step_size=step_size, gamma=0.2)
    enc_scheduler.step()
    dec_scheduler.step()
"""