import time as tm
import os, pdb, sys
import random
import copy

# from torch.optim.lr_scheduler import StepLR
from operators.evaluate import RewardMonitor
from utils.external.bleu import BLEU
from utils.internal.clock import *
from objects.components import *

class Learner(object):
  def __init__(self, args, module, processor, monitor, task=None):
    self.verbose = args.verbose
    self.debug = args.debug
    self.epochs = args.epochs
    self.batch_size = args.batch_size
    self.user = args.user
    self.decay_times = args.decay_times
    self.teach_ratio = args.teacher_forcing

    self.processor = processor
    self.monitor = monitor
    self.logger = monitor.logger
    self.vocab = processor.vocab
    self.module = module
    self.task = task

  def supervise(self, params):
    supervise_start_time = tm.time()
    self.logger.info('Starting supervised learning for {} epochs ...'.format(self.epochs))
    self.module.init_optimizer()

    train_data = self.processor.datasets['train']
    val_data = self.processor.datasets['val']
    self.run_epochs(self.module.model, train_data, val_data)

    self.logger.info("Done training {}".format(params.task))
    time_past(supervise_start_time)

  def run_epochs(self, model, train_data, val_data):
    """ main methods are run_epochs, train, validate, predict, and inference """
    for epoch in range(self.epochs):
      val_every = self.monitor.start_epoch(self.debug, epoch, self.logger)
      starting_checkpoint(epoch, self.epochs, use_cuda)
      for batch in train_data.batch(self.batch_size, shuffle=True):
        loss = self.train(model, batch)
        self.monitor.update_train(loss)

        if self.monitor.time_to_validate():
          val_results = self.validate(model, val_data)
          train_results = self.validate(model, train_data, self.verbose)
          self.monitor.update_val(val_results, train_results)
          if self.monitor.best_so_far():
            self.monitor.summarize_results(self.verbose)
            model.save_checkpoint(self.monitor)
            model.prune_saves()
            val_data.record_preds(preds=model.run_glad_inference(val_data),
                to_file=os.path.join(model.save_dir, 'dev.pred.json'))
          else:
            self.monitor.summarize_results(False)
          if self.monitor.should_early_stop():
            break
      self.monitor.end_epoch()

  def train(self, model, batch):
    model.train()   # set to training mode
    model.zero_grad()
    loss, scores = run_inference(model, batch)

    loss.backward()
    clip_gradient(model, clip=10)
    self.module.optimizer.step()
    return loss.item()

  def validate(self, model, val_data, this_time=True):
    if not this_time: return {}
    model.eval()  # val period has no training, so teach ratio is 0
    predictions = []

    for val_batch in val_data.batch(self.batch_size):
      loss, scores = run_inference(model, val_batch)
      predictions += model.extract_predictions(scores)

    # TODO: user the summary method to get finer control of metrics returned
    # return self.monitor.summarize_results(self.logger)
    return val_data.evaluate_preds(predictions)


  ''' Modified since predictions are now single classes rather than sentences
  predicted_tokens = [vocab.index_to_word(x, task) for x in scores]
  query_tokens = [vocab.index_to_word(y[0], task) for y in queries]
  target_tokens = [vocab.index_to_word(z[0], task) for z in targets]

  turn_success = [pred == tar[0] for pred, tar in zip(scores, targets)]
  avg_loss = loss.item() / output_var.shape[0]
  self.monitor.update_val(loss, metrics)
  exact_success = (predictions[0].item() == targets[0])
  rank_success = targets[0] in predictions '''

  def end_to_end(self, params):
    # just a naive wrapper since end to end learning is reinforcement learning
    self.reinforce(params)

  def reinforce(self, params):
    """ main methods are run_episodes, store_experience, and next """
    reinforce_start_time = tm.time()
    self.logger.info('Starting reinforcement learning ...')
    self.success_rate_threshold = params.threshold

    if params.warm_start:  #  TODO: check that a pretrained model doesn't already exist
      self.module.run_mode = 3
      self.warm_start_simulation()
    # self.module.user.goal_sets = self.processor.datasets
    # self.module.user.learning_phase = "train"
    self.run_episodes(params.epochs)

    self.logger.info("Done training {}".format(params.task))
    time_past(reinforce_start_time)

  def run_one_episode(self, monitor, simulator_type, collect_data=False):
    monitor.start_episode()
    self.module.initialize_episode(simulator_type)   # module is policy_manager
    self.module.start_conversation(simulator_type)

    episode_over = False
    while not episode_over:
      episode_over, reward = self.module.next(record_user_data=collect_data)
      monitor.status["episode_reward"] += reward
      monitor.status["turn_count"] += 1

      if episode_over:
        if monitor.status["episode_reward"] > 0:
          monitor.status["success"] = True
    monitor.end_episode()

  def run_episodes(self, num_episodes, planning_steps=5):
    """ Run loop for training the RL agent
      1) run exactly one episode of training real agent
      2) gather data with with rule user_sim and neural user_sim (ie. world)
      3) Update weights and train the two user simulators
      4) Occasionally summarize results and save checkpoints
    """
    for episode in range(num_episodes):
      self.module.model.predict_mode = False
      self.run_one_episode(self.monitor, 'rule')

      # Gather examples for world and planning together
      self.module.model.predict_mode = True
      self.module.world_model.predict_mode = True
      self.gather_data_for_user(planning_steps, episode)

      # For gathering validation examples
      self.module.model.predict_mode = False
      self.module.world_model.predict_mode = False
      self.gather_data_for_agent(50, episode)

      # extra boost for improving performance by filling buffer
      simulation_success_rate = self.monitor.simulation_successes[-1]
      if simulation_success_rate > self.monitor.best_success_rate:
        self.monitor.summarize_results()
        self.module.model.save_checkpoint(self.monitor)
        self.module.save_performance_records(self.monitor)

      if self.monitor.best_so_far(simulation_success_rate):
        self.module.model.predict_mode = True
        self.module.world_model.predict_mode = True
        self.gather_data_for_user(planning_steps, episode)

      self.module.model.train(self.batch_size, 1, self.verbose)
      self.module.model.reset_dqn_target()

      self.module.world_model.train(self.batch_size, 1)
      self.monitor.summarize_results(episode % 14 == 0)

    self.monitor.summarize_results()
    self.module.model.save_checkpoint(self.monitor)
    self.module.save_performance_records(self.monitor)

  # Use neural-based environment to gather data for training the user simulator
  def gather_data_for_user(self, num_episodes, global_episode):
    user_monitor = RewardMonitor(['success_rate'])
    for episode in range(num_episodes):
      planning_steps = 5
      if episode % planning_steps == 0:
        self.run_one_episode(user_monitor, 'rule', collect_data=True)
      else:
        self.run_one_episode(user_monitor, 'neural', collect_data=True)
    if global_episode > 0 and global_episode % 10 == 0:
      user_monitor.summarize_results(self.verbose, "World Results - ")

  # Use rule-based environment to gather data for training the RL agent
  def gather_data_for_agent(self, num_episodes, global_episode):
    # if self.monitor.success_rate > 0.7:
    #   self.module.run_mode = 0
    agent_monitor = RewardMonitor(['success_rate'])
    for episode in range(num_episodes):
      self.run_one_episode(agent_monitor, 'rule')
      if self.module.run_mode == 0 and episode > 10:
        pdb.set_trace()
    report_results = (global_episode % 10 == 0)
    agent_monitor.summarize_results(report_results, "Simulation Results - ")
    self.monitor.simulation_successes.append(agent_monitor.success_rate)

  # Use rule-based environment to gather pre-training warm start data
  def warm_start_simulation(self, num_episodes=100):
    """  Load pre-computed training data for warm start
    loader = self.processor.loader
    preloaded_buffer = loader.pickle_data('warm_up_experience_pool_seed3081_r5')
    self.module.model.experience_replay_pool = preloaded_buffer
    pre_examples = loader.pickle_data('warm_up_experience_pool_seed3081_r5_user')
    world_model.training_examples = pre_examples
    """
    print("Collect data from warm start")    # self.logger.info
    warm_monitor = RewardMonitor(['success_rate'])
    for episode in range(num_episodes):
      # ensure exeprience replay pool is capped to max len using a deque
      self.run_one_episode(warm_monitor, 'rule', collect_data=True)
    warm_monitor.summarize_results(self.verbose, "Warm Start Results - ")

    self.module.world_model.train(self.batch_size, num_batches=5)
    self.module.model.warm_start = 2
    buffer_size = len(self.module.model.experience_replay_pool)
    print("Current experience replay buffer size {}".format(buffer_size))


"""
    n_iters = 600 if self.debug else len(train_data)
    step_size = n_iters/(self.decay_times+1)
    enc_scheduler = StepLR(enc_optimizer, step_size=step_size, gamma=0.2)
    dec_scheduler = StepLR(dec_optimizer, step_size=step_size, gamma=0.2)
    enc_scheduler.step()
    dec_scheduler.step()
"""
