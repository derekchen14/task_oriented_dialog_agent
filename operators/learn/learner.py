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
    self.run_epochs(self.module, train_data, val_data)

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
            summary, unique_id = self.monitor.summarize_results(self.verbose)
            model.save(summary, unique_id)
            model.prune_saves()
            val_data.record_preds(preds=model.run_glad_inference(val_data),
                to_file=os.path.join(model.save_dir, 'dev.pred.json'))
          else:
            summary, _ = self.monitor.summarize_results(False)
          if self.monitor.should_early_stop():
            break
      self.monitor.end_epoch()

  def train(self, model, batch):
    model.train()   # set to training mode
    model.zero_grad()
    loss, scores = run_inference(model, batch)

    loss.backward()
    clip_gradient(model, clip=10)
    model.optimizer.step()
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

  def reinforce(self, params):
    """ main methods are run_episodes, store_experience, and next """
    reinforce_start_time = tm.time()
    self.logger.info('Starting reinforcement learning ...')
    self.success_rate_threshold = params.threshold

    planning_steps = 4
    self.num_sim_epochs = planning_steps + 1

    if params.warm_start:  #  TODO: check that a pretrained model doesn't already exist
      print('warm_start starting ...')
      self.warm_start_simulation()
      print('warm_start finished, start RL training ...')
    # self.module.user.goal_sets = self.processor.datasets
    # self.module.user.learning_phase = "train"
    self.run_episodes(params.epochs)

    self.logger.info("Done training {}".format(params.task))
    time_past(reinforce_start_time)

  def run_one_episode(self, collect_data=False):
    self.monitor.start_episode()
    self.module.initialize_episode(True)   # module is policy_manager

    episode_over = False
    while not episode_over:
      episode_over, reward = self.module.next(record_training_data_for_user=collect_data)
      self.monitor.status["episode_reward"] += reward
      self.monitor.status["turn_count"] += 1

      if episode_over:
        if self.monitor.status["episode_reward"] > 0:
          self.monitor.status["success"] = True
        self.monitor.end_episode()

  def run_episodes(self, num_episodes):
    dialog_manager = self.module
    self.performance_records = {'success_rate': [], 'ave_turns': [], 'ave_reward': []}
    # self.performance_records = defaultdict(list)
    self.best_model = {'model': copy.deepcopy(self.module.model)}
    self.best_res = {'success_rate': 0, 'ave_reward': float('-inf'),
            'ave_turns': float('inf'), 'epoch': 0}

    for episode in range(num_episodes):  #progress_bar(
      self.module.model.predict_mode = False
      self.run_one_episode()

      self.module.model.predict_mode = True
      self.module.world_model.predict_mode = True
      self.gather_data_for_world_training()

      self.module.model.predict_mode = False
      self.module.world_model.predict_mode = False
      simulation_res = self.simulation_epoch(50, episode)

      self.performance_records['success_rate'].append(simulation_res['success_rate'])
      self.performance_records['ave_turns'].append(simulation_res['ave_turns'])
      self.performance_records['ave_reward'].append(simulation_res['ave_reward'])

      if simulation_res['success_rate'] >= self.best_res['success_rate']:
        if simulation_res['success_rate'] >= self.success_rate_threshold:  # threshold = 0.30
          self.module.model.predict_mode = True
          self.module.world_model.predict_mode = True
          self.gather_data_for_world_training()

      if simulation_res['success_rate'] > self.best_res['success_rate']:
          self.best_model['model'] = copy.deepcopy(self.module.model)
          self.best_res['success_rate'] = simulation_res['success_rate']
          self.best_res['ave_reward'] = simulation_res['ave_reward']
          self.best_res['ave_turns'] = simulation_res['ave_turns']
          self.best_res['epoch'] = episode

      self.module.model.train(self.batch_size, 1, self.verbose)
      self.module.model.reset_dqn_target()

      # should probably be moved up?
      self.module.world_model.train(self.batch_size, 1)

      save_check_point = 10  # print every
      if episode > 0 and episode % save_check_point == 0:  # and params['trained_model_path'] == None: # save the model every 10 episodes
        print("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (
          self.performance_records['success_rate'][-1],
          self.performance_records['ave_reward'][-1],
          self.performance_records['ave_turns'][-1],self.best_res['success_rate']))

        self.module.save_model(self.best_model['model'], episode,
              self.best_res['epoch'], self.best_res['success_rate'])
        self.module.save_performance_records(self.performance_records)
        self.monitor.summarize_results(verbose=True)

    self.module.save_performance_records(self.performance_records)


  def simulation_epoch(self, simulation_epoch_size, episode_count):
    # uses the rulebased environment, trains the RL Agent
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    dialog_manager = self.module
    for episode in range(simulation_epoch_size):
      dialog_manager.initialize_episode(use_environment=True)
      episode_over = False
      while (not episode_over):
        episode_over, reward = dialog_manager.next(record_training_data_for_user=False)
        cumulative_reward += reward
        if episode_over:
          if reward > 0:
            successes += 1
          if reward > 0 and self.verbose:
            print("simulation episode %s: Success" % (episode))
          elif self.verbose:
            print("simulation episode %s: Fail" % (episode))
          cumulative_turns += dialog_manager.state_tracker.turn_count

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns) / simulation_epoch_size
    if episode_count > 0 and episode_count % 10 == 0:
      print("simulation success rate %s, ave reward %s, ave turns %s" % (
        res['success_rate'], res['ave_reward'], res['ave_turns']))

    return res

  def gather_data_for_world_training(self, grounded_for_model=False):
    # simulation_epoch_for_training / gathering examples for the world model
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    dialog_manager = self.module
    for episode in range(self.num_sim_epochs):
      use_environment = (episode % self.num_sim_epochs == 0)
      dialog_manager.initialize_episode(use_environment or grounded_for_model)

      episode_over = False
      while (not episode_over):
        # sometimes rulebased, and then neural world model
        episode_over, reward = dialog_manager.next()
        cumulative_reward += reward
        if episode_over:
          if reward > 0:
            successes += 1
          if reward > 0 and self.verbose:
            print("simulation episode %s: Success" % (episode))
          elif self.verbose:
            print("simulation episode %s: Fail" % (episode))
          cumulative_turns += dialog_manager.state_tracker.turn_count

    res['success_rate'] = float(successes) / self.num_sim_epochs
    res['ave_reward'] = float(cumulative_reward) / self.num_sim_epochs
    res['ave_turns'] = float(cumulative_turns) / self.num_sim_epochs
    # print("simulation success rate %s, ave reward %s, ave turns %s" % (
    #     res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res

  """ Warm_Start Simulation (by Rule Policy) """
  def warm_start_simulation(self):
    # uses rulebased environment
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0
    warm_start_epochs = 10
    dialog_manager = self.module

    for episode in range(warm_start_epochs):
      dialog_manager.initialize_episode(use_environment=True)
      episode_over = False
      while (not episode_over):
        episode_over, reward = dialog_manager.next()
        cumulative_reward += reward
        if episode_over:
          if reward > 0:
            successes += 1
          if reward > 0 and self.verbose:
            print("simulation episode %s: Success" % (episode))
          elif self.verbose:
            print("simulation episode %s: Fail" % (episode))
          cumulative_turns += dialog_manager.state_tracker.turn_count

      warm_start_run_epochs += 1
      # if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
      #     break
    self.module.world_model.train(self.batch_size, num_batches=5)

    self.module.model.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns) / warm_start_run_epochs
    print("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (
      episode + 1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    print("Current experience replay buffer size %s" % (len(self.module.model.experience_replay_pool)))

  def warm_start_simulation_preload(self):
    loader = self.processor.loader
    preloaded_buffer = loader.pickle_data('warm_up_experience_pool_seed3081_r5')
    self.module.model.experience_replay_pool = preloaded_buffer
    pre_examples = loader.pickle_data('warm_up_experience_pool_seed3081_r5_user')
    world_model.training_examples = pre_examples
    self.module.world_model.train(self.batch_size, 5)

    self.module.model.warm_start = 2
    print("Current experience replay buffer size %s" % (len(self.module.model.experience_replay_pool)))


  """
  def run_episodes(self, num_episodes):
    for episode in range(num_episodes):  #progress_bar(
      self.monitor = self.run_one_episode(self.monitor)
      self.run_simulations()
    self.monitor.summarize_results(True)

  def run_simulations(self):
    # run simulation to generate experiences that are stored in replay buffer
    if self.user == "command" or self.module.agent.model_type == "rulebased": return

    num_simulations = 3 if self.debug or self.verbose else 100
    if self.verbose: print("Running {} simulations".format(num_simulations))
    sim_monitor = RewardMonitor()
    for sim_episode in range(num_simulations):
      sim_monitor = self.run_one_episode(sim_monitor, collect_data=True)
    sim_monitor.summarize_results(self.verbose)

    if self.monitor.best_so_far(sim_monitor.success_rate):
      self.module.save_checkpoint(sim_monitor)

    best_model['model'] = copy.deepcopy(agent)
    best_res['success_rate'] = simulation_res['success_rate']
    best_res['ave_reward'] = simulation_res['ave_reward']
    best_res['ave_turns'] = simulation_res['ave_turns']
    best_res['epoch'] = episode
      agent.clone_dqn = copy.deepcopy(agent.dqn)
      agent.train(batch_size, 1)
      agent.predict_mode = False

    n_iters = 600 if self.debug else len(train_data)
    step_size = n_iters/(self.decay_times+1)
    enc_scheduler = StepLR(enc_optimizer, step_size=step_size, gamma=0.2)
    dec_scheduler = StepLR(dec_optimizer, step_size=step_size, gamma=0.2)
    enc_scheduler.step()
    dec_scheduler.step()
"""