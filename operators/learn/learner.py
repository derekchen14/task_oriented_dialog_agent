import time as tm
import pdb, sys
import random
import logging

from torch.nn import NLLLoss as NegLL_Loss
from torch.optim.lr_scheduler import StepLR

from utils.external.bleu import BLEU
from utils.internal.clock import *
from objects.components import *

class Learner(object):
  def __init__(self, args, model, processor, tracker, task=None):
    self.verbose = args.verbose
    self.debug = args.debug
    self.epochs = args.epochs
    self.decay_times = args.decay_times
    self.teach_ratio = args.teacher_forcing

    self.processor = processor
    self.tracker = tracker
    self.vocab = processor.vocab
    self.model = model
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

  def rulebased(self, task, module, data):
    """ main method is simple next """
    self.run_episodes(count, status)

  def reinforce(self, task, module, data):
    """ main methods are store_experience, next, learn """
    pass

  def supervise(self, task, module, data):
    """ main methods are train, validate, and inference """
    self.learn_start = tm.time()
    logging.info('Starting to learn ...')
    self.model.init_optimizer()
    self.criterion = NegLL_Loss()

    n_iters = 600 if self.debug else len(self.processor.datasets['train'])
    print_every, plot_every, val_every = print_frequency(self.verbose, self.debug)
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # step_size = n_iters/(self.decay_times+1)
    # enc_scheduler = StepLR(enc_optimizer, step_size=step_size, gamma=0.2)
    # dec_scheduler = StepLR(dec_optimizer, step_size=step_size, gamma=0.2)
    # enc_scheduler.step()
    # dec_scheduler.step()

    for epoch in range(self.epochs):
      start = tm.time()
      starting_checkpoint(epoch, self.epochs, use_cuda)
      for i, train_pair in enumerate(self.processor.datasets['train']):
        input_var, output_var = train_pair
        loss = self.train(input_var, output_var)
        print_loss_total += loss
        plot_loss_total += loss

        if i > 0 and i % print_every == 0:
          self.tracker.train_steps.append(i + 1)
          print_loss_avg = print_loss_total / print_every
          print_loss_total = 0  # reset the print loss
          print('{1:3.1f}% complete {2}, Train Loss: {0:.4f}'.format(
              print_loss_avg, (i/n_iters * 100.0), timeSince(start, i/n_iters )))
          self.tracker.update_loss(print_loss_avg, "train")

        if i > 0 and i % val_every == 0:
          self.tracker.val_steps.append(i + 1)
          batch_val_loss, batch_bleu, batch_success = [], [], []
          for val_input, val_output in self.processor.datasets['val']:
            val_loss, bs, ts = self.validate(val_input, val_output, task)
            batch_val_loss.append(val_loss)
            batch_bleu.append(bs)
            batch_success.append(ts)

          self.tracker.batch_processing(batch_val_loss, batch_bleu, batch_success)
          if self.tracker.should_early_stop(i):
            print("Early stopped at val epoch {}".format(self.tracker.val_epoch))
            self.tracker.completed_training = False
            break
      if self.tracker.best_so_far() and not self.debug:
        summary = self.tracker.generate_summary()
        identifier = "epoch={0}_success={1:.4f}_recall@two={2:.4f}".format(
              epoch, summary["accuracy"], summary["recall@k=2"])
        self.model.save(summary, identifier)

    logging.info("Done training {}".format(task))
    time_past(self.learn_start)

  def simulation_epoch(self, simulation_epoch_size):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in xrange(simulation_epoch_size):
      dialog_manager.initialize_episode()
      episode_over = False
      while(not episode_over):
        episode_over, reward = dialog_manager.next_turn()
        cumulative_reward += reward
        if episode_over:
          if reward > 0:
            successes += 1
            print ("simulation episode %s: Success" % (episode))
          else: print ("simulation episode %s: Fail" % (episode))
          cumulative_turns += dialog_manager.state_tracker.turn_count

    res['success_rate'] = float(successes)/simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward)/simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns)/simulation_epoch_size
    print ("simulation success rate %s, ave reward %s, ave turns %s" % (res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res

  """ Warm_Start Simulation (by Rule Policy) """
  def warm_start_simulation(self):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0
    for episode in xrange(warm_start_epochs):
      dialog_manager.initialize_episode()
      episode_over = False
      while(not episode_over):
        episode_over, reward = dialog_manager.next_turn()
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

  def run_episodes(self, count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    if (agt == 9 or agt == 12 or agt == 13) and params['trained_model_path'] == None and warm_start == 1:
      print ('warm_start starting ...')
      warm_start_simulation()
      print ('warm_start finished, start RL training ...')

    for episode in xrange(count):
      print ("Episode: %s" % (episode))
      dialog_manager.initialize_episode()
      episode_over = False

      while(not episode_over):
        episode_over, reward = dialog_manager.next_turn()
        cumulative_reward += reward

        if episode_over:
          if reward > 0:
            print ("Successful Dialog!")
            successes += 1
          else: print ("Failed Dialog!")

          cumulative_turns += dialog_manager.state_tracker.turn_count

      # simulation
      if (agt == 9 or agt == 12 or agt == 13) and params['trained_model_path'] == None:
        agent.predict_mode = True
        simulation_res = simulation_epoch(simulation_epoch_size)

        performance_records['success_rate'][episode] = simulation_res['success_rate']
        performance_records['ave_turns'][episode] = simulation_res['ave_turns']
        performance_records['ave_reward'][episode] = simulation_res['ave_reward']

        if simulation_res['success_rate'] >= best_res['success_rate']:
          if simulation_res['success_rate'] >= success_rate_threshold: # threshold = 0.30
            agent.experience_replay_pool = []
            simulation_epoch(simulation_epoch_size)

        if simulation_res['success_rate'] > best_res['success_rate']:
          best_model['model'] = copy.deepcopy(agent)
          best_res['success_rate'] = simulation_res['success_rate']
          best_res['ave_reward'] = simulation_res['ave_reward']
          best_res['ave_turns'] = simulation_res['ave_turns']
          best_res['epoch'] = episode

        agent.clone_dqn = copy.deepcopy(agent.dqn)
        agent.train(batch_size, 1)
        agent.predict_mode = False

        print ("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (performance_records['success_rate'][episode], performance_records['ave_reward'][episode], performance_records['ave_turns'][episode], best_res['success_rate']))
        if episode % save_check_point == 0 and params['trained_model_path'] == None: # save the model every 10 episodes
          save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], episode)
          save_performance_records(params['write_model_dir'], agt, performance_records)

      print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (episode+1, count, successes, episode+1, float(cumulative_reward)/(episode+1), float(cumulative_turns)/(episode+1)))
    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (successes, count, float(cumulative_reward)/count, float(cumulative_turns)/count))
    status['successes'] += successes
    status['count'] += count

    if (agt == 9 or agt == 12 or agt == 13)  and params['trained_model_path'] == None:
      save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], count)
      save_performance_records(params['write_model_dir'], agt, performance_records)