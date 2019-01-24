
################################################################################
# load trained NLG model
################################################################################
nlg_model_path = params['nlg_model_path']
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs) # load nlg templates

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)


################################################################################
# load trained NLU model
################################################################################
nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)


################################################################################
# Dialog Manager
################################################################################
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, kb)

################################################################################
#   Run num_episodes Conversation Simulations
################################################################################
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size'] # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']


""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward':float('-inf'), 'ave_turns': float('inf'), 'epoch':0}
best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}


""" Save model """
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.p' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    if (agt == 9 or agt == 12 or agt == 13): checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    checkpoint['params'] = params
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print 'saved model in %s' % (filepath, )
    except Exception, e:
        print 'Error: Writing model fails: %s' % (filepath, )
        print e

""" save performance numbers """
def save_performance_records(path, agt, records):
    filename = 'agt_%s_performance_records.json' % (agt)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "wb"))
        print 'saved model in %s' % (filepath, )
    except Exception, e:
        print 'Error: Writing model fails: %s' % (filepath, )
        print e

""" Run N simulation Dialogues """
def simulation_epoch(simulation_epoch_size):
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
def warm_start_simulation():
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



def run_episodes(count, status):
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
    
    
run_episodes(num_episodes, status)
