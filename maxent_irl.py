'''
Implementation of maximum entropy inverse reinforcement learning in
  Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
  https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

Acknowledgement:
  This implementation is largely influenced by Matthew Alger's maxent implementation here:
  https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py

By Yiren Lu (luyirenmax@gmail.com), May 2017
'''
import numpy as np
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
from utils import *
import copy
import pandas as pd
import csv
import mod_trajectory as tj
import bad_trajectory as bdtj



R_DISCOUNT = 0.1
R_RAISE = 0.1
GAMMA = 0.8
ACT_RAND = 0.1
H = 7
W = 7

def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T)
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


  returns:
    p       Nx1 vector - state visitation frequencies
  """


  N_STATES, _, N_ACTIONS = np.shape(P_a)


  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T])

  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1


  mu[:,0] = mu[:,0]/len(trajs)

  for s in range(N_STATES):
    for t in range(T-1):
      if deterministic:
        mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])

  p = np.sum(mu, 1)
  return p

def mod_feat_exp(feat_map,trajs):
  feat_exp = np.zeros([feat_map.shape[1]])
  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[step.cur_state,:]
  feat_exp = feat_exp/len(trajs)
  return feat_exp

def generate_newtrajs(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
  """gatheres expert demonstrations

  inputs:
  gw          Gridworld - the environment
  policy      Nx1 matrix
  n_trajs     int - number of trajectories to generate
  rand_start  bool - randomly picking start position or not
  start_pos   2x1 list - set start position, default [0,0]
  returns:
  trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
  """

  trajs = []
  for i in range(n_trajs):
    if rand_start:
      # override start_pos
      start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

    episode = []
    gw.reset(start_pos)
    cur_state = start_pos
    cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
    episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
    # while not is_done:
    for _ in range(len_traj):
        cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(next_state)]))
        episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
        if is_done:
            break
    trajs.append(episode)
  return trajs

def optimal_direction(state,policy, Width, Height):
  #print "policy",policy

  if(int(policy) == 0):
    if(Height*Width - Height < state < Height*Width):
      state = state
    else:
      state += Height
  elif(int(policy) == 1):
    if(state < Height):
      state = state
    else:
      state -= Height
  elif(int(policy) == 2):
    if((state % (Height-1)) == 0):
      state = state
    else:
      state += 1
  elif(int(policy) == 3):
    if(state % Height == 0):
      state = state
    else:
      state -= 1
  else:
    state = state
  return state

def e_greedy_direction(state, policy, Width, Height, epsilon):
  rn = np.random.rand()
  #print "random", rn
  if(rn < epsilon):
    if(state == 0):
      randlist = [0,2,4]
      policy = np.random.choice(randlist)
    elif(state == 6):
      randlist = [0,3,4]
      policy = np.random.choice(randlist)
    elif(state == 42):
      randlist = [1,2,4]
      policy = np.random.choice(randlist)
    elif(state == 48):
      randlist = [1,3,4]
      policy = np.random.choice(randlist)
    elif(0 < state < 6):
      randlist = [0,2,3,4]
      policy = np.random.choice(randlist)
    elif((state % 7) == 0):
      randlist = [0,1,2,4]
      policy = np.random.choice(randlist)
    elif((state % 6) == 0):
      randlist = [0,1,3,4]
      policy = np.random.choice(randlist)
    elif(42 < state < 48):
      randlist = [1, 2, 3, 4]
      policy = np.random.choice(randlist)
    else:
      policy = np.random.randint(0,4)

  if(int(policy) == 0):
    if(Height*Width - Height < state < Height*Width):
      state = state
    else:
      state += Height
  elif(int(policy) == 1):
    if(state < Height):
      state = state
    else:
      state -= Height
  elif(int(policy) == 2):
    if((state % (Height-1)) == 0):
      state = state
    else:
      state += 1
  elif(int(policy) == 3):
    if(state % Height == 0):
      state = state
    else:
      state -= 1
  else:
    state = state

  return state



def get_optimaltrajectory(policy, Height, Width, Length):
  opt_traj=[]
  state = 0
  opt_traj.append(0)

  while(state != (Height * Width - 1) and len(opt_traj) < Length):
    next_state = optimal_direction(state,policy[state], Height, Width)
    opt_traj.append(next_state)
    state = next_state
    #print len(opt_traj)

  return opt_traj

def get_trajectory_egreedy(policy, Height, Width, Length):
  e_traj=[]
  state = 0
  next_state = 0
  e_traj.append(0)

  while(state != (Height * Width -1) and len(e_traj) < Length and next_state < 49):
    next_state = e_greedy_direction(state, policy[state], Height, Width, 0.1)
    e_traj.append(next_state)
    state = next_state

  return e_traj

def match_rate(method, o_traj, e_traj):
  if(method == 'simple'):
    match_state = (set(o_traj) & set(e_traj))
    m_rate = float(len(match_state)) / float(len(o_traj))
    #print "match_state", match_state, len(match_state), len(o_traj)
    #print "m_rate", m_rate

  WINDOWSIZE = 3

  if(method == 'step'):
    match_count = 0

    m_rate = float(match_count)/ float(len(o_traj))
    #print "m_rate", m_rate,
    #print "match_count", match_count


  return m_rate

def maxent_irl(gw, feat_map, P_a, gamma, trajs, lr, n_iters):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                       landing at state s1 when taking action
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewardsF
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)


  rmap_gt = np.zeros([H, W])

  irl_gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)

  MRATE_THRESHOLD = 0.7

  exp_count = 0

  # init parameters
  theta = np.random.uniform(size=(feat_map.shape[1],))

  # calc feature expectations
  feat_exp = np.zeros([feat_map.shape[1]])

  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[step.cur_state,:]
  feat_exp = feat_exp/len(trajs)

  exp_length = 17
  check_opt_traj =[]

  update_time = []

  data_stepsize = []

  #exp_traj = [0, 7, 14, 15, 22, 29, 28, 35, 42, 43, 44, 45, 46, 47, 48]
  exp_traj = [0, 7, 8, 9, 16, 23 , 22, 21 , 28, 35, 42, 43, 44, 45, 46, 47, 48]

  # training
  for iteration in range(n_iters):

    if iteration % (n_iters/20) == 0:
      print 'iteration: {}/{}'.format(iteration, n_iters)
    print 'iteration: {}/{}'.format(iteration, n_iters)
    # compute reward function
    rewards = np.dot(feat_map, theta)

    # compute policy
    value, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=False)

    true_value, true_policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)

    #value3, policy3 = value_iteration.value_iteration(P_a, rewards, gamma, error=0.3, deterministic=True)

    #print "true_policy", true_policy
    #print "policy3", policy3

    # compute new trajectory
    new_trajs = generate_newtrajs(gw, true_policy, n_trajs=100, len_traj=20, rand_start=False)

    opt_traj = get_optimaltrajectory(true_policy,7,7,20)

    e_traj = get_trajectory_egreedy(true_policy, 7, 7, 20)

    #print opt_traj
    #print len(opt_traj)
    print e_traj
    print len(e_traj)

    '''
    if((exp_length >= len(opt_traj)-1) and (opt_traj != check_opt_traj)):
      trajs = new_trajs
      exp_length = len(opt_traj)
      exp_count += 1
      for episode in trajs:
        for step in episode:
          feat_exp += feat_map[step.cur_state,:]
      feat_exp = feat_exp/len(trajs)
      check_opt_traj=opt_traj
    '''

    print "exp_count", exp_count


    #compare optimal trajectory
    '''
    m_rate = match_rate('simple',opt_traj,exp_traj)

    if((exp_length >= len(opt_traj)-1) and (m_rate >= MRATE_THRESHOLD) and (opt_traj != check_opt_traj)):
      trajs = new_trajs
      exp_length = len(opt_traj)
      exp_count += 1
      for episode in trajs:
        for step in episode:
          feat_exp += feat_map[step.cur_state,:]
      feat_exp = feat_exp/len(trajs)
      check_opt_traj=opt_traj
      update_time.append(iteration)
    '''



    #compare epsilon-greedy trajectory

    m_rate = match_rate('simple',e_traj,exp_traj)

    if((exp_length >= len(e_traj)) and (m_rate >= MRATE_THRESHOLD) and (e_traj != check_opt_traj)):
      trajs = new_trajs
      exp_length = len(e_traj)
      exp_count += 1
      for episode in trajs:
        for step in episode:
          feat_exp += feat_map[step.cur_state,:]
      feat_exp = feat_exp/len(trajs)
      check_opt_traj=e_traj
      update_time.append(iteration)


    print "update_time", update_time
    #print data_stepsize
    data_stepsize.append(len(check_opt_traj))


    with open('results/step_size.csv', 'a')  as f:
      f.write(str(iteration))
      f.write(",")
      f.write(str(data_stepsize[iteration]))
      f.write('\n')
      f.close


    # compute state visition frequences
    svf = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False)

    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)

    #if (iteration >= 50):
      #feat_exp = mod_feat_exp(feat_map, bdtj.bad_trajs())
      #grad = feat_exp - feat_map.T.dot(svf)

    # update params
    theta += lr * grad

    rewards = np.dot(feat_map, theta)#policy

    _, check_policy = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True) #policy

    df = pd.DataFrame(check_policy)

    df.T.to_csv('results/policy77.csv',mode='a',index=False, header=False)



  rewards = np.dot(feat_map, theta)

  # return sigmoid(normalize(rewards))
  return normalize(rewards)