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
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm


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
  for t in range(T-1):
    for s in range(N_STATES):
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
    #print "random"
    if(state == 0):
      randlist = [0,2]
      policy = np.random.choice(randlist)
    elif(state == Height):
      randlist = [0,3]
      policy = np.random.choice(randlist)
    elif(state == 42):
      randlist = [1,2]
      policy = np.random.choice(randlist)
    elif(state == 48):
      randlist = [1,3]
      policy = np.random.choice(randlist)
    elif(0 < state < 6):
      randlist = [0,2,3]
      policy = np.random.choice(randlist)
    elif((state % 7) == 0):
      randlist = [0,1,2]
      policy = np.random.choice(randlist)
    elif((state == 13) or (state == 20) or (state == 27) or (state == 34) or (state == 41)):
      randlist = [0,1,3]
      policy = np.random.choice(randlist)
    elif(42 < state < 48):
      randlist = [1, 2, 3]
      policy = np.random.choice(randlist)
    else:
      randlist = [0, 1, 2, 3]
      policy = np.random.choice(randlist)
  else:
    #print "not_random"
    policy = policy

    #print policy

  if(int(policy) == 0):
    if(state==42):
      randlist = [0, 2]
      policy = np.random.choice(randlist)
    elif(Height*Width - Height <= state < Height*Width):
      randlist = [1, 2, 3]
      policy = np.random.choice(randlist)
    else:
      policy = policy
  elif(int(policy) == 1):
    if(state==0):
      randlist = [0, 2]
      policy = np.random.choice(randlist)
    elif(state==6):
      randlist = [0, 3]
      policy = np.random.choice(randlist)
    elif(state < Height):
      randlist = [0, 2, 3]
      policy = np.random.choice(randlist)
    else:
      policy = policy
  elif(int(policy) == 2):
    if(state==6):
      randlist = [0, 3]
      policy = np.random.choice(randlist)
    elif((state==6) or (state==13) or (state==20) or (state==27) or (state==34) or (state==41)):
      randlist = [0, 1, 3]
      policy = np.random.choice(randlist)
    else:
      policy = policy
  elif(int(policy) == 3):
    if(state==0):
      randlist=[0, 2]
      policy = np.random.choice(randlist)
    elif(state==42):
      randlist= [1,2]
      policy = np.random.choice(randlist)
    elif(state % Height == 0):
      randlist = [0, 1, 2]
      policy = np.random.choice(randlist)
    else:
      policy = policy
  else:
    policy = policy

  if(int(policy) == 0):
    if(Height*Width - Height <= state < Height*Width):
      state = state
    else:
      state += Height
  elif(int(policy) == 1):
    if(state < Height):
      state = state
    else:
      state -= Height
  elif(int(policy) == 2):
    if((state==6) or (state==13) or (state==20) or (state==27) or (state==34) or (state==41)):
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

  return opt_traj

def get_trajectory_egreedy(policy, Height, Width, Length):
  e_traj=[]
  state = 0
  next_state = 0
  e_traj.append(0)

  while(state != (Height * Width -1) and len(e_traj) < Length and next_state < 49):
    next_state = e_greedy_direction(state, policy[state], Height, Width, 0.3)
    e_traj.append(next_state)
    state = next_state
  return e_traj

def match_rate(method, o_traj, e_traj):
  if(method == 'simple'):
    match_state = set(set(o_traj) & set(e_traj))
    m_rate = float(len(match_state)) / float(len(o_traj))

  if(method == 'step'):
    match_count = 0
    m_rate = float(match_count)/ float(len(o_traj))

  return m_rate

def tune_rate(iteration, n_iters, m_rate, update_time):
  progress = iteration / n_iters
  rate = m_rate

  if((progress > 0.3) and (len(update_time) < 2)):
    rate = m_rate - 0.1
  if((progress > 0.5) and (len(update_time) < 2)):
    rate = m_rate - 0.3

  return rate

def make_traj(i_length, candidate):
  episode = []
  trajs=[]
  for i in range(i_length):
    if(len(candidate) > i):
      episode.append(Step(cur_state=candidate[i], action=0, next_state=candidate[i], reward=0, done=False))
    else:
      episode.append(Step(cur_state=48, action=0, next_state=48, reward=0, done=False))
  trajs.append(episode)

  return trajs

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    a = x-min
    result = a.astype('float')/(max-min)
    return result



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
  MRATE_THRESHOLD = 0.9
  exp_count = 0

  SLIDESIZE = 100


  # state number
  state_num = np.array([i+1 for i in range(H*W)])
  statecount = np.zeros(H*W,dtype=float)

  # init parameters
  theta = np.random.uniform(size=(feat_map.shape[1],))

  # calc feature expectations
  feat_exp = np.zeros([feat_map.shape[1]])

  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[step.cur_state,:]
  feat_exp = feat_exp/len(trajs)

  check_opt_traj =[]

  update_time = []

  data_stepsize = []

  #case2
  
  exp_traj = [0,1,2,3,4,5,6,13,20,19,18,17,24,31,38,45,46,47,48]
  e_traj = [0,1,2,3,4,5,6,13,20,19,18,17,24,31,38,45,46,47,48]
  

  #case1
  """
  exp_traj = [0,7,14,15,22,29,28,35,42,43,44,45,46,47,48]
  e_traj = [0,7,14,15,22,29,28,35,42,43,44,45,46,47,48]
  """

  #case3
  """
  exp_traj = [0,1,8,15,22,31,32,33,40,41,48]
  e_traj = [0,1,8,15,22,31,32,33,40,41,48]
  """
  #case3 new
  """
  exp_traj = [0,1,8,15,22,23, 24, 31,32,33,40,41,48]
  e_traj = [0,1,8,15,22,23,24,31,32,33,40,41,48]
  """

  #case4
  """
  exp_traj = [0,1,8,15,14,21,28,29,30,31,32,33,40,41,48]
  e_traj =[0,1,8,15,14,21,28,29,30,31,32,33,40,41,48]
  """

  #exp3
  """
  exp_traj = [0,1,2,3,4,5,6,13,20,27,26,25,24,31,38,45,46,47,48]
  e_traj = [0,1,2,3,4,5,6,13,20,27,26,25,24,31,38,45,46,47,48]
  """

  #exp3 case1
  """
  exp_traj = [0,1,2,3,4,5,6,13,12,11,10,9,16,23,30,37,44,45,46,47,48]
  e_traj = [0,1,2,3,4,5,6,13,12,11,10,9,16,23,30,37,44,45,46,47,48]
  """

  #exp3 case2
  """
  exp_traj=[0,1,8,15,14,21,28,29,30,31,24,17, 10, 11, 12, 19,26,33,40,47,48]
  e_traj=[0,1,8,15,14,21,28,29,30,31,24,17, 10, 11, 12, 19,26,33,40,47,48]
  """

  #exp3 case2 new
  """
  exp_traj=[0,1,8,15,14,21,28,29,30,31,24,17, 10, 11,12, 13, 20 ,27,34,41,48]
  e_traj=[0,1,8,15,14,21,28,29,30,31,24,17,10,11,12,13,20,27,34,41,48]
  """

  select_candidate = []

  maxstate = 0
  overstate =[]

  # training
  for iteration in tqdm(range(n_iters)):

    #print 'iteration: {}/{}'.format(iteration, n_iters)
    # compute reward function
    if(iteration == 0):
      rewards = np.dot(feat_map, theta)

    # compute policy
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=False)
    _, true_policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)

    """
    if (iteration % 150 == 0):
      plt.figure(figsize=(20,20))
      img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map', block=False)
      plt.plot()
      now = datetime.datetime.now()
      figname = "results/reward/rewards_{0:%m%d%H%M%S}".format(now) + ".png"
      plt.savefig(figname)

      plt.figure(figsize=(20,20))
      img_utils.heatmap2d(np.reshape(value, (H, W), order='F'), 'Value Map', block=False)
      plt.plot()
      figname1 = "results/value/value_{0:%m%d%H%M%S}".format(now) + ".png"
      plt.savefig(figname1)
      #plt.show()
    """

    # compute new trajectory

    #new_trajs = generate_newtrajs(gw, true_policy, n_trajs=100, len_traj=30, rand_start=False)
    #opt_traj = get_optimaltrajectory(true_policy,7,7,20)

    #if terminal == 48
    candidate = get_trajectory_egreedy(true_policy, 7, 7, 20)

    re_candidate = sorted(set(candidate), key=candidate.index)

    e_traj = copy.deepcopy(re_candidate)

    if exp_traj[-2] in re_candidate:
      select_candidate = copy.deepcopy(re_candidate)
    print " "
    print "candidate   ", candidate
    print "re_candidate", re_candidate
    print "exp_traj    ", exp_traj

    #compare epsilon-greedy trajectory
    m_rate = match_rate('simple',e_traj,exp_traj)

    print m_rate

    #m_threshold = tune_rate(iteration, n_iters, MRATE_THRESHOLD, update_time)
    #print ("m_threshold", m_threshold)

    if((len(exp_traj) > len(e_traj)) and ((check_opt_traj != e_traj)) and (m_rate >= MRATE_THRESHOLD) and ((48 in e_traj))):
      trajs = make_traj(20,  e_traj)
      exp_count += 1
      feat_exp = np.zeros([feat_map.shape[1]])
      for episode in trajs:
        for step in episode:
          feat_exp += feat_map[step.cur_state,:]
      feat_exp = feat_exp/len(trajs)
      check_opt_traj=e_traj
      exp_traj = e_traj
      update_time.append(iteration)


    """
    if(iteration == 100):
      trajs = tj.exp1_case3_correct()
      exp_count += 1
      feat_exp = np.zeros([feat_map.shape[1]])
      for episode in trajs:
        for step in episode:
          feat_exp += feat_map[step.cur_state,:]
      feat_exp = feat_exp/len(trajs)
      check_opt_traj= [0,1,8,15,16,23,24,31,32,33,40,41,48]
      exp_traj =  [0,1,8,15,16,23,24,31,32,33,40,41,48]
      update_time.append(iteration)
    """


    #print "exp_traj    ", exp_traj
    print "update_time", update_time

    data_stepsize.append(len(check_opt_traj))

    '''
    with open('results/step_size.csv', 'a')  as f:
      f.write(str(iteration))
      f.write(",")
      f.write(str(data_stepsize[iteration]))
      f.write('\n')
      f.close
    '''
    '''
    with open('results/expert.csv', 'a')  as f:
      f.write(str(iteration))
      f.write(",")
      for state in exp_traj:
        f.write(str(state))
        f.write(",")
      f.write('\n')
      f.close
    '''
    '''
    with open('results/candidate.csv', 'a')  as f:
      f.write(str(iteration))
      f.write(",")
      for state in candidate:
        f.write(str(state))
        f.write(",")
      f.write('\n')
      f.close
    '''
    '''
    with open('results/re_candidate.csv', 'a')  as f:
      f.write(str(iteration))
      f.write(",")
      for state in re_candidate:
        f.write(str(state))
        f.write(",")
      f.write('\n')
      f.close
    '''
    '''
    with open('results/select_candidate.csv', 'a')  as f:
      f.write(str(iteration))
      f.write(",")
      for state in select_candidate:
        f.write(str(state))
        f.write(",")
      f.write('\n')
      f.close
    '''

    # compute state visition frequences
    svf = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False)

    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)

    # update params
    theta += lr * grad

    rewards = np.dot(feat_map, theta)#policy

    rewards = normalize(rewards)

    for t in range(len(candidate)):
      statecount[candidate[t]] +=1

    
    if(iteration % SLIDESIZE ==0 and iteration != 0):
      fig = plt.figure()
      left = state_num
      height = statecount
      # minmax_h = min_max(height)
      plt.bar(left,height,color="#FF5B70")
      plt.title("iteration{0}".format(iteration))
      plt.savefig('results/statecount{0}'.format(iteration) + '.png')
      print height

      overstate = theta
      for i in range(len(height)):
        if(height[i] > SLIDESIZE + 50):
          overstate[i] = 0.0
          print "overstate{0}".format(i)
      statecount = np.zeros(H*W,dtype=int)

    if(iteration > SLIDESIZE):
      theta = overstate
    



  rewards = np.dot(feat_map, theta)
  #return rewards
  return normalize(rewards)