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


R_DISCOUNT = 0.1
R_RAISE = 0.1
GAMMA = 0.8
H = 10
W = 10

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

def action_moveX(action, x):

    if action == 0:
        x = 1
        #print 'r'
        return x
    elif action == 1:
        x = -1
        #print 'l'
        return x
    elif action == 2:
        #print 'd'
        return 0
    elif action == 3:
        #print 'u'
        return 0
    else :
        #print 's'
        return 0

def action_moveY(action, y):

    if action == 0:
        #print 'r'
        return 0
    elif action == 1:
        #print 'l'
        return 0
    elif action == 2:
        y = 1
        #print 'd'
        return y
    elif action == 3:
        y = -1
        #print 'u'
        return y
    else :
        #print 's'
        return 0


def Find_badstate(start_x, start_y, goal_x, goal_y, policy, height, width, badstate):
    
    
    current_x = start_x
    current_y = start_y
    
    policy2 = np.zeros((width,height))

    for i in range(width):
        for j in range(height):
            policy2[i,j] = np.argmax(policy[j+height*i])
            
            
    #print policy2
        
                   
    iteration = 0
    next_x = 0
    next_y = 0
    
    current_state = 0
    count = 0
    
    while((next_x != goal_x and next_y != goal_y) and iteration <= 20):
        iteration += 1
        
        #print 'policy_iteration', iteration
        #print 'current_x', current_x
        #print 'current_y', current_y
        #print policy2[current_x, current_y]
        past_x = current_x
        past_y = current_y
        
        #print 'current_state', current_state
        #print 'badstate' , badstate[0]
        
        if current_state == int(badstate[0]):
            print 'count'
            count += 1
        
        next_x = past_x + action_moveX(policy2[past_x, past_y], past_x)
        next_y = past_y + action_moveY(policy2[past_x, past_y], past_y)
        
        #print 'next_x', next_x
        #print 'next_y', next_y
        
        current_x = next_x
        current_y = next_y
        
        current_state = width * current_x + current_y
        
    return count

def max_dir(policy):
    for pre_s in range(10*10):
        if (np.argmax(policy[pre_s])==0):
            print pre_s, 'r'
        elif (np.argmax(policy[pre_s]==1)):
            print pre_s, 'l'
        elif (np.argmax(policy[pre_s]==2)):
            print pre_s, 'd'
        elif (np.argmax(policy[pre_s]==3)):
            print pre_s, 'u'
        else:
            print pre_s, 's'


def change_dir(policy):
    for pre_s in range(100):
        if(policy[pre_s]==0):
            print pre_s,'r'
        elif(policy[pre_s]==1):
            print pre_s,'l'
        elif(policy[pre_s]==2):
            print pre_s,'d'
        elif(policy[pre_s]==3):
            print pre_s,'u'
        else:
            print pre_s,'s'

def maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
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
    rewards     Nx1 vector - recoverred state rewards
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)



  # init parameters
  theta = np.random.uniform(size=(feat_map.shape[1],))

  # calc feature expectations
  feat_exp = np.zeros([feat_map.shape[1]])
  for episode in trajs:
    for step in episode:
      feat_exp += feat_map[step.cur_state,:]
  feat_exp = feat_exp/len(trajs)

  # training
  for iteration in range(n_iters):

    if iteration % (n_iters/20) == 0:
      print 'iteration: {}/{}'.format(iteration, n_iters)
    print 'iteration: {}/{}'.format(iteration, n_iters)
    # compute reward function
    rewards = np.dot(feat_map, theta)

    # compute policy
    value, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=False)

    #max_dir(policy)

    # compute state visition frequences
    svf = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False)

    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)

    #print feat_exp


    # update params
    theta += lr * grad

    rewards = np.dot(feat_map, theta)#policy

    _, check_policy = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True) #policy

    #print check_policy

    df = pd.DataFrame(check_policy)

    df.T.to_csv('results/policy77.csv',mode='a',index=False, header=False)



  rewards = np.dot(feat_map, theta)

  # return sigmoid(normalize(rewards))
  return normalize(rewards)

def convert_one2two(rewards, height, width):
  new_rewards_xy = np.zeros((width,height))
  #new_rewards_xy[1,4] = 3.2

  for i in range(width):
    for j in range(height):
      new_rewards_xy[i,j] = rewards[j+height*i]

  return new_rewards_xy


def reward_decrease(rewards, R_GAMMA, BAD_X, BAD_Y, width, height):
  new_rewards = copy.copy(rewards)

  badpos = np.array([BAD_X,BAD_Y])


  for i in range(width):
    for j in range(height):
      pos = np.array([i,j])
      new_rewards[i,j] = new_rewards[i,j]*(1-pow(R_GAMMA,np.linalg.norm(badpos-pos,ord=0)+1))

  return new_rewards

def convert_two2one(rewards2,width,height):
  rewards1 = np.zeros(width*height)

  for i in range(width):
    for j in range(height):
      rewards1[j+height*i] = rewards2[i,j]

  return rewards1

