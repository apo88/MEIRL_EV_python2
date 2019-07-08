import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple

import img_utils
from mdp import gridworld
from mdp import value_iteration
from maxent_irl import *
import mod_trajectory as mod
import datetime

Step = namedtuple('Step','cur_state action next_state reward done')

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=7, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=7, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.8, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.1, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=100, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=False)
PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=1000, type=int, help='number of iterations')
PARSER.add_argument('-rg', '--r_gamma', default=0.3, type=float, help='discount factor for rewards')
PARSER.add_argument('-bx', '--bad_x', default= 0, type=int, help='bad state of x orign')
PARSER.add_argument('-by', '--bad_y', default= 4, type=int, help='bad state of y orign')
PARSER.add_argument('-bs', '--bad_state', default=50, type=int, help='bad state of origin')
ARGS = PARSER.parse_args()

GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters
R_GAMMA = ARGS.r_gamma
BAD_X = ARGS.bad_x
BAD_Y = ARGS.bad_y
BAD_STATE = ARGS.bad_state

Bad_states = list()


def feature_coord(gw):
  N = gw.height * gw.width
  feat = np.zeros([N, 2])
  for i in range(N):
    iy, ix = gw.idx2pos(i)
    feat[i,0] = iy
    feat[i,1] = ix
  return feat

def feature_basis(gw):
  """
  Generates a NxN feature map for gridworld
  input:
    gw      Gridworld
  returns
    feat    NxN feature map - feat[i, j] is the l1 distance between state i and state j
  """
  N = gw.height * gw.width
  feat = np.zeros([N, N])
  for i in range(N):
    for y in range(gw.height):
      for x in range(gw.width):
        iy, ix = gw.idx2pos(i)
        feat[i, gw.pos2idx([y, x])] = abs(iy-y) + abs(ix-x)
  return feat


def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
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



def main():
  N_STATES = H * W
  N_ACTIONS = 5

  """while True:
      print "BAD_STATE入力"
      bad = raw_input('>> ')
      if bad == 'ok':
          break
      Bad_states.append(bad)
  """

  #print Bad_states
  # init the gridworld
  # rmap_gt is the ground truth for rewards
  rmap_gt = np.zeros([H, W])

  #print R_MAX
  #goal coordinates
  rmap_gt[H-1, W-1] = R_MAX
  # rmap_gt[H-1, 0] = R_MAX

  gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)

  rewards_gt = np.reshape(rmap_gt, H*W, order='F')
  P_a = gw.get_transition_mat()

  values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)

  #print policy_gt
  # use identity matrix as feature
  feat_map = np.eye(N_STATES)

  # other two features. due to the linear nature,
  # the following two features might not work as well as the identity.
  # feat_map = feature_basis(gw)
  # feat_map = feature_coord(gw)
  np.random.seed(0)

  #print type(policy_gt)

  trajs = generate_demonstrations(gw, policy_gt, n_trajs=N_TRAJS, len_traj=L_TRAJ, rand_start=RAND_START)

  trajs = mod.init_trajs()


  rewards = maxent_irl(gw, feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)

  np.savetxt('results/rewards.txt', rewards)

  values, policy = value_iteration.value_iteration(P_a, rewards, GAMMA, error=0.01, deterministic=True)


  # plots
  plt.figure(figsize=(20,20))
  img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map', block=False)
  plt.plot()
  now = datetime.datetime.now()
  figname = "results/rewards_{0:%m%d%H%M}".format(now) + ".png"
  print(figname)
  plt.savefig(figname)
  plt.show()
  #plt.figure(figsize=(20,20))
  #img_utils.heatmap2d(np.reshape(values, (H,W), order='F'), 'Policy Map', block=False)
  #plt.plot()
  #plt.show()


  # plt.subplot(2, 2, 4)
  # img_utils.heatmap3d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Recovered', block=False)


if __name__ == "__main__":
  main()