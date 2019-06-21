import numpy as numpy
from collections import namedtuple


Step = namedtuple('Step','cur_state action next_state reward done')

def bad_trajs():
      trajs=[[Step(cur_state=0, action=0, next_state=7, reward=0.0, done=False), Step(cur_state=7, action=0, next_state=14, reward=0.0, done=False), Step(cur_state=14, action=2, next_state=15, reward=0.0, done=False), Step(cur_state=15, action=2, next_state=16, reward=0.0, done=False), Step(cur_state=16, action=2, next_state=17, reward=0.0, done=False), Step(cur_state=17, action=0, next_state=24, reward=0.0, done=False), Step(cur_state=24, action=0, next_state=31, reward=0.0, done=False), Step(cur_state=31, action=2, next_state=32, reward=0.0, done=False), Step(cur_state=32, action=2, next_state=33, reward=0.0, done=False), Step(cur_state=33, action=0, next_state=40, reward=0.0, done=False), Step(cur_state=40, action=0, next_state=47, reward=0.0, done=False), Step(cur_state=47, action=2, next_state=48, reward=0.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False), Step(cur_state=48, action=4, next_state=48, reward=1.0, done=False)]]

      return trajs