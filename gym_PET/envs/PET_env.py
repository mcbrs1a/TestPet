#PET_env.py
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class PETEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}

def __init__(self):
	self.__version__ = "0.1.0"
	print("PET_env - Version {}".format(self.__version__))

# General variables defining the environment

   
def _step(self, action):
    
def _reset(self):
    
def _render(self, mode='human', close=False):
    
