#PET_env.py
#contains class for the enviroment
#cd '/anaconda3/envs/TestPet/TestPet'
#import gym
#import gym_PET
#env = gym.make('PET-v0')
#env.render()
import gym
import sys
sys.path.append('/home/putz/anaconda3/envs/TestPet/TestPet/gym_PET/envs')
import PET_read as PT
from gym import error, spaces, utils
from gym.utils import seeding

class PETEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.__version__ = "0.1.0"
        self.test=1
        self.action_space = spaces.Discrete(2)
        print("PET_env - Version {}".format(self.__version__))

# General variables defining the environment
   
    def _step(self, action):  
        self.STEP="basic"
    
    def _reset(self):
        self.RESET="none"
       
    def _render(self, mode='human', close=False):
        self.RENDER="how"
        print('how')
        PET_img_data=PT.get_data()
        slice_0 = PET_img_data[125, :, :,0]
        slice_1 = PET_img_data[:, 160, :,0]
        slice_2 = PET_img_data[:, :, 100,0]
        PT.show_slices([slice_0, slice_1, slice_2])
        PT.plt.show()
    






