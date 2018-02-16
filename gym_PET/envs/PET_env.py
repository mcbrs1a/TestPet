# also do this  conda install -c https://conda.anaconda.org/kne pybox2d
#pip install --ignore-installed --upgrade tensorflow
#PET_env.py
#contains class for the enviroment
#cd '/anaconda3/envs/TestPet/TestPet'
#import gym
#import gym_PET
#env = gym.make('PET-v0')
#env.render()



#for i in range(10):
#    env.render()
#     action = env.action_space.sample()
#     env.step(action)
#    env.reset()
   
import Box2D
import gym
import time
import sys
sys.path.append('/home/petic/anaconda/envs/A3Cexample/TestPet/gym_PET/envs')
import PET_read as PT
import numpy as np
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding

class PETEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.__version__ = "0.1.0"
        self.test=1
        self.statew=30
        self.stateh=30
        self.stated=30
        self.img=PT.get_data()
        im=self.img
        min=self.img.min()
        max=self.img.max()
        self.sz=PT.get_siz()
        self.ref=PT.get_ref_for_dice()
        sa=self.sz[0]
        sb=self.sz[1]
        print('max image is')
        print(self.img.max())
        self.la=None 
        self.lb=None
        self.lc=None
        self.ld=None
        self.le=None
        self.lf=None
        self.lg=None
        self.lh=None
        self.li=None
        self.lj=None
        self.lk=None
        self.ll=None


        fig, axes = plt.subplots(1, len(self.sz))
        self.fig=fig
        self.axes=axes
        if (len(self.sz)==3):
            sc=self.sz[2]
            self.sa=round(self.sz[0]/2)
            self.sb=round(self.sz[1]/2)
            self.sc=round(self.sz[2]/2)
            self.statea=[100,100,100,130,130,130,30,30,30]
            self.state=self.img[self.statea[0]:self.statea[3],self.statea[1]:self.statea[4],self.statea[2]:self.statea[5]]
            self.stateax=self.statea[0]
            self.stateay=self.statea[1]
            self.stateaz=self.statea[2]
            self.statebx=self.statea[3]
            self.stateby=self.statea[4]
            self.statebz=self.statea[5]

            #self.observation_space=spaces.Box(min,max,[sa,sb,sc])
            self.observation_space=im
            self.action_space=spaces.Box(np.array([-10,-10,-10]), np.array([+10,+10,+10])) #Just translation atm tx,ty,tz
           
        else:
            self.sa=round(self.sz[0]/2)
            self.sb=round(self.sz[1]/2)
            self.observation_space=spaces.Box(min,max,[sa,sb])
            self.statea=[0,0,0,0,5,5]
            self.state=self.img[self.statea[0]:self.statea[3],self.statea[1]:self.statea[4]]
            self.stateax=self.statea[0]
            self.stateay=self.statea[1]
            self.statebx=self.statea[2]
            self.stateby=self.statea[3]
            
            self.action_space=spaces.Box(np.array([-1,-1]), np.array([+1,+1]))
          

        print('dimensions are')
        print(self.sz)
        print("PET_env - Version {}".format(self.__version__))

    def seed(self, seed=None):
        #returns random state or observation object
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        self.seed()


# General variables defining the environment
   
    def _step(self, action):  

        if action is not None:
            self.la.remove()
            self.lb.remove()
            self.lc.remove() 
            self.ld.remove()
            self.le.remove()
            self.lf.remove()
            self.lg.remove() 
            self.lh.remove()
            self.li.remove() 
            self.lj.remove()
            self.lk.remove()
            self.ll.remove()

            self.statea[0]=self.statea[0]+int(action[0])
            self.statea[1]=self.statea[1]+int(action[1])
            self.statea[2]=self.statea[2]+int(action[2])
            self.statea[3]=self.statea[3]+int(action[0])
            self.statea[4]=self.statea[4]+int(action[1])
            self.statea[5]=self.statea[5]+int(action[2])



            self.stateax=self.statea[0]
            self.stateay=self.statea[1]
            self.stateaz=self.statea[2]
            self.statebx=self.statea[3]
            self.stateby=self.statea[4]
            self.statebz=self.statea[5]


            if self.stateax+self.statew>(self.sz[0]-1):
               print('dimx initialized two big reducing')
               print(self.stateax)
               self.stateax=(self.sz[0]-self.statew)-1
               print(self.stateax)

            if self.stateay+self.stated>(self.sz[1]-1):
               print('dimy initialized two big reducing')
               print(self.stateay)
               self.stateay=(self.sz[1]-self.stated)-1
               print(self.stateay)

            if self.stateaz+self.stateh>(self.sz[2]-1):
               print('size of image in z is')
               print(self.sz[2])
               print('dimz initialized two big reducing')
               print('inital z is')
               print(self.stateaz)
               print('height is')
               print(self.stateh)
               self.stateaz=(self.sz[2]-self.stateh)-1
               print('reduced z is')
               print(self.stateaz)

            self.statebx=self.stateax+self.statew
            self.stateby=self.stateay+self.stateh
            self.statebz=self.stateaz+self.stated

            self.statea[0]=self.stateax
            self.statea[1]=self.stateay
            self.statea[2]=self.stateaz
            self.statea[3]=self.statebx
            self.statea[4]=self.stateby
            self.statea[5]=self.statebz

            self.sa=round((self.stateax+self.statebx)/2)
            self.sb=round((self.stateay+self.stateby)/2)
            self.sc=round((self.stateaz+self.statebz)/2)


            self.state=self.img[self.statea[0]:self.statea[3],self.statea[1]:self.statea[4],self.statea[2]:self.statea[5]]
            reward = PT.get_structsim(self.ref,self.state)
            print('struct sim reward is', reward)

            done = bool(reward==1)


        return np.array(self.state), float(reward), done, self.statea, {}
            



    def _reset(self):
    #must reurn the "state" or the observable
    #in this case its the [] top left bottom right
        self.RESET="none"
        print('dimensions are')
        print(self.sz)
        self.la.remove()
        self.lb.remove()
        self.lc.remove() 
        self.ld.remove()
        self.le.remove()
        self.lf.remove()
        self.lg.remove() 
        self.lh.remove()
        self.li.remove() 
        self.lj.remove()
        self.lk.remove()
        self.ll.remove()
 
        if (len(self.sz)==3): #width height depth max size is an eight of image
            self.statew=round(np.random.uniform(low=0, high=self.sz[0]/8))
            self.stateh=round(np.random.uniform(low=0, high=self.sz[1]/8))
            self.stated=round(np.random.uniform(low=0, high=self.sz[2]/8))
            self.stateax=round(np.random.uniform(low=0, high=self.sz[0])-1)
            self.stateay=round(np.random.uniform(low=0, high=self.sz[1])-1)
            self.stateaz=round(np.random.uniform(low=0, high=self.sz[2])-1)
          
         
           

            if self.stateax+self.statew>(self.sz[0]-1):
               print('dimx initialized two big reducing')
               print(self.stateax)
               self.stateax=(self.sz[0]-self.statew)-1
               print(self.stateax)

            if self.stateay+self.stated>(self.sz[1]-1):
               print('dimy initialized two big reducing')
               print(self.stateay)
               self.stateay=(self.sz[1]-self.stated)-1
               print(self.stateay)

            if self.stateaz+self.stateh>(self.sz[2]-1):
               print('size of image in z is')
               print(self.sz[2])
               print('dimz initialized two big reducing')
               print('inital z is')
               print(self.stateaz)
               print('height is')
               print(self.stateh)
               self.stateaz=(self.sz[2]-self.stateh)-1
               print('reduced z is')
               print(self.stateaz)

            self.statebx=self.stateax+self.statew
            self.stateby=self.stateay+self.stateh
            self.statebz=self.stateaz+self.stated

            #self.action_space=spaces.Discrete(3) #Just translation atm tx,ty,tz
         
            self.sa=round((self.stateax+self.statebx)/2)
            self.sb=round((self.stateay+self.stateby)/2)
            self.sc=round((self.stateaz+self.statebz)/2)

            #adjusted here
            self.statea=[100,100,100,130,130,130,30,30,30]
            self.stateax=self.statea[0]
            self.stateay=self.statea[1]
            self.stateaz=self.statea[2]
            self.statebx=self.statea[3]
            self.stateby=self.statea[4]
            self.statebz=self.statea[5]
            self.statew=self.statea[6]
            self.stateh=self.statea[7]
            self.stated=self.statea[8]
           
            self.sa=round((self.stateax+self.statebx)/2)
            self.sb=round((self.stateay+self.stateby)/2)
            self.sc=round((self.stateaz+self.statebz)/2)

            self.statea=[self.stateax, self.stateay, self.stateaz, self.statebx, self.stateby, self.statebz, self.statew, self.stateh, self.stated]

            self.sa=round(self.sz[0]/2)
            self.sb=round(self.sz[1]/2)
            self.sc=round(self.sz[2]/2)
            self.statea=[100,100,100,130,130,130,30,30,30]
            self.state=self.img[self.statea[0]:self.statea[3],self.statea[1]:self.statea[4],self.statea[2]:self.statea[5]]
            self.stateax=self.statea[0]
            self.stateay=self.statea[1]
            self.stateaz=self.statea[2]
            self.statebx=self.statea[3]
            self.stateby=self.statea[4]
            self.statebz=self.statea[5]


            self.state=self.img[self.stateax:self.statebx,self.stateay:self.stateby,self.stateaz:self.statebz]





            #self.axes.cla()
            return np.array(self.state)
        else:
            self.statew=round(np.random.uniform(low=0, high=self.sz[0]/8))
            self.stateh=round(np.random.uniform(low=0, high=self.sz[1]/8))
            self.stateax=round(np.random.uniform(low=0, high=self.sz[0])-1) 
            self.stateay=round(np.random.uniform(low=0, high=self.sz[1])-1)

            #self.action_space=spaces.Discrete(3) #Just translation atm tx,ty,tz

            if self.statea+self.statew>(self.sz[0]-1):
                print('dimx initialized two big reducing')
                self.stateax=(sz[0]-self.statew)-1    

            if self.stateb+self.statew>(self.sz[1]-1):
               print('dimy initialized two big reducing')
               self.stateay=(sz[1]-self.stated)-1

            self.statebx=self.stateax+self.statew
            self.stateby=self.stateay+self.stated

            self.statebx=self.stateax+self.statew
            self.stateby=self.stateay+self.stated

            self.sa=round((self.stateax+self.statebx)/2)
            self.sb=round((self.stateay+self.stateby)/2)

            self.statea=[self.stateax, self.stateay, self.statebx, self.stateby, self.statew, self.stateh]
            self.state=self.img[self.stateax:self.statebx,self.stateay:self.stateby]

            return np.array(self.state)

        #obs=self.reset()
       
    def _render(self, mode='human', close=False):
        self.RENDER="how"
        print('returning data')
        

        PT.plt.ion()
        PT.plt.show()
        PT.plt.draw()
        
        slice_0 = self.img[self.sa, :, :]
        slice_1 = self.img[:, self.sb, :]
        slice_2 = self.img[:, :, self.sc]
        ax=PT.show_slices([slice_0, slice_1, slice_2],self.fig,self.axes)
     
        obs=self.statea
        print('obs is')
        print(obs)

        la=ax[0].plot(np.linspace(obs[1],obs[4], num=abs(obs[4]-obs[1])),np.linspace(obs[2],obs[2], num=abs(obs[4]-obs[1])),"r-") #first dim static
        lb=ax[0].plot(np.linspace(obs[1],obs[4], num=abs(obs[4]-obs[1])),np.linspace(obs[5],obs[5], num=abs(obs[4]-obs[1])),"b-")
        lc=ax[0].plot(np.linspace(obs[1],obs[1], num=abs(obs[2]-obs[5])),np.linspace(obs[2],obs[5], num=abs(obs[5]-obs[2])),"y-")
        ld=ax[0].plot(np.linspace(obs[4],obs[4], num=abs(obs[2]-obs[5])),np.linspace(obs[2],obs[5], num=abs(obs[5]-obs[2])),"g-")


        le=ax[1].plot(np.linspace(obs[0],obs[3], num=abs(obs[0]-obs[3])),np.linspace(obs[2],obs[2], num=abs(obs[0]-obs[3])),"r-") # second dim static
        lf=ax[1].plot(np.linspace(obs[0],obs[3], num=abs(obs[0]-obs[3])),np.linspace(obs[5],obs[5], num=abs(obs[0]-obs[3])),"b-")
        lg=ax[1].plot(np.linspace(obs[0],obs[0], num=abs(obs[2]-obs[5])),np.linspace(obs[2],obs[5], num=abs(obs[5]-obs[2])),"y-")
        lh=ax[1].plot(np.linspace(obs[3],obs[3], num=abs(obs[2]-obs[5])),np.linspace(obs[2],obs[5], num=abs(obs[2]-obs[5])),"g-")


        li=ax[2].plot(np.linspace(obs[0],obs[3], num=abs(obs[0]-obs[3])),np.linspace(obs[1],obs[1], num=abs(obs[0]-obs[3])),"r-") # third dim static
        lj=ax[2].plot(np.linspace(obs[0],obs[3], num=abs(obs[0]-obs[3])),np.linspace(obs[4],obs[4], num=abs(obs[0]-obs[3])),"b-")
        lk=ax[2].plot(np.linspace(obs[0],obs[0], num=abs(obs[4]-obs[1])),np.linspace(obs[4],obs[1], num=abs(obs[4]-obs[1])),"y-")
        ll=ax[2].plot(np.linspace(obs[3],obs[3], num=abs(obs[4]-obs[1])),np.linspace(obs[4],obs[1], num=abs(obs[4]-obs[1])),"g-")

        PT.plt.show()
        self.la=la[0] 
        self.lb=lb[0] 
        self.lc=lc[0] 
        self.ld=ld[0]
        self.le=le[0]
        self.lf=lf[0] 
        self.lg=lg[0] 
        self.lh=lh[0]
        self.li=li[0] 
        self.lj=lj[0]
        self.lk=lk[0]
        self.ll=ll[0]
        self.ll=ll[0] 
        PT.plt.pause(0.5)


        #time.sleep(5)
        #PT.plt.gcf().clear()

        #ax[0].plot(np.linspace(100,100, num=100),np.linspace(1,100, num=100))
        return self.state


    






