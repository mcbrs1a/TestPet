#run episode
#parameters are used to choose action from observation
#state and observation same thing

import numpy as np
import gym
import random
import gym_PET
import tensorflow as tf
import numpy as np
env = gym.make('PET-v0')
import PET_read as PT
observations=env.render()
sh=observations.shape
ref=PT.get_ref_for_dice()
stru=PT.get_structsim(ref, observations)
parameters=np.random.rand(sh[0], sh[1], sh[2])*2-1

def policy_gradient():  
     params.get_shape().as_list()
    params = tf.get_variable("policy_parameters",[4,2])
    state = tf.placeholder("float",[None,4])
    linear = tf.matmul(state,params)
    probabilities = tf.nn.softmax(linear)


def switch_action(action, tit):
    print('switching action')
    if tit[0]==0 or tit[1]==0 or tit[2]==0:
        acr=np.flatnonzero(np.array(tit) == 0)
        for k in range(len(acr)):
            action[acr[k-1]]=-action[acr[k-1]]
    return action

tom=[0,0,0]
tom[0]=random.randint(-9,9)
tom[1]=random.randint(-9,9)
tom[2]=random.randint(-9,9)

def run_episode(env, parameters, tom):
    #observation=env.reset()
    observations=env.render()
    env.reset()
    observations=env.render()
    totalreward=np.array([])
    action=[0,0,0]
    state=[0,0,0,0,0,0]
    for _ in range(200):
        #'iterate in direction of maximum similarity'
        rewzm=PT.get_structsim(observations[:,:,int(observations.shape[2]/2)],ref[:,:,int(ref.shape[2]/2)])
        rewzl=PT.get_structsim(observations[:,:,int(observations.shape[2]/2)-5],ref[:,:,int(ref.shape[2]/2)])
        rewzr=PT.get_structsim(observations[:,:,int(observations.shape[2]/2)+5],ref[:,:,int(ref.shape[2]/2)])
        rewzlb=PT.get_structsim(observations[:,:,int(observations.shape[2]/2)-10],ref[:,:,int(ref.shape[2]/2)])
        rewzrb=PT.get_structsim(observations[:,:,int(observations.shape[2]/2)+10],ref[:,:,int(ref.shape[2]/2)])
        zl=[rewzl, rewzlb]
        itzl=zl.index(max(zl))
        zr=[rewzr, rewzrb]
        itzr=zr.index(max(zr))
        zv=[zl[itzl], rewzm, zr[itzr]] #+1 0 -1
        itz=zv.index(max(zv))
        print('z is done')
        rewym=PT.get_structsim(observations[:,int(observations.shape[2]/2),:],ref[:,int(ref.shape[2]/2),:])
        rewyl=PT.get_structsim(observations[:,int(observations.shape[2]/2)-5,:],ref[:,int(ref.shape[2]/2),:])
        rewyr=PT.get_structsim(observations[:,int(observations.shape[2]/2)+5,:],ref[:,int(ref.shape[2]/2),:])
        rewylb=PT.get_structsim(observations[:,int(observations.shape[2]/2)-10,:],ref[:,int(ref.shape[2]/2),:])
        rewyrb=PT.get_structsim(observations[:,int(observations.shape[2]/2)+10,:],ref[:,int(ref.shape[2]/2),:])
        yl=[rewyl, rewylb]
        ityl=yl.index(max(yl))
        yr=[rewyr, rewyrb]
        ityr=yr.index(max(yr))
        yv=[yl[ityl], rewym, yr[ityr]] #+1 0 -1
        ity=yv.index(max(yv))
        print('y is done')
        rewxm=PT.get_structsim(observations[int(observations.shape[2]/2),:,:],ref[int(ref.shape[2]/2),:,:])
        rewxl=PT.get_structsim(observations[int(observations.shape[2]/2)-5,:,:],ref[int(ref.shape[2]/2),:,:])
        rewxr=PT.get_structsim(observations[int(observations.shape[2]/2)+5,:,:],ref[int(ref.shape[2]/2),:,:])
        rewxlb=PT.get_structsim(observations[int(observations.shape[2]/2)-10,:,:],ref[int(ref.shape[2]/2),:,:])
        rewxrb=PT.get_structsim(observations[int(observations.shape[2]/2)+10,:,:],ref[int(ref.shape[2]/2),:,:])
        xl=[rewxl, rewxlb]
        itxl=xl.index(max(xl))
        xr=[rewxr, rewxrb]
        itxr=xr.index(max(xr))
        xv=[xl[itxl], rewym, xr[itxr]] #+1 0 -1
        itx=xv.index(max(xv))
        print('x is done')
        tit=observations.shape
        action=[tom[itx],tom[ity],tom[itz]]
        action=switch_action(action,tit)
        print('action is', action)
        print('state is', state)
        observations, reward, done, state, info = env.step(action)
        env.render()
        totalreward=np.append(totalreward,reward)
        if done:
            break
    return totalreward


bestparams = None
bestreward = 0
strw=np.array([])

for _ in range(30):
    __=_
    print('iteration', __)
    _=gym.make('PET-v0')
    _.render()
    _.reset()
    tom[0]=random.randint(0,9)
    tom[1]=random.randint(0,9)
    tom[2]=random.randint(0,9)
    reward = run_episode(_,parameters,tom)
    strw=np.append(strw,reward)
    if max(reward) > bestreward:
        bestreward=max(reward)
        bestparams = tom
        if max(reward) >0.99:
            break

PT.plt.hist(strw[600:800],100)





