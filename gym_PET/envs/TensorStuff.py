
#define graph

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

def policy_gradient():
    with tf.variable_scope("policy", reuse=tf.AUTO_REUSE ):
    #for bounding box of 30,30,30
    #Initialize parameters to avoid boundary
        bl=[30, 30, 30] 
        params = tf.get_variable("policy_parameters",[30,30,30,3]) #this is whats changed to minimize loss
        actions = tf.placeholder("float",[None,3])
        paramsr=tf.reshape(params, [params.shape[0]*params.shape[1]*params.shape[2],params.shape[3]])
        state = tf.placeholder("float",[None,params.shape[0]*params.shape[1]*params.shape[2]])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,paramsr)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
        log_probabilities = tf.log(good_probabilities)
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return params, linear, probabilities, state, actions, advantages, optimizer
      

def value_gradient():
    with tf.variable_scope("value", reuse=tf.AUTO_REUSE):  
    # sess.run(calculated) to calculate value of state
        print('can we print')
        paramsaaa = tf.get_variable("policy_parametersaaa",[30,30,30,3])
        state = tf.placeholder("float",[None,paramsaaa.shape[0]*paramsaaa.shape[1]*paramsaaa.shape[2]])
        w1 = tf.get_variable("w1",[state.shape[1],10]) #altered
        b1 = tf.get_variable("b1",[10])#altered
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1]) #altered
        b2 = tf.get_variable("b2",[1]) #altered
        calculated = tf.matmul(h1,w2) + b2
        # sess.run(optimizer) to update the value of a state
        newvals = tf.placeholder("float",[None,1])
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, w1, b1, h1, w2, b2, loss

def run_episode(env, policy, value_grad, sess):
    print("Performing Run")
    params, linear, pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    params, linear, pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals,vl_optimizer,vl_w1, vl_b1, vl_h1, vl_w2, vl_b2, vl_loss = value_grad
    env.render()
    observation = env.reset() 
    env.render()
    totalreward = 0
    states = []
    lossa=[]
    actions = []
    advantages = []
    transitions = []
    update_vals = []
    probb=[]
    linbb=[]
    parb=[]
    for _ in range(10): 
        print("Obtaining examples")
    #calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        x=obs_vector
        inda=[x==1000]
        indb=[x==-1000]
        x_min=x.min(axis=(0,1,2),keepdims=True)
        x_max=x.max(axis=(0,1,2),keepdims=True)
        x = (x - x_min)/(x_max-x_min)
        x=x*2
        x[inda]=1000
        x[indb]=-1000
        obs_vectorr=x
        #obs_vectorR=np.reshape(obs_vectorr, [obs_vector.shape[0],obs_vector.shape[1]* obs_vector.shape[2]* obs_vector.shape[3]])
        obs_vectorR=np.reshape(obs_vectorr, [obs_vector.shape[0],obs_vector.shape[1]* obs_vector.shape[2]* obs_vector.shape[3]])
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vectorR})
        lin = sess.run(linear,feed_dict={pl_state: obs_vectorR})
        par = sess.run(params,feed_dict={pl_state: obs_vectorR})
        #print(lin)
        #print(probs)
        #action = 0 if random.uniform(0,1) < probs[0][0] else 1
        actionn=np.around((probs-0.5)*10)
        action=actionn[0]
        #action[0]=0
        #action[1]=0
        #action[2]=-5
        #record the transition
        #states.append(observation)
        states.append(obs_vectorR)
        #actionblank = action
        #actions.append(actionblank)
        probb.append(probs)
        linbb.append(lin)
        parb.append(par)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, statea, actiontook, info = env.step(action)
        actionblank = actiontook
        actions.append(actionblank)
        env.render()
        transitions.append((old_observation, actiontook, reward))
        totalreward += reward
        if done:
            break
    for index, trans in enumerate(transitions):
        print("Iterating through examples")
        #print("index is", index)  
        obs, action, reward = trans
        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            print("Finding Future rewards")
            #print("index2 is", index2) 
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        obs_vectorR=np.reshape(obs_vector, [obs_vector.shape[0],obs_vector.shape[1]* obs_vector.shape[2]* obs_vector.shape[3]])
        print("Feeding through Neural Network")
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vectorR})[0][0]
        advantages.append(future_reward - currentval)
        update_vals.append(future_reward)
    print("Updating Neural Network")
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: np.squeeze(states), vl_newvals: update_vals_vector})
    lossa.append(sess.run(vl_loss, feed_dict={vl_state: np.squeeze(states), vl_newvals: update_vals_vector}))
    advantages_vector = np.expand_dims(advantages, axis=1)
    print("Updating Policy")
    sess.run(pl_optimizer, feed_dict={pl_state: np.squeeze(states), pl_advantages: advantages_vector, pl_actions: actions})
    return totalreward, lossa

#xa=[]
#ya=[]
#za=[]
#pxa=[]
#pya=[]
#pza=[]

#for t in range(200):
#    xa.append(actions[t][0])
#    ya.append(actions[t][1])
#    za.append(actions[t][2])
#    pxa.append(probb[t][0][0])
#    pya.append(probb[t][0][1])
#    pza.append(probb[t][0][2])


#plt.hist(pxa, normed=True, bins=10)
#plt.hist(pya, normed=True, bins=10)
#plt.hist(pza, normed=True, bins=10)
#plt.imshow

#plt.hist(xa, normed=True, bins=10)
#plt.hist(ya, normed=True, bins=10)
#plt.hist(za, normed=True, bins=10)
#plt.imshow

#Next, we compute the return of each transition, and update the neural network to reflect this. We don't care about the specific action we took from each state, only what the average return for the state over all actions is.

advantages_vector=[]
update_vals=[]
advantages=[]

env.render()
env.reset()
env.render()
policy_grad = policy_gradient()
value_grad = value_gradient()

graph1 = tf.Graph()
sess = tf.Session(graph=graph1)
sess = tf.InteractiveSession()
tf.global_variables_initializer
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())

rr=[]

print ("running episode")
reward, ll = run_episode(env, policy_grad, value_grad, sess)
rr.append(reward)

reward=[]
rewarda=[]
ll=[]
lla=[]

for i in range(200):
    print ("running episode", i)
    reward, ll = run_episode(env, policy_grad, value_grad, sess)
    rewarda.append(reward)
    lla.append(ll)
    rr.append(reward)

