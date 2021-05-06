import gym
import numpy as np
import time
import numpy as np

env = gym.make('FrozenLake-v0')

STATES = env.observation_space.n
#number of state
print(STATES)

ACTIONS = env.action_space.n
#number of action for each state
print(ACTIONS)


#Yo chai Qtable create garya initially zero
Q = np.zeros((STATES, ACTIONS))


EPISODES = 2000
MAX_STEPS = 100
LEARNING_RATE = 0.81
GAMMA = 0.96 #discount
RENDER = False #initailly

epsilon = 0.9 #90% chances of using random action


rewards = []
for episode in range(EPISODES):
    state = env.reset() #reset to default state
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()

        #pick a action
        if np.random.uniform(0,1) < epsilon: #check if a randomly selected value is less then epsilon
            action = env.action_space.sample() #take random action
        else:
            action = np.argmax(Q[state, :]) #use Q table action

        new_state, reward, done, _ = env.step(action)
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state,:]) - Q[state, action])
        state = new_state
        

        if done:
            rewards.append(reward)
            print(f'state {state} with action {action} and reward {reward}')
            epsilon-=0.001
            break #reached goal


print(Q)
count = 0
for i in Q:
    print(f"For state {count} action {np.argmax(i)} with reward {np.max(i)}")
    count+=1
print(f"Average Rewaard: {sum(rewards)/len(rewards)}")


