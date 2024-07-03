import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict

EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999

def default_Q_value():
    return 0

if __name__ == "__main__":
    env = gym.envs.make("FrozenLake-v1")
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while not done:
            # Implementing ε-greedy policy for action selection
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  # Explore: choose a random action
            else:
                # Exploit: choose the action with max Q-value for the current state
                action = np.argmax([Q_table[(obs, a)] for a in range(env.action_space.n)])
            
            # Take the action and observe the outcome state and reward
            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # Q-value update
            # If the episode is not done, use the standard Q-learning update
            # If done, update with the reward only, since there is no next state
            if not terminated and not truncated:
                # Calculate the maximum Q-value for the actions in the new state
                next_max = np.max([Q_table[(new_obs, a)] for a in range(env.action_space.n)])
                # Update the Q-value for the current state and action
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + \
                                          LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
            else:
                # Update the Q-value with the reward only
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + \
                                          LEARNING_RATE * reward
            
            # Transition to the new state
            obs = new_obs
            # Add the reward to the total episode reward
            episode_reward += reward
            # Check if the episode is done
            done = terminated or truncated

        # Decay the value of EPSILON
        EPSILON *= EPSILON_DECAY

        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward)

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON))

    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    model_file.close()
    #########################