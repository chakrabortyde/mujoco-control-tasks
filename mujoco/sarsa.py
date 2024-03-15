import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

env = gym.make("HalfCheetah-v2")

alpha = 0.01
episodes = [i for i in range(100)]
state_n_bins = 2
action_n_bins = 2
state_min_value = -25
state_max_value = 25
action_shape = env.action_space.shape
state_shape = env.observation_space.shape
action_bound_low = env.action_space.low
action_bound_high = env.action_space.high
state_bound_low = env.observation_space.low
state_bound_low[state_bound_low == -np.inf] = state_min_value
state_bound_high = env.observation_space.high
state_bound_high[state_bound_high == np.inf] = state_max_value
shape = (np.power(state_shape[0], state_n_bins), np.power(action_shape[0], action_n_bins))

q_table = np.zeros(shape)


def digitize_action(x):
    est = KBinsDiscretizer(n_bins=action_n_bins, encode='ordinal', strategy='uniform')
    est.fit([action_bound_low, action_bound_high])
    return est.transform(np.reshape(x, (1, -1)))[0].astype(int)


def digitize_state(x):
    est = KBinsDiscretizer(n_bins=state_n_bins, encode='ordinal', strategy='uniform')
    est.fit([state_bound_low, state_bound_high])
    return est.transform(np.reshape(x, (1, -1)))[0].astype(int)


def max_q(state):
    return np.argmax(q_table[state])


def next_q_value(reward, next_state, discount_factor=0.99):
    return reward + discount_factor * np.max(q_table[next_state])


def exploration_rate(n, min_rate=0.1):
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


# log = ""
avg_reward_l1 = []
avg_reward_l2 = []
avg_reward_l3 = []
avg_reward_l4 = []
avg_reward_l5 = []
total_rewards = 0
list_of_alpha = [0.01, 0.05, 0.1, 0.5, 1]
print("Running {0} episodes...".format(episodes))
for alpha in list_of_alpha:
    for i, e in enumerate(range(100)):
        iterations = 0
        tot_reward = 0
        prev_state, done = digitize_state(env.reset()), False
        while not done:
            action = digitize_action(env.action_space.sample())

            observation, reward, done, info = env.step(action)
            next_state = digitize_state(observation)

            next_q = next_q_value(reward, next_state)
            prev_q = q_table[prev_state][action]
            q_table[prev_state][action] = (1 - alpha) * prev_q + alpha * next_q

            tot_reward = tot_reward + reward
            iterations = iterations + 1
            prev_state = next_state
            # env.render()
        if alpha == list_of_alpha[0]:
            avg_reward_l1.append(tot_reward / iterations)
        if alpha == list_of_alpha[1]:
            avg_reward_l2.append(tot_reward / iterations)
        if alpha == list_of_alpha[2]:
            avg_reward_l3.append(tot_reward / iterations)
        if alpha == list_of_alpha[3]:
            avg_reward_l4.append(tot_reward / iterations)
        if alpha == list_of_alpha[4]:
            avg_reward_l5.append(tot_reward / iterations)
        total_rewards = total_rewards + tot_reward
        # log += "{0},{1},{2},{3}\n".format(i, tot_reward, tot_reward / iterations, total_rewards)
        print("Average reward for episode {0} \t: {1}".format(i, tot_reward / iterations))
env.close()

plt.plot(episodes, avg_reward_l1, label='0.01')
plt.plot(episodes, avg_reward_l2, label='0.05')
plt.plot(episodes, avg_reward_l3, label='0.1')
plt.plot(episodes, avg_reward_l4, label='0.5')
plt.plot(episodes, avg_reward_l5, label='1')
plt.legend()
plt.show()

# with open("log.txt", "w+") as file:
#     file.write(log)
