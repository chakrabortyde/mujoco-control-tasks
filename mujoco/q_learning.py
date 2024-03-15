# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import random
import time
import gym
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import tensorflow as tf
from tensorboard import main as tb
import matplotlib.pyplot as plt


def epsilon_greedy_action_selection(epsilon, Q_values, env, est_, state):
    if random.uniform(0, 1) < epsilon:
        continuous_action = env.action_space.sample().reshape(1, -1)
        discrete_action = tuple(map(int, est_.transform(continuous_action).flatten()))

    else:
        max_Q_value = 0
        max_action = None
        for indices in np.ndindex(Q_values.shape[env.observation_space.shape[0]:]):
            if max_action is None:
                max_action = tuple(indices)
                max_Q_value = Q_values[state + max_action]
                continue
            action = tuple(indices)
            if Q_values[state + action] > max_Q_value:
                max_Q_value = Q_values[state + action]
                max_action = action
        discrete_action = max_action
        continuous_action = est_.inverse_transform(np.array(max_action).reshape(1, -1))

    return continuous_action, discrete_action


'''
Problem 1 - not taking greedy actions (how to take?)
Problem 2 - not setting the openai-gym environment to a specific state. After getting the next state, the next action
            is generated from env.action_sample(), which returns a sample action from some random state and not the 
            specific next state.
Problem 3 - clipping of observations having infinite ranges by sampling 'n' observations, getting the maximum and minimum
            values of each observation and then setting them as the thresholds of max and min ranges.
'''


def cheetahEnvironment():
    env = gym.make("HalfCheetah-v3")

    '''
    Initialize the number of bins to discretize continuous states and actions.
    '''
    ns_bin = 2
    na_bin = 2

    n_state_bins = [ns_bin] * env.observation_space.shape[0]
    n_action_bins = [na_bin] * env.action_space.shape[0]

    '''
    Getting the lower and upper bounds for states and actions
    '''
    lower_bounds_state, upper_bounds_state = [-25] * env.observation_space.shape[0], [25] * env.observation_space.shape[
        0]  # get_state_bounds(env, n_obs)
    lower_bounds_action, upper_bounds_action = list(env.action_space.low), list(env.action_space.high)

    est_state = KBinsDiscretizer(ns_bin, encode='ordinal')
    est_action = KBinsDiscretizer(na_bin, encode='ordinal')
    est_state.fit([lower_bounds_state, upper_bounds_state])
    est_action.fit([lower_bounds_action, upper_bounds_action])

    Q_values = np.zeros(tuple(n_state_bins) + tuple(n_action_bins, ))
    epsilon = 0.99
    learning_rate = 0.01
    discount = 0.99
    total_reward = 0

    writer = tf.summary.create_file_writer('./.logs')
    writer.set_as_default()

    average_reward = []
    for epoch in range(1000):
        avg_reward = 0
        total_reward_per_epoch = 0
        continuous_observation = env.reset()
        steps = 0

        while True:
            state = tuple(map(int, tuple(est_state.transform(continuous_observation.reshape(1, -1)).flatten())))
            continuous_action, action = epsilon_greedy_action_selection(epsilon, Q_values, env, est_action, state)

            # get the next observation from the environment by performing and action; discretize that observation
            continuous_next_observation, reward, done, _ = env.step(continuous_action)
            next_state = tuple(
                map(int, tuple(est_state.transform(continuous_next_observation.reshape(1, -1)).flatten())))

            # obtain the maximum Q value for the next state i.e. max Q(s', a')
            max_Q_value_next_state = 0
            for indices in np.ndindex(Q_values.shape[env.observation_space.shape[0]:]):
                action_index = tuple(indices)
                if Q_values[next_state + action_index] > max_Q_value_next_state:
                    max_Q_value_next_state = Q_values[next_state + action_index]

            # learning update equation
            Q_values[state + action] += learning_rate * (reward + (discount * max_Q_value_next_state)
                                                         - Q_values[state + action])

            avg_reward += reward
            steps += 1
            continuous_observation = continuous_next_observation

            # env.render()

            if done:
                break

        epsilon = math.log10((math.exp(epsilon) + 1) / 25)
        total_reward_per_epoch = avg_reward
        avg_reward = avg_reward / steps
        average_reward.append(avg_reward)
        total_reward += avg_reward

        tf.summary.scalar("Total Reward per Epoch", total_reward_per_epoch, step=epoch)
        tf.summary.scalar("Average Reward per Epoch", avg_reward, step=epoch)
        tf.summary.scalar("Total Reward", total_reward, step=epoch)

        print("Epoch: ", epoch + 1, "Total reward: ", total_reward_per_epoch, "Avg_Reward: ", avg_reward)

    writer.flush()

    # state, done = env.reset(), False
    #
    # while not done:
    #
    #     max_Q_value_next_state = 0
    #     for indices in np.ndindex(Q_values.shape[env.observation_space.shape[0]:]):
    #         action_index = tuple(indices)
    #         if Q_values[next_state + action_index] > max_Q_value_next_state:
    #             max_Q_value_next_state = Q_values[next_state + action_index]
    #
    #
    plt.plot(average_reward)
    plt.show()
    env.close()


def create_tiling(feat_range, bins, offset):
    return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + offset


def create_tilings(feat_ranges, bins, offsets, n_tilings):
    tilings = []

    for tile_index in range(n_tilings):
        tiling_bin = bins[tile_index]
        tiling_offset = offsets[tile_index]

        tiling = []

        for feat_index in range(len(feat_ranges)):
            feat_range = feat_ranges[feat_index]
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_index], tiling_offset[feat_index])
            tiling.append(feat_tiling)

        tilings.append(tiling)

    return np.array(tilings)


def get_tile_coding(feature, tilings):
    """
    feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
    tilings: tilings with a few layers
    return: the encoding for the feature on each layer
    """
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]  # tiling on that dimension
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)


def QLinearFunctionApproximation():
    feat_ranges = [[-25, 25]] * 17
    n_tilings = 3
    bins = [[10, 10] * 17] * 3
    offsets = [[0] * 17, [x * 0.1 for x in range(1, 18)], [x * 0.4 for x in range(1, 18)]]

    tilings = create_tilings(feat_ranges, bins, offsets, n_tilings)
    feature = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1]
    coding = get_tile_coding(feature, tilings)

    feature_Vector = []

    for i in range(len(coding[0])):
        sum_ = 0
        for j in range(n_tilings):
            sum_ += coding[j, i]
        feature_Vector.append(sum_)

    print(feature_Vector)


def feature_extractor():
    env = gym.make('HalfCheetah-v3')
    observations = []
    actions = []

    state = env.reset()
    print("Observation space: ", env.observation_space.shape)
    print("Action space: ", env.action_space.shape)

    print("Sample Observation:", env.observation_space.sample())
    print("Sample Action:", env.action_space.sample())

    for _ in range(1):
        observations.append(state)
        action = env.action_space.sample()
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        print(info)
        # state = next_state

    # observations = np.array(observations)
    # actions = np.array(actions)
    #
    # X = (observations - np.min(observations)) / (np.max(observations) - np.min(observations))
    # Y = (actions - np.min(actions)) / (np.max(actions) - np.min(actions))
    #
    # input_obs = tf.keras.Input(shape=observations[0].shape)
    # input_actions = tf.keras.Input(shape=actions[0].shape)
    #
    # input_ = tf.keras.layers.Concatenate(axis=-1)([input_obs, input_actions])
    # hidden_1 = tf.keras.layers.Dense(units=200, activation='relu')(input_)
    # hidden_2 = tf.keras.layers.Dense(units=200, activation='relu')(hidden_1)
    # output_ = tf.keras.layers.Dense(units=actions.shape[1], activation='sigmoid')(hidden_2)
    #
    # model = tf.keras.models.Model(inputs=[input_obs, input_actions], outputs=output_)
    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.fit(x=[X, Y], y=Y, epochs=3)
    #
    # # feature_Vector = model.predict([X[-1].reshape(1, -1), Y[-1].reshape(1, -1)])
    #
    # weights = np.random.randn(env.action_space.shape[0])
    # discount = 0.2
    # learning_rate = 0.8
    #
    # state = env.reset()
    # for epoch in range(10):
    #     env.render()
    #
    #     action = env.action_space.sample()
    #     next_state, reward, done, _ = env.step(action)
    #
    #     feats_curr_state = model.predict([state.reshape(1, -1), action.reshape(1, -1)])
    #     Q_values_curr_state = np.multiply(weights, feats_curr_state)
    #
    #     feats_next_state = model.predict([next_state.reshape(1, -1), env.action_space.sample().reshape(1, -1)])
    #     Q_values_next_state = np.multiply(weights, feats_next_state)
    #
    #     difference = (reward + (discount * np.max(Q_values_next_state))) - Q_values_curr_state
    #     weights = weights + (learning_rate * difference * feats_curr_state)
    #
    #     state = next_state
    #
    # state, done = env.reset(), False
    # while not done:
    #     action = env.action_space.sample()
    #     action = np.multiply(weights, model.predict(state.reshape(1, -1)))
    #     next_state, _, done, _ = env.step(action)
    #     state = next_state
    #     env.render()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # QLinearFunctionApproximation()
    # feature_extractor()
    cheetahEnvironment()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
