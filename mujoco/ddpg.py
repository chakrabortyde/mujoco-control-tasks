import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, concatenate

ENV_NAME = 'Ant-v2'


def ou_noise(x, mu, sigma, theta):
    return theta * (mu - x) + sigma * np.random.randn(1)


def actor_network(env):
    actor = Sequential()
    actor.add(Dense(32, input_shape=(env.observation_space.shape[0],)))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(env.action_space.shape[0]))
    actor.add(Activation('linear'))
    return actor


def critic_network(env):
    action_input = tf.keras.Input((env.action_space.shape[0],))
    state_input = tf.keras.Input((env.observation_space.shape[0],))
    concatenate_input = concatenate([state_input, action_input])
    critic = Dense(32)(concatenate_input)
    critic = Activation('relu')(critic)
    critic = Dense(16)(critic)
    critic = Activation('relu')(critic)
    critic = Dense(1)(critic)
    critic = Activation('linear')(critic)
    critic = tf.keras.Model(inputs=[state_input, action_input], outputs=critic)
    return critic


def actor_loss(critic, state):
    def maximize_Q(action_true, action_pred):
        return - critic([tf.reshape(state, (1, -1)), tf.reshape(action_pred, (1, action_pred.shape[1]))])
    return maximize_Q


def critic_loss():
    def mse(y, Q):
        return tf.reduce_mean(tf.square(y - Q))
    return mse


def target_network_loss(tau, weights, target_weights):
    updated_weights = []
    for i, weight in enumerate(weights):
        updated_weights.append(tau * weight + (1 - tau) * target_weights[i])
    return updated_weights


def set_replay_buffer(mem_limit, env):
    memory_buffer = []
    for _ in range(mem_limit):
        observation = env.reset().reshape(1, -1)
        action = env.action_space.sample().reshape(1, -1)
        next_observation, reward, _, __ = env.step(action)
        memory_buffer.append([observation, action, reward, next_observation.reshape(1, -1)])
    return memory_buffer


def main():
    env = gym.make(ENV_NAME)
    discount_factor = 0.4
    tau = 0.99
    theta = 0.15
    mu = 0
    sigma = 0.3
    mem_limit = 10000
    N = 100

    actor = actor_network(env)
    critic = critic_network(env)

    target_actor = actor_network(env)
    target_actor.set_weights(actor.get_weights())
    target_critic = critic_network(env)
    target_critic.set_weights(critic.get_weights())

    state = env.reset().reshape(-1, 1)

    actor.compile(optimizer='adam', loss=actor_loss(critic, state))
    critic.compile(optimizer='adam', loss=critic_loss())

    replay_buffer = set_replay_buffer(mem_limit, env)

    for episode in range(10):
        observation, done = env.reset().reshape(1, -1), False
        state = observation
        avg_reward = 0
        count = 0
        noise = np.random.randn(1, env.action_space.shape[0])

        while not done:
            action = actor.predict(observation) + ou_noise(noise, mu, sigma, theta)
            next_observation, reward, done, _ = env.step(action)
            next_observation = next_observation.reshape(1, -1)
            replay_buffer.append([observation, action, reward, next_observation])
            minibatch = random.sample(replay_buffer, (N % len(replay_buffer)))
            y = reward + (discount_factor * target_critic.predict(
                [next_observation, target_actor.predict(next_observation)]))

            # y = []
            # observations = []
            # actions = []
            # for index in range(len(minibatch)):
            #     y.append(minibatch[index][2] + discount_factor * target_critic.predict([minibatch[index][3], target_actor.predict(minibatch[index][3])]).flatten())
            #     observations.append(minibatch[index][0].flatten())
            #     actions.append(minibatch[index][1].flatten())
            #
            # observations = np.array(observations).reshape(N, env.observation_space.shape[0])
            # actions = np.array(actions).reshape(N, env.action_space.shape[0])
            # y = np.array(y).reshape(N, 1)

            critic.fit(x=[observation, action], y=y, verbose=0)
            actor.fit(x=observation, y=action, verbose=0)

            target_critic.set_weights(target_network_loss(tau, critic.weights, target_critic.weights))
            target_actor.set_weights(target_network_loss(tau, actor.weights, target_actor.weights))

            env.render()

            avg_reward += reward
            count += 1
            observation = next_observation
            state = observation

            print("Step: ", count + 1, "Total Reward per epoch: ", avg_reward)
        print("Average Reward: ", avg_reward / count, " Step: ", count)


if __name__ == '__main__':
    main()
