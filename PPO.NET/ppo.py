"""
Title: Proximal Policy Optimization
Author: [Ilias Chrysovergis](https://twitter.com/iliachry)
Date created: 2021/06/24
Last modified: 2021/06/24
Description: Implementation of a Proximal Policy Optimization agent for the CartPole-v0 environment.
Accelerator: NONE
"""

"""
## Introduction

This code example solves the CartPole-v0 environment using a Proximal Policy Optimization (PPO) agent.

### CartPole-v0

A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The system is controlled by applying a force of +1 or -1 to the cart.
The pendulum starts upright, and the goal is to prevent it from falling over.
A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
After 200 steps the episode ends. Thus, the highest return we can get is equal to 200.

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)

### Proximal Policy Optimization

PPO is a policy gradient method and can be used for environments with either discrete or continuous action spaces.
It trains a stochastic policy in an on-policy way. Also, it utilizes the actor critic method. The actor maps the
observation to an action and the critic gives an expectation of the rewards of the agent for the observation given.
Firstly, it collects a set of trajectories for each epoch by sampling from the latest version of the stochastic policy.
Then, the rewards-to-go and the advantage estimates are computed in order to update the policy and fit the value function.
The policy is updated via a stochastic gradient ascent optimizer, while the value function is fitted via some gradient descent algorithm.
This procedure is applied for many epochs until the environment is solved.

![Algorithm](https://i.imgur.com/rd5tda1.png)

- [PPO Original Paper](https://arxiv.org/pdf/1707.06347.pdf)
- [OpenAI Spinning Up docs - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### Note

This code example uses Keras and Tensorflow v2. It is based on the PPO Original Paper,
the OpenAI's Spinning Up docs for PPO, and the OpenAI's Spinning Up implementation of PPO using Tensorflow v1.

[OpenAI Spinning Up Github - PPO](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py)
"""

"""
## Libraries

For this example the following libraries are used:

1. `numpy` for n-dimensional arrays
2. `tensorflow` and `keras` for building the deep RL PPO agent
3. `gym` for getting everything we need about the environment
4. `scipy.signal` for calculating the discounted cumulative sums of vectors
"""
"""
## Functions and class
"""




import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PPO:
    def __init__(
        self,
        observation_dimensions,
        num_actions,
        steps_per_epoch,
        policy_learning_rate=3e-4,
        value_function_learning_rate=1e-3,
        clip_ratio=0.2,
        hidden_sizes=(64, 64),
        gamma=0.99,
        lam=0.95
    ):
        self.observation_dimensions = observation_dimensions
        self.steps_per_epoch = steps_per_epoch
        self.hidden_sizes = hidden_sizes
        self.num_actions = num_actions
        self.policy_learning_rate = policy_learning_rate
        self.value_function_learning_rate = value_function_learning_rate
        self.clip_ratio = clip_ratio

        # Initialize the buffer
        #self.buffer = Buffer(
        #    self.observation_dimensions,
        #    self.steps_per_epoch
        #)

        # Initialize the actor and the critic as keras models
        observation_input = keras.Input(
            shape=(self.observation_dimensions,),
            dtype=tf.float32
        )

        logits = self.mlp(
            observation_input,
            list(self.hidden_sizes) + [self.num_actions],
            tf.tanh,
            None
        )

        self.actor = keras.Model(inputs=observation_input, outputs=logits)

        value = tf.squeeze(
            self.mlp(
                observation_input,
                list(self.hidden_sizes) + [1],
                tf.tanh,
                None
            ),
            axis=1
        )

        self.critic = keras.Model(inputs=observation_input, outputs=value)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(
            learning_rate=self.policy_learning_rate
        )

        self.value_optimizer = keras.optimizers.Adam(
            learning_rate=self.value_function_learning_rate
        )

    def mlp(self, x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    @tf.function
    def sample_action(self, observation):
        # Sample action from actor
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    @tf.function
    def train_policy(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        # Train the policy by maxizing the PPO-Clip objective
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(
                    observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(
            policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.actor.trainable_variables))

        self.kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        self.kl = tf.reduce_sum(self.kl)
        return self.kl

    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        # Train the value function by regression on mean-squared error
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean(
                (return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(
            value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.critic.trainable_variables))

    def save(self, path):
        """Salva o modelo treinado em um arquivo."""
        self.actor.save(f"{path}_actor.h5")
        self.critic.save(f"{path}_critic.h5")

    def load(self, path):
        """Carrega um modelo previamente treinado de um arquivo."""
        self.actor = keras.models.load_model(f"{path}_actor.h5")
        self.critic = keras.models.load_model(f"{path}_critic.h5")


"""
## Visualizations

Before training:

![Imgur](https://i.imgur.com/rKXDoMC.gif)

After 8 epochs of training:

![Imgur](https://i.imgur.com/M0FbhF0.gif)

After 20 epochs of training:

![Imgur](https://i.imgur.com/tKhTEaF.gif)
"""
