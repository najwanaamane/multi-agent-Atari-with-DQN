# dqn_agent.py

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

class DQNAgent:
    def __init__(self, action_space, state_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99, learning_rate=0.001):
        self.action_space = action_space
        self.state_space = state_space  
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        """Builds a simple Convolutional Neural Network for Q-value prediction."""
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=self.state_space))
        model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.action_space, activation='linear'))  # Output Q-values for each action
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        """Chooses an action based on epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_space))  # Explore
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def remember(self, state, action, reward, next_state, done):
        """Stores the agent's experiences for experience replay (optional)."""
        # Placeholder method for memory replay, can be expanded later
        pass

    def train(self, batch_size):
        """Train the model based on experiences (Experience Replay)."""
        # Placeholder for training method, experience replay would be added here
        pass

    def update_epsilon(self):
        """Decays epsilon after each episode to reduce exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
