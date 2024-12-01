import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from collections import deque

class DQNAgent:
    def __init__(self, action_space, state_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99, learning_rate=0.001, memory_size=2000):
        self.action_space = action_space
        self.state_space = state_space  
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)  # Memory buffer
        self.batch_size = 32  # Default batch size for training
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
        """Stores the agent's experiences for experience replay."""
        self.memory.append((state, action, reward, next_state, done))  # Add experience to memory buffer

    def train(self, batch_size=None):
        """Train the model based on experiences (Experience Replay)."""
        batch_size = batch_size or self.batch_size  # Use provided batch size or default
        if len(self.memory) < batch_size:
            return  # Skip training if there aren't enough experiences in memory

        # Sample a random batch from memory
        minibatch = random.sample(self.memory, batch_size)

        # Prepare the states and targets for batch processing
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            # Ensure state and next_state have the correct shape: (height, width, channels)
            state = np.squeeze(state)  # Remove extra dimensions: (210, 160, 3)
            next_state = np.squeeze(next_state)  # Same for next state
            print(f"State shape before training: {state.shape}")  # Debugging line

            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])

            target_f = self.model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target

            # Append the state and target to the lists
            states.append(np.expand_dims(state, axis=0))  # Add batch dimension back
            targets.append(target_f[0])

        # Train the model on the entire batch
        self.model.fit(np.vstack(states), np.vstack(targets), batch_size=batch_size, epochs=1, verbose=0)

    def update_epsilon(self):
        """Decays epsilon after each episode to reduce exploration."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
