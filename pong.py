import streamlit as st
import numpy as np
import pygame
import gym
from gym import spaces
from dqn_agent import DQNAgent  
import random

# Multi-Agent Pong Environment 
class MultiAgentPongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.width, self.height = 400, 300
        self.paddle_width, self.paddle_height = 10, 60
        self.ball_size = 10

        self.action_space = spaces.Discrete(3)  # [0: stay, 1: up, 2: down]
        self.observation_space = spaces.Box(low=0, high=255, shape=(5,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.reset()

    def reset(self):
        self.paddle1_y = self.height // 2 - self.paddle_height // 2
        self.paddle2_y = self.height // 2 - self.paddle_height // 2
        self.ball_x, self.ball_y = self.width // 2, self.height // 2
        self.ball_dx, self.ball_dy = np.random.choice([-3, 3]), np.random.choice([-2, 2])
        self.rewards = [0, 0]
        self.done = False
        return self._get_observation()

    def step(self, actions):
        self._move_paddle(actions[0], "paddle1")
        self._move_paddle(actions[1], "paddle2")

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_dy *= -1

        if self._check_collision("paddle1"):
            self.ball_dx *= -1
        if self._check_collision("paddle2"):
            self.ball_dx *= -1

        if self.ball_x <= 0:
            self.rewards[1] += 1
            self.done = True
        elif self.ball_x >= self.width:
            self.rewards[0] += 1
            self.done = True

        return self._get_observation(), self.rewards, self.done, {}

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (10, self.paddle1_y, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.width - 20, self.paddle2_y, self.paddle_width, self.paddle_height))
        pygame.draw.ellipse(self.screen, (255, 255, 255),
                            (self.ball_x, self.ball_y, self.ball_size, self.ball_size))

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))  # Convert to (height, width, channels)

    def close(self):
        pygame.quit()

    def _get_observation(self):
        return {
            "agent1": np.array([self.paddle1_y, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy], dtype=np.float32),
            "agent2": np.array([self.paddle2_y, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy], dtype=np.float32),
        }

    def _move_paddle(self, action, paddle):
        if paddle == "paddle1":
            if action == 1 and self.paddle1_y > 0:
                self.paddle1_y -= 5
            elif action == 2 and self.paddle1_y < self.height - self.paddle_height:
                self.paddle1_y += 5
        elif paddle == "paddle2":
            if action == 1 and self.paddle2_y > 0:
                self.paddle2_y -= 5
            elif action == 2 and self.paddle2_y < self.height - self.paddle_height:
                self.paddle2_y += 5

    def _check_collision(self, paddle):
        if paddle == "paddle1":
            return (self.ball_x <= 20 and
                    self.paddle1_y <= self.ball_y <= self.paddle1_y + self.paddle_height)
        elif paddle == "paddle2":
            return (self.ball_x >= self.width - 30 and
                    self.paddle2_y <= self.ball_y <= self.paddle2_y + self.paddle_height)

# Simulation 
def run_simulation():
    st.title("Multi-Agent Pong Simulation")
    env = MultiAgentPongEnv()

    # Pass the action space and state space size directly to the DQNAgent
    state_size = 5  # Observation size
    action_size = env.action_space.n  # Number of actions available
    
    agent1 = DQNAgent(state_size=state_size, action_size=action_size)
    agent2 = DQNAgent(state_size=state_size, action_size=action_size)

    episodes = 10
    batch_size = 32
    frame_placeholder = st.empty()

    for e in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action1 = agent1.act(obs["agent1"])
            action2 = agent2.act(obs["agent2"])
            actions = [action1, action2]
            next_obs, rewards, done, _ = env.step(actions)

            agent1.remember(obs["agent1"], action1, rewards[0], next_obs["agent1"], done)
            agent2.remember(obs["agent2"], action2, rewards[1], next_obs["agent2"], done)
            agent1.replay(batch_size)
            agent2.replay(batch_size)

            obs = next_obs

            frame = env.render()
            frame_placeholder.image(frame, channels="RGB")

    env.close()

if __name__ == "__main__":
    run_simulation()
