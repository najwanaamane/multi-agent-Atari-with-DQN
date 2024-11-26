import pygame
import gym
import numpy as np
from dqn_agent import DQNAgent  

np_bool = getattr(np, 'bool', np.bool_)

class BreakoutEnv:
    def __init__(self):
        # Initialize the Breakout environment
        self.env = gym.make("Breakout-v0", render_mode="rgb_array")  # Render mode set to 'rgb_array'
        self.width, self.height = 400, 600  # Default size of the Breakout window
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.done = False
        self.score = 0

    def reset(self):
        """Reset the environment to its initial state.""" 
        state, _ = self.env.reset()
        self.score = 0
        self.done = False
        return np.array(state)

    def step(self, action):
        """Take action for both agents in the game environment."""
        state, reward, done, truncated, info = self.env.step(action)
        
        if done or truncated:
            self.done = True
        return np.array(state), reward, self.done, {}

    def render(self):
        """Render the game environment."""
        frame = self.env.render()  
        return frame

    def close(self):
        """Close the game environment."""
        self.env.close()


# Main simulation script
def run_simulation():
    # Create the Breakout environment and two agents (independent environments)
    env1 = BreakoutEnv() 
    env2 = BreakoutEnv() 
    
    agent1 = DQNAgent(state_space=(env1.height, env1.width, 3), action_space=env1.env.action_space.n)
    agent2 = DQNAgent(state_space=(env2.height, env2.width, 3), action_space=env2.env.action_space.n)

    episodes = 1000
    total_reward = 0
    for e in range(episodes):
        state1 = env1.reset()  # Reset environment for agent 1
        state2 = env2.reset()  # Reset environment for agent 2
        
        done1 = done2 = False
        while not done1 and not done2:
            # Agent 1 takes action
            action1 = agent1.act(state1)
            
            # Agent 2 takes action
            action2 = agent2.act(state2)
            
            # Take the step for both agents in their respective environments
            next_state1, reward1, done1, _ = env1.step(action1)
            next_state2, reward2, done2, _ = env2.step(action2)
            
            # Accumulate rewards (you can adjust this as needed)
            total_reward += reward1 + reward2
            
            # Train both agents
            agent1.remember(state1, action1, reward1, next_state1, done1)
            agent2.remember(state2, action2, reward2, next_state2, done2)

            agent1.train(batch_size=32)
            agent2.train(batch_size=32)

            # Move to the next state for both agents
            state1 = next_state1
            state2 = next_state2
            
            # Render the environments (two different game boards)
            env1.render()  # Show agent 1's board
            env2.render()  # Show agent 2's board

        agent1.update_epsilon()
        agent2.update_epsilon()

        if e % 100 == 0:
            print(f"Episode {e}/{episodes}, Total Reward: {total_reward}")
            total_reward = 0

    env1.close()  # Close environment for agent 1
    env2.close()  # Close environment for agent 2


if __name__ == "__main__":
    run_simulation()
