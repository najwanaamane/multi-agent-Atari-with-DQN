import pygame
import gym
import numpy as np
from dqn_agent import DQNAgent  # Import your DQNAgent class

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
        """Take an action in the environment."""
        state, reward, done, truncated, info = self.env.step(action)
        self.done = done or truncated
        return np.array(state), reward, self.done, info

    def render(self):
        """Render the game environment."""
        frame = self.env.render()  
        pygame.surfarray.blit_array(self.screen, frame.swapaxes(0, 1))
        pygame.display.flip()

    def close(self):
        """Close the game environment."""
        self.env.close()

# Main simulation script
def run_simulation():
    # Create a shared Breakout environment
    env = BreakoutEnv()
    
    # Create two agents
    agent1 = DQNAgent(state_space=(env.height, env.width, 3), action_space=env.env.action_space.n)
    agent2 = DQNAgent(state_space=(env.height, env.width, 3), action_space=env.env.action_space.n)

    episodes = 10
    total_rewards = [0, 0]

    for e in range(episodes):
        state = env.reset()
        done = False
        turn = 0  # Alternate turns between agents

        while not done:
            # Decide which agent's turn it is
            current_agent = agent1 if turn % 2 == 0 else agent2

            # Agent takes an action
            action = current_agent.act(state)

            # Environment responds
            next_state, reward, done, _ = env.step(action)
            total_rewards[turn % 2] += reward

            # Train the current agent
            current_agent.remember(state, action, reward, next_state, done)
            current_agent.train(batch_size=32)

            # Update the state and turn
            state = next_state
            turn += 1

            # Render the environment
            env.render()

        # Update epsilon for both agents
        agent1.update_epsilon()
        agent2.update_epsilon()

        # Print episode summary
        if e % 100 == 0:
            print(f"Episode {e}/{episodes}")
            print(f"  Agent 1 Total Reward: {total_rewards[0]}")
            print(f"  Agent 2 Total Reward: {total_rewards[1]}")
            total_rewards = [0, 0]

    env.close()

if __name__ == "__main__":
    run_simulation()
