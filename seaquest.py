import numpy as np
import random
from dqn_agent import DQNAgent  

# Define the MultiAgentSeaquestEnv class
class MultiAgentSeaquestEnv:
    def __init__(self):
        self.width = 10  # grid size 
        self.height = 10
        self.num_agents = 2  # For multi-agent
        self.state = np.zeros((self.width, self.height, 3))  # Example RGB state space
        self.done = False

    def reset(self):
        """Reset the environment to the initial state."""
        self.state = np.zeros((self.width, self.height, 3))  # Reset the state
        self.done = False
        return self.state

    def step(self, actions):
        """Take a step in the environment based on the actions of the agents."""
        rewards = [0, 0]  # Placeholder rewards for both agents

        # Simplified reward logic for cooperation (combined rewards for cooperation)
        if actions[0] == 0 and actions[1] == 0:  # If both agents move towards the goal
            rewards = [1, 1]  # Both agents receive positive rewards for cooperation
        else:
            rewards = [-1, -1]  # Both agents are penalized for not cooperating or failing the task
        
        next_state = self.state  # Update the state based on actions (here it's simplified)
        done = self.done
        return next_state, rewards, done, {}

    def render(self, action1, action2):
        """Render the environment (visualization)."""
        print(f"Agent 1 chose action: {action1}")
        print(f"Agent 2 chose action: {action2}")
        print("Rendering environment...")

    def close(self):
        """Close the environment."""
        print("Closing environment.")


# Main simulation function
def run_simulation():
    # Initialize the environment
    env = MultiAgentSeaquestEnv()
    state = env.reset()

    # Initialize separate agents for both agents
    state_size = 3 * env.width * env.height  # Flattened state from RGB image size (example)
    action_size = 4  # Left, right, up, down
    agent1 = DQNAgent(state_size, action_size)  # Agent 1
    agent2 = DQNAgent(state_size, action_size)  # Agent 2

    done = False
    batch_size = 32  # Define batch size for experience replay

    while not done:
        # Each agent takes an action independently
        action1 = agent1.act(state)
        action2 = agent2.act(state)
        actions = [action1, action2]

        # Environment responds to both agents' actions
        next_state, rewards, done, _ = env.step(actions)
        
        # Store experiences for each agent independently
        agent1.remember(state, action1, rewards[0], next_state, done)
        agent2.remember(state, action2, rewards[1], next_state, done)
        
        # Train both agents
        agent1.replay(batch_size)
        agent2.replay(batch_size)
        
        # Render the environment and show the actions of both agents
        env.render(action1, action2)
        
        # Update state for next iteration
        state = next_state

    env.close()

# Start the simulation when the script runs
if __name__ == "__main__":
    run_simulation()
