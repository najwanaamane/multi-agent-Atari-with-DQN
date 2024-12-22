import pygame
import random
import numpy as np
from dqn_agent import DQNAgent  # Assuming DQNAgent is defined in dqn_agent.py

# Space Invaders Environment for Cooperative Agents
class SpaceInvadersEnv:
    def __init__(self, num_agents=3):
        pygame.init()

        # Screen dimensions
        self.width, self.height = 400, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Space Invaders - Cooperative Mode")

        # Colors
        self.bg_color = (0, 0, 0)
        self.agent_color = (0, 255, 0)  # Shared color for the cooperative agents
        self.enemy_color = (255, 0, 0)
        self.bullet_color = (0, 0, 255)

        # Game settings
        self.agent_width, self.agent_height = 40, 10
        self.enemy_width, self.enemy_height = 30, 30
        self.bullet_width, self.bullet_height = 5, 10
        self.enemy_size = 30
        self.enemy_speed = 2
        self.agent_speed = 5

        self.num_agents = num_agents
        self.agents = []  # List of agent positions and bullet states
        self.reset()

        # Initialize enemies
        self.enemies = []
        self.create_enemies()

        # Score
        self.score = 0  # Single shared score for both agents

    def create_enemies(self):
        """Create a grid of enemies."""
        for x in range(0, self.width, self.enemy_size):
            for y in range(50, 150, self.enemy_size):
                self.enemies.append([x, y])

    def reset(self):
        """Reset the game to its initial state."""
        self.agents = []  # Reset agent positions
        agent_x = self.width // 2 - self.agent_width // 2
        agent_y = self.height - 40
        for _ in range(self.num_agents):
            self.agents.append({"x": agent_x, "y": agent_y, "bullets": []})
        self.enemies = []
        self.create_enemies()
        self.score = 0
        self.done = False

    def step(self, actions):
        """Process a step in the game with the agents' actions."""
        action = self.get_cooperative_action(actions)  # Combine the actions from all agents
        self.handle_action(action)

        self.update_bullets()
        self.update_enemies()

        # Check for collision
        self.check_collisions()

        # Return observations, rewards, done, and info
        observation = self.get_observation()
        return [observation], self.score, self.done, {}

    def get_cooperative_action(self, actions):
        """Merge actions from all agents to make a cooperative decision."""
        left, right, shoot = 0, 0, 0
        for action in actions:
            if action == 1:
                left += 1
            elif action == 2:
                right += 1
            elif action == 3:
                shoot += 1

        # Cooperative decision: Move left if more agents choose left, and similarly for right
        move_action = 0
        if left > right:
            move_action = 1  # Move left
        elif right > left:
            move_action = 2  # Move right

        # Shoot if any agent chooses to shoot
        shoot_action = 3 if shoot > 0 else 0

        return (move_action, shoot_action)

    def handle_action(self, action):
        """Handle the cooperative action (move or shoot)."""
        move_action, shoot_action = action

        # Apply the action to all agents
        for agent in self.agents:
            if move_action == 1 and agent["x"] > 0:  # Move left
                agent["x"] -= self.agent_speed
            elif move_action == 2 and agent["x"] < self.width - self.agent_width:  # Move right
                agent["x"] += self.agent_speed
            if shoot_action == 3:  # Shoot
                agent["bullets"].append([agent["x"] + self.agent_width // 2 - self.bullet_width // 2, agent["y"]])

    def update_bullets(self):
        """Update positions of all bullets."""
        for agent in self.agents:
            for bullet in agent["bullets"][:]:
                bullet[1] -= 5  # Move up
                if bullet[1] < 0:
                    agent["bullets"].remove(bullet)  # Remove bullet if it goes off screen

    def update_enemies(self):
        """Update enemy movements."""
        for enemy in self.enemies[:]:
            enemy[1] += self.enemy_speed
            if enemy[1] >= self.height:
                self.done = True  # Game over if any enemy reaches the bottom of the screen
                break

    def check_collisions(self):
        """Check for collisions between bullets and enemies."""
        for agent in self.agents:
            for bullet in agent["bullets"][:]:
                for enemy in self.enemies[:]:
                    if (bullet[0] >= enemy[0] and bullet[0] <= enemy[0] + self.enemy_size and
                            bullet[1] >= enemy[1] and bullet[1] <= enemy[1] + self.enemy_size):
                        self.enemies.remove(enemy)
                        agent["bullets"].remove(bullet)
                        self.score += 1
                        break

    def render(self):
        """Render the game on the screen."""
        self.screen.fill(self.bg_color)

        # Draw the agents (spaceship controlled by all agents)
        for agent in self.agents:
            pygame.draw.rect(self.screen, self.agent_color, (agent["x"], agent["y"], self.agent_width, self.agent_height))

        # Draw the enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.enemy_color, pygame.Rect(enemy[0], enemy[1], self.enemy_size, self.enemy_size))

        # Draw the bullets
        for agent in self.agents:
            for bullet in agent["bullets"]:
                pygame.draw.rect(self.screen, self.bullet_color, pygame.Rect(bullet[0], bullet[1], self.bullet_width, self.bullet_height))

        # Display the score
        font = pygame.font.SysFont('Arial', 24)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def get_observation(self):
        """Return the current screen as the observation."""
        return np.array(pygame.surfarray.array3d(self.screen))

    def close(self):
        """Close the environment."""
        pygame.quit()


# Main simulation script for cooperative agents
def run_simulation():
    # Create the environment and agents
    num_agents = 3
    env = SpaceInvadersEnv(num_agents=num_agents)
    agents = [DQNAgent(action_space=4, state_space=(env.height, env.width, 3)) for _ in range(num_agents)]

    episodes = 10
    for e in range(episodes):
        env.reset()
        done = False
        while not done:
            actions = [agent.act(env.get_observation()) for agent in agents]
            observations, rewards, done, _ = env.step(actions)
            for agent_idx, agent in enumerate(agents):
                agent.remember(observations[0], actions[agent_idx], rewards, observations[0], done)
                agent.learn()  # Train each agent independently
            env.render()

    env.close()


if __name__ == "__main__":
    run_simulation()
