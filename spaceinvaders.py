# spaceinvaders.py

import pygame
import random
import numpy as np
from dqn_agent import DQNAgent  

np_bool = getattr(np, 'bool', np.bool_)

class MultiAgentSpaceInvadersEnv:
    def __init__(self):
        pygame.init()

        # Screen dimensions
        self.width, self.height = 400, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Space Invaders - Multi-Agent")

        # Colors
        self.bg_color = (0, 0, 0)
        self.agent_color = (0, 255, 0)
        self.enemy_color = (255, 0, 0)
        self.bullet_color = (0, 0, 255)

        # Game settings
        self.agent_width, self.agent_height = 40, 10
        self.enemy_width, self.enemy_height = 30, 30
        self.bullet_width, self.bullet_height = 5, 10
        self.enemy_size = 30
        self.enemy_speed = 2
        self.agent_speed = 5

        self.agent1_x = self.width // 2 - self.agent_width // 2
        self.agent1_y = self.height - 40
        self.agent2_x = self.width // 2 - self.agent_width // 2
        self.agent2_y = self.height - 60

        # Initialize enemies
        self.enemies = []
        self.create_enemies()

        # Initialize bullets
        self.bullets = []
        self.enemy_bullets = []

        # Score
        self.score = 0

        # Game state
        self.done = False

    def create_enemies(self):
        """Create a grid of enemies."""
        for x in range(0, self.width, self.enemy_size):
            for y in range(50, 150, self.enemy_size):
                self.enemies.append([x, y])

    def reset(self):
        """Reset the game to its initial state."""
        self.agent1_x = self.width // 2 - self.agent_width // 2
        self.agent1_y = self.height - 40
        self.agent2_x = self.width // 2 - self.agent_width // 2
        self.agent2_y = self.height - 60
        self.enemies = []
        self.create_enemies()
        self.bullets = []
        self.enemy_bullets = []
        self.score = 0
        self.done = False

    def step(self, actions):
        """Process a step in the game with actions for both agents."""
        agent1_action, agent2_action = actions
        self.handle_actions(agent1_action, agent2_action)

        self.update_bullets()
        self.update_enemies()

        # Check for collision
        self.check_collisions()

        # Return observation, reward, done, and info
        return self.get_observation(), [self.score, self.score], self.done, {}

    def handle_actions(self, agent1_action, agent2_action):
        """Handle the actions for both agents."""
        if agent1_action == 1 and self.agent1_x > 0:  # Move left
            self.agent1_x -= self.agent_speed
        elif agent1_action == 2 and self.agent1_x < self.width - self.agent_width:  # Move right
            self.agent1_x += self.agent_speed

        if agent2_action == 1 and self.agent2_x > 0:  # Move left
            self.agent2_x -= self.agent_speed
        elif agent2_action == 2 and self.agent2_x < self.width - self.agent_width:  # Move right
            self.agent2_x += self.agent_speed

        # Shoot bullets
        if agent1_action == 3:
            self.bullets.append([self.agent1_x + self.agent_width // 2 - self.bullet_width // 2, self.agent1_y])
        if agent2_action == 3:
            self.bullets.append([self.agent2_x + self.agent_width // 2 - self.bullet_width // 2, self.agent2_y])

    def update_bullets(self):
        """Update positions of the bullets."""
        for bullet in self.bullets[:]:
            bullet[1] -= 5  # Move up
            if bullet[1] < 0:
                self.bullets.remove(bullet)  # Remove bullet if it goes off screen

    def update_enemies(self):
        """Update enemy movements."""
        for enemy in self.enemies[:]:
            enemy[1] += self.enemy_speed
            if enemy[1] >= self.height:
                self.done = True  # Game over if any enemy reaches the bottom of the screen
                break

    def check_collisions(self):
        """Check for collisions between bullets and enemies."""
        for bullet in self.bullets[:]:
            for enemy in self.enemies[:]:
                if (bullet[0] >= enemy[0] and bullet[0] <= enemy[0] + self.enemy_size and
                        bullet[1] >= enemy[1] and bullet[1] <= enemy[1] + self.enemy_size):
                    self.enemies.remove(enemy)
                    self.bullets.remove(bullet)
                    self.score += 1
                    break

    def render(self):
        """Render the game on the screen."""
        self.screen.fill(self.bg_color)

        # Draw the agents
        pygame.draw.rect(self.screen, self.agent_color, (self.agent1_x, self.agent1_y, self.agent_width, self.agent_height))
        pygame.draw.rect(self.screen, self.agent_color, (self.agent2_x, self.agent2_y, self.agent_width, self.agent_height))

        # Draw the enemies
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.enemy_color, pygame.Rect(enemy[0], enemy[1], self.enemy_size, self.enemy_size))

        # Draw the bullets
        for bullet in self.bullets:
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


# Main simulation script
def run_simulation():
    # Create the environment and agents
    env = MultiAgentSpaceInvadersEnv()
    agent = DQNAgent(action_space=4, state_space=(env.height, env.width, 3))  # SpaceInvaders screen is 600x400 with 3 color channels

    episodes = 1000
    for e in range(episodes):
        env.reset()
        state = env.get_observation()
        done = False
        while not done:
            action1 = agent.act(state)  # Get action for agent1
            action2 = agent.act(state)  # Get action for agent2
            next_state, reward, done, _ = env.step((action1, action2))
            agent.remember(state, (action1, action2), reward, next_state, done)
            agent.train(batch_size=32)
            state = next_state
            env.render()

        agent.update_epsilon()  # Update epsilon for exploration

    env.close()

# Run the simulation
if __name__ == "__main__":
    run_simulation()
