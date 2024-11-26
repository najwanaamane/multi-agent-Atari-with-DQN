import importlib
import os
import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import streamlit as st

np_bool = getattr(np, 'bool', np.bool_)

image_folder = "images/"

simulations = {
    "Pong Simulation": {
        "script": "pong",
        "image": os.path.join(image_folder, "pong.png"),
        "description": """
            **Pong Simulation**:
            The two agents take turns controlling paddles to hit a ball back and forth. The goal is to maximize their scores, and they will learn how to position their paddles to return the ball successfully. Cooperative or competitive depending on the setup (e.g., can be cooperative if they are both trying to avoid losing).
        """
    },
    "Breakout": {
        "script": "breakout",
        "image": os.path.join(image_folder, "breakout.png"),
        "description": """
            **Breakout Simulation**:
            The agents control paddles to break bricks. The agents need to learn to aim and position the paddle optimally. Here, the behavior could be competitive if the agents try to break different bricks or cooperative if they share a common goal of breaking all bricks.
        """
    },
    "Seaquest": {
        "script": "seaquest",
        "image": os.path.join(image_folder, "seaquest.png"),
        "description": """
            **Seaquest Simulation**:
            The agents must collect items (e.g., fish) while avoiding enemies (e.g., sharks). The competitive aspect may come into play if the agents try to collect items faster or outpace each other in capturing more items.
        """
    },
    "Space Invaders": {
        "script": "spaceinvaders",
        "image": os.path.join(image_folder, "spaceinvaders.png"),
        "description": """
            **Space Invaders**:
            The agents control a ship to destroy waves of invaders. The competitive behavior may arise when agents try to outperform each other by achieving a higher score.
        """
    },
}

# Streamlit App
st.title("Multi-Agent Atari Game Simulations 🎮")
st.write("""          
The DQN algorithm, as described in the paper **"Playing Atari with Deep Reinforcement Learning"** from **DeepMind Technologies**, is used for training agents to play Atari games by learning directly from pixel inputs.    
This project extends the original work by introducing multi-agent reinforcement learning (MARL), where multiple agents learn to play the games simultaneously and interact with each other.

""")

# Sidebar for selecting report or simulation
report_option = st.sidebar.selectbox("Select Option", ["Simulation", "Report"])

if report_option == "Simulation":
    # Create a single row for the simulations
    cols = st.columns(len(simulations))

    # Add images and clickable buttons for each simulation in one line
    selected_simulation = None
    for idx, (simulation, details) in enumerate(simulations.items()):
        with cols[idx]:  # Each column corresponds to one simulation
            # Show the simulation image with reduced size
            if os.path.exists(details['image']):
                st.image(details['image'], width=150)
            else:
                st.image("https://via.placeholder.com/150", width=150)  # Placeholder for missing images

            # Create a button for each simulation
            if st.button(simulation):
                selected_simulation = details['script']
                st.write(f"You selected: **{simulation}**")
                st.write("Running simulation...")

    # Run the selected simulation directly inside the app
    if selected_simulation:
        try:
            # Import the selected script dynamically
            simulation_module = importlib.import_module(selected_simulation)

            # Create the environment dynamically based on the selected simulation
            if selected_simulation == "pong":
                env = gym.make('Pong-v0', render_mode='rgb_array')
            elif selected_simulation == "breakout":
                env = gym.make('Breakout-v0', render_mode='rgb_array')
            elif selected_simulation == "seaquest":
                env = gym.make('Seaquest-v0', render_mode='rgb_array')
            elif selected_simulation == "spaceinvaders":
                env = gym.make('SpaceInvaders-v0', render_mode='rgb_array')

            # Initialize the state
            state, _ = env.reset()  # Reset the environment to start
            state = np.reshape(state, [1, *env.observation_space.shape])  # Reshape state to match the CNN input shape

            # Check if the agent has a constructor that accepts action_space and state_space
            if hasattr(simulation_module, 'DQNAgent'):
                # Create the DQN agent
                agent = simulation_module.DQNAgent(action_space=env.action_space.n, state_space=state.shape[1:])
            else:
                raise AttributeError(f"Agent class 'DQNAgent' not found in {selected_simulation} module")

            # Create a placeholder for the image
            image_placeholder = st.empty()  # Create an empty placeholder for updating the image

            # Create lists to store rewards, cumulative rewards, and actions
            episode_rewards = []
            cumulative_rewards = []

            # Display a spinner while the simulation runs
            with st.spinner(f"Running {selected_simulation}..."):
                total_reward = 0
                for e in range(10):  # Set the number of episodes
                    state, _ = env.reset()
                    state = np.reshape(state, [1, *env.observation_space.shape])
                    done = False
                    episode_reward = 0  # Reset episode reward for each new episode
                    while not done:
                        action = agent.act(state)
                        next_state, reward, done, truncated, info = env.step(action)
                        next_state = np.reshape(next_state, [1, *env.observation_space.shape])
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        episode_reward += reward

                        # Capture the frame from the environment
                        frame = env.render()  # Capture the frame from the environment

                        # Update the placeholder with the new frame
                        image_placeholder.image(frame, channels="RGB", use_container_width=True)

                        if done:
                            episode_rewards.append(episode_reward)
                            cumulative_rewards.append(np.sum(episode_rewards))  # Cumulative reward
                            st.write(f"Episode finished. Total Reward: {episode_reward}")
                            break

                    # Train the agent after each episode
                    agent.train(batch_size=32)

            # Store the simulation results in session state
            st.session_state.episode_rewards = episode_rewards
            st.session_state.cumulative_rewards = cumulative_rewards
            st.session_state.selected_simulation_image = details['image']  # Store the selected simulation's image

        except Exception as e:
            st.error(f"An error occurred: {e}")

# If the user selects "Report", display the report with plots
if report_option == "Report":
    st.sidebar.write("### Simulation Report")
    st.write("""This section provides detailed reports for the simulation results.""")

    # Sidebar to select which simulation's report to view
    selected_simulation_report = st.sidebar.selectbox("Select Simulation for Report", list(simulations.keys()))

    # Show details of the selected simulation
    if selected_simulation_report:
        st.write(f"### {selected_simulation_report} Report")
        st.write(simulations[selected_simulation_report]["description"])

        # Show the simulation image with reduced size
        image_path = simulations[selected_simulation_report]["image"]
        if os.path.exists(image_path):
            st.image(image_path, width=150)
        else:
            st.image("https://via.placeholder.com/150", width=150)  # Placeholder for missing images

        # Retrieve the data from session state and display the corresponding plots
        if 'episode_rewards' in st.session_state and 'cumulative_rewards' in st.session_state:
            # Display the corresponding plot only for the selected simulation
            if selected_simulation_report == "Pong Simulation":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Cumulative Reward Plot
                ax1.plot(st.session_state.cumulative_rewards, label="Cumulative Reward", color='blue')
                ax1.set_xlabel("Episodes")
                ax1.set_ylabel("Cumulative Reward")
                ax1.set_title("Pong - Cumulative Reward per Episode")
                ax1.legend()

                # Reward per Episode Plot
                ax2.plot(st.session_state.episode_rewards, label="Reward per Episode", color='red')
                ax2.set_xlabel("Episodes")
                ax2.set_ylabel("Reward")
                ax2.set_title("Pong - Reward per Episode")
                ax2.legend()

                st.pyplot(fig)
            elif selected_simulation_report == "Breakout":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Cumulative Reward Plot
                ax1.plot(st.session_state.cumulative_rewards, label="Cumulative Reward", color='blue')
                ax1.set_xlabel("Episodes")
                ax1.set_ylabel("Cumulative Reward")
                ax1.set_title("Breakout - Cumulative Reward per Episode")
                ax1.legend()

                # Reward per Episode Plot
                ax2.plot(st.session_state.episode_rewards, label="Reward per Episode", color='red')
                ax2.set_xlabel("Episodes")
                ax2.set_ylabel("Reward")
                ax2.set_title("Breakout - Reward per Episode")
                ax2.legend()

                st.pyplot(fig)
            elif selected_simulation_report == "Seaquest":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Cumulative Reward Plot
                ax1.plot(st.session_state.cumulative_rewards, label="Cumulative Reward", color='green')
                ax1.set_xlabel("Episodes")
                ax1.set_ylabel("Cumulative Reward")
                ax1.set_title("Seaquest - Cumulative Reward per Episode")
                ax1.legend()

                # Reward per Episode Plot
                ax2.plot(st.session_state.episode_rewards, label="Reward per Episode", color='purple')
                ax2.set_xlabel("Episodes")
                ax2.set_ylabel("Reward")
                ax2.set_title("Seaquest - Reward per Episode")
                ax2.legend()

                st.pyplot(fig)
            elif selected_simulation_report == "Space Invaders":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Cumulative Reward Plot
                ax1.plot(st.session_state.cumulative_rewards, label="Cumulative Reward", color='yellow')
                ax1.set_xlabel("Episodes")
                ax1.set_ylabel("Cumulative Reward")
                ax1.set_title("Space Invaders - Cumulative Reward per Episode")
                ax1.legend()

                # Reward per Episode Plot
                ax2.plot(st.session_state.episode_rewards, label="Reward per Episode", color='orange')
                ax2.set_xlabel("Episodes")
                ax2.set_ylabel("Reward")
                ax2.set_title("Space Invaders - Reward per Episode")
                ax2.legend()

                st.pyplot(fig)
