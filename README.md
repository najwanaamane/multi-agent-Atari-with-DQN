# E2E multi-agent-Atari-with-DQN
Deep reinforcement learning for multi-agent Atari game simulations using DQN, deployed on a Streamlit app hosted on an EC2 instance.



---


This project is based on the work **Playing Atari with Deep Reinforcement Learning** by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller from **DeepMind Technologies**. The original paper demonstrates how deep reinforcement learning (DRL) can be applied to Atari games using a single agent. This project extends the original work by introducing **multi-agent reinforcement learning** (MARL), where multiple agents learn to play the games simultaneously and interact with each other.

In this project, the agents are trained using the **Deep Q-Learning Network (DQN)** algorithm, and we simulate multi-agent setups for four Atari games: **Pong**, **Seaquest**, **Breakout**, and **Space Invaders**. The agents interact in a shared environment, and their actions and rewards influence each other, making the training process more complex and challenging compared to the original single-agent setup.

The project has been **Dockerized** for easy deployment, and the web interface for visualizing the game simulation has been **deployed on an EC2 instance**.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Dockerization & EC2 Deployment](#dockerization-and-ec2-deployment)
- [Project Structure](#project-structure)
- [Training Process](#training-process)
- [Multi-Agent Interaction](#multi-agent-interaction)
- [Results](#results)

## Project Overview   

 ![image](https://github.com/user-attachments/assets/1bfe8503-490f-4a2d-93d9-8090b97e5c8f)   

This project includes the following key components:

1. **Multi-Agent Atari Game Environments**: The agents are trained to play four Atari games:

    - **Seaquest**
      
     ![image](https://github.com/user-attachments/assets/00ddc60a-843e-4453-9f09-747657a51bfa)
 
   - **Pong**
     
     ![image](https://github.com/user-attachments/assets/1bd951f2-9d33-4871-ad73-154afb798229)    
   

   - **Breakout**
     
     ![image](https://github.com/user-attachments/assets/9f2490e7-e082-46d9-b206-f2e15fc7b58b)    

   - **Space Invaders**
     
     ![image](https://github.com/user-attachments/assets/533d002d-3e3f-4de0-b4c7-72d0c9ddbe6d)    

  


3. **DQN Agent**: A Deep Q-Learning Network (DQN) agent that learns the optimal action-policy for the games through experience replay and epsilon-greedy exploration.

4. **Streamlit Interface**: A real-time simulation interface using Streamlit to render the games for visualization.

5. **Dockerization**: The project is containerized using Docker to facilitate easy deployment and scaling.

6. **Deployment on EC2**: The application has been successfully deployed on an EC2 instance to allow for real-time game simulation and visualization in the cloud.



## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/najwanaamane/multi-agent-Atari-with-DQN.git
cd multi-agent-Atari-with-DQN
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure that you have `pygame`, `gym`, and `tensorflow` correctly installed.

## How to Run

 To run the multi-agent simulation with the Streamlit interface, run the following command:

```bash
streamlit run app.py
```

This will open a web interface in your browser where you can watch the agents learn and play multiple atari games.



## Dockerization and EC2 Deployment

This project has been **Dockerized** to make deployment easier and more efficient. Docker allows for creating an isolated environment, making the application portable across different systems. Additionally, the application is deployed on an **EC2 instance** for cloud-based usage.


### **Deploying on EC2 and Accessing Remotely**

To deploy the application on an **AWS EC2 instance** using Docker, follow these steps:

1. **SSH into EC2 Instance**:
   - SSH into your AWS EC2 instance where the application will be deployed. You can do this using the following command (replace `<your-ec2-ip>` with the public IP of your EC2 instance):

     ```bash
     ssh -i <your-key.pem> ubuntu@<your-ec2-ip>
     ```

2. **Dockerize the Application**:
   - Once you’re connected to the EC2 instance, navigate to your project directory and run the following Docker Compose command to build and run your container:

     ```bash
     docker-compose up --build
     ```

   - This will automatically build the Docker image and start the container, making your Streamlit app accessible on port 8501.

3. **Access the Application Remotely**:
   - Once the container is running, you can access the Streamlit interface remotely by opening the **EC2 instance’s public IP address** in your browser at port 8501.

     For example:

     ```bash
     http://<EC2_PUBLIC_IP>:8501
     ```

   - In our case, the application can be accessed via the following URL:

     [https://13.53.190.102:8501/](https://13.53.190.102:8501/)

This setup allows you to run your Streamlit app in a Docker container on an EC2 instance and access it from anywhere using the EC2 instance's public IP.

## Project Structure

```text
multi-agent-atari-dqn/
│
├── app.py                   # Streamlit interface for multi-agent Pong simulation
├── dqn_agent.py             # Deep Q-Learning Agent implementation
├── pong.py       # Pong game simulation (DQN agent for Pong)
├── seaquest.py   # Seaquest game simulation (DQN agent for Seaquest)
├── breakout.py   # Breakout game simulation (DQN agent for Breakout)
├── space_invaders.py # Space Invaders game simulation (DQN agent for Space Invaders)
├── requirements.txt         # List of project dependencies
├── Dockerfile            # Dockerfile for building the Docker image   
├── docker-compose.yml        # with volumes for app persistence and logs
└── README.md                # Project documentation
```

- **app.py**: Contains the Streamlit interface for the multi-agent Pong simulation.
- **dqn_agent.py**: Defines the DQN agent class, including the neural network model and training methods.
- **[game_name].py**: Contains the logic for each of the Atari game simulations (e.g., `pong_simulation.py`, `seaquest_simulation.py`, `breakout_simulation.py`, `space_invaders_simulation.py`).
- **Dockerfile**: A Dockerfile for containerizing the project.   
- **docker-compose.yml**:with volumes to manage the deployment setup efficiently   
- **requirements.txt**: A list of dependencies needed to run the project.

## Training Process

The agents learn through the following process:

1. **Initialization**: A DQN agent is initialized with random weights.
2. **Experience Replay**: The agent stores experiences (state, action, reward, next_state) in memory.
3. **Action Selection**: The agent uses an epsilon-greedy strategy to choose actions. With probability `epsilon`, the agent explores random actions, otherwise it exploits the learned Q-values.
4. **Q-Learning Update**: After each action, the Q-values are updated using the Bellman equation:     
![image](https://github.com/user-attachments/assets/00a0c8d1-8618-46c0-b674-9bd64ecf6565)

   where `r_t` is the reward, `γ` is the discount factor, and `Q(s_{t+1}, a)` is the maximum future reward.
5. **Epsilon Decay**: The exploration rate (`epsilon`) is gradually reduced to encourage more exploitation as training progresses.


---

**Multi-Agent Interaction**

In this multi-agent setup, multiple agents share the environment, and their actions influence each other. This introduces a level of complexity as the agents must not only learn optimal policies for their individual tasks but also account for the behavior of other agents. Below are examples of how the agents interact in each game, along with the associated approaches for cooperation and competition:

---

**Pong**

In the multi-agent Pong simulation, two agents are playing against each other. The agents act as opponents, and their goal is to hit the ball back and forth, avoiding letting the ball pass by them. The agents must learn not only to track the ball but also to anticipate their opponent's movements. The interaction between agents involves competitive play where each agent's success depends on the other agent's actions.

**Competitive Approach:**

Each player aims to score points, penalizing the other with every point scored. It is a zero-sum game where direct competition prevails, with each player trying to outperform the other.

---

**Seaquest**

In Seaquest, two agents cooperate in a shared environment where they navigate the ocean, avoid enemies, and collect divers. The agents must work together to maximize their rewards by avoiding the same threats and collecting as many divers as possible.

**Joint Action Reward Approach:**

Here, agents receive positive rewards only when they act together, such as moving in the same direction. Negative rewards are given for misaligned actions, emphasizing the need for cooperation to achieve the common goal.

---

**Breakout**

In Breakout, two agents control paddles at the bottom of the screen, trying to bounce the ball to break bricks. In this multi-agent setup, each agent controls one paddle, and they must work together to prevent the ball from falling off the screen while breaking bricks.

**Indirect Cooperative Approach:**

Although the agents take turns without explicit coordination, their actions influence the same game state, making them indirectly cooperative. For example, one agent’s failure can make the task harder for the other.

---

**Space Invaders**

In Space Invaders, two agents control separate spaceships to shoot and destroy rows of alien invaders. The agents must avoid enemy fire and work together to clear all invaders.

**Majority Vote for Action Approach:**

The actions of the agents are decided through a majority vote. If the majority chooses a direction, all agents follow that decision. Cooperation is crucial to eliminate enemies and avoid failure, with rewards based on this cooperation to succeed in reaching the objective.

--- 



## Results

The agents' performance improves over time as they learn through trial and error. The results are visualized using the Streamlit interface,in the report section where reward/epsode and cumulative reward plots are displayed,  showing the real-time performance of the agents in the selected game.   

![image](https://github.com/user-attachments/assets/8e341812-abf5-479c-afe7-5c45271f2b91)

---

