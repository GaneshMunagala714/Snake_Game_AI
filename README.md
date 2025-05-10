# Snake Reinforcement Learning Project

This repository contains implementations of two reinforcement learning algorithms (Deep Q-Network and Proximal Policy Optimization) applied to the classic Snake game. The project provides a comparative analysis of these approaches for training an agent to play Snake.



## Project Overview

The Snake game is a classic problem for reinforcement learning, requiring agents to learn complex behaviors like path planning, obstacle avoidance, and long-term reward maximization. This project implements:

1. **Deep Q-Network (DQN)** - A value-based reinforcement learning approach
2. **Proximal Policy Optimization (PPO)** - A policy gradient method with stability improvements
3. **Comparative analysis framework** - For evaluating algorithm performance

## Key Features

- Custom Snake game environment with Pygame
- Complete implementations of DQN and PPO algorithms
- Comprehensive evaluation and visualization tools
- Detailed documentation and analysis

## Installation Requirements

```bash
# Clone this repository
git clone https://github.com/GaneshMunagala714/Snake_Game_AI.git
cd snake-reinforcement-learning

# Install required dependencies
pip install -r requirements.txt
```

Required dependencies:
- Python 3.7+
- PyTorch
- Pygame
- NumPy
- Matplotlib
- Pandas

## Project Structure

```
.
├── agent.py                     # DQN agent implementation
├── compare_agents.py            # Performance comparison utilities
├── comparison.png               # Performance visualization
├── dqn_training_log.csv         # DQN training metrics
├── game.py                      # Snake game environment
├── manual.py                    # Manual control mode
├── model.py                     # Neural network architecture
├── model.pth                    # Saved model weights
├── ppo_agent.py                 # PPO agent implementation
├── ppo_agent_improved.py        # Enhanced PPO implementation
├── ppo_train.py                 # PPO training script
├── ppo_train_improved.py        # Enhanced PPO training
├── ppo_training_log.csv         # PPO training metrics
├── ppo_vs_dqn_comparison.png    # Algorithm comparison visualization
├── README.md                    # Project documentation
└── train.py                     # DQN training script
```

## Usage Instructions

### Training a DQN Agent

```bash
# Start training a DQN agent from scratch
python train.py

# The model will save automatically as model.pth
# Training progress is displayed in real-time
```

### Training a PPO Agent

```bash
# Start training a PPO agent from scratch
python ppo_train.py

# For improved PPO implementation
python ppo_train_improved.py
```

### Comparing Agent Performance

```bash
# After training both agents, run the comparison script
python compare_agents.py

# This will generate comparison visualizations
```

### Playing Manually

```bash
# To play the game manually
python manual.py
```

## Implementation Details

### Game Environment (`game.py`)

The Snake game environment is implemented with the following key features:
- 640x480 pixel game window
- Grid-based movement (20px blocks)
- Reward structure:
  - +10 for eating food
  - -10 for collision (death)
  - -0.1 small penalty per step (encourages efficient paths)
- Game over conditions:
  - Snake collides with wall
  - Snake collides with itself
  - Frame limit exceeded (prevents infinite loops)

### DQN Implementation (`agent.py`, `model.py`, `train.py`)

The DQN agent uses:
- 11-dimensional state space (danger detection, direction, food location)
- 3 possible actions (straight, right, left turns)
- Experience replay buffer (100,000 transitions)
- Epsilon-greedy exploration strategy
- Network architecture:
  - Input layer: 11 neurons
  - Hidden layer: 256 neurons with ReLU activation
  - Output layer: 3 neurons (Q-values for each action)
- Hyperparameters:
  - Learning rate: 0.001
  - Discount factor (gamma): 0.9
  - Batch size: 1000

### PPO Implementation (`ppo_agent.py`, `ppo_train.py`)

The PPO agent uses:
- Same state representation as DQN
- Policy network with similar architecture
- Advantage estimation for policy updates
- Clipped objective function for stable learning
- Hyperparameters:
  - Learning rate: 0.0003
  - Discount factor (gamma): 0.99
  - Clipping epsilon: 0.2

## Performance Analysis

The repository includes tools for analyzing agent performance:
- Score progression over training episodes
- Average rewards
- Learning efficiency
- Visual comparisons between algorithms

Key findings from our experiments:
- DQN tends to learn faster in early stages
- PPO shows more stable performance in later stages
- Both algorithms successfully learn effective snake-playing strategies

## Educational Value

This project serves as an educational resource for:
- Understanding reinforcement learning implementation
- Comparing different RL approaches on the same problem
- Visualizing learning progress and algorithm performance
- Implementing deep learning models in PyTorch

## Future Work

Potential extensions to this project:
- Additional RL algorithms (A2C, SAC, etc.)
- Hyperparameter optimization studies
- Curriculum learning approaches
- Visual feature extraction using convolutional networks
- Multi-agent competitive scenarios

## License

This project is available under the MIT License.

## Citation

If you use this code in your research, please cite:

```
@misc{snake-reinforcement-learning,
  author = {Your Name},
  title = {Snake Reinforcement Learning: DQN vs PPO},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/GaneshMunagala714/Snake_Game_AI}
}
```

## Acknowledgments

- This project was developed as part of a course on artificial intelligence
- Special thanks to the PyTorch and Pygame development teams
