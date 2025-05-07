# Snake_Game_AI


#  Snake Game AI using PPO (Proximal Policy Optimization)

This project implements a Reinforcement Learning (RL) agent that learns to play the classic Snake game using the PPO (Proximal Policy Optimization) algorithm. The environment is built using Pygame, and the model is trained using PyTorch.

##  Demo
[![Watch the video](https://img.youtube.com/vi/YOUR_VIDEO_LINK_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_LINK_HERE)

## Features

- Deep Reinforcement Learning using PPO
- Game environment created from scratch in Pygame
- Custom reward shaping for better learning performance
- Evaluation and logging of agent performance
- Support for saving and loading models

##  Algorithm

We use **Proximal Policy Optimization (PPO)** — a policy gradient method that improves training stability and sample efficiency.

## Project Structure

```
├── game.py             # Snake game environment using Pygame
├── ppo_agent.py        # PPO agent definition (Actor-Critic model)
├── ppo_train.py        # Training script using PPO
├── evaluation.py       # Evaluate trained model and generate performance stats
├── assets/             # Optional: assets for GUI (if used)
├── models/             # Saved model weights
└── README.md           # Project overview and instructions
```

## Requirements

- Python 3.7+
- PyTorch
- Pygame
- NumPy
- Matplotlib (optional, for plotting)

Install all dependencies:

```bash
pip install -r requirements.txt
```

##  How to Run

### Train the Model

```bash
python ppo_train.py
```

### Evaluate the Model

```bash
python evaluation.py
```

### Play the Game (Manual Control - Optional)

```bash
python game.py
```

##  Evaluation Metrics

- Total reward per episode
- Number of apples eaten
- Survival time (steps)
- Average reward over test episodes

##  Screenshots

![Game Screenshot](assets/screenshot.png)

## Team Members

| Name   | Role           |
|--------|----------------|
| Ganesh | AI Engineer    |
| Akash  | Data Scientist |
| Naveen | Data Analyst   |

## References

- [PPO Paper - OpenAI](https://arxiv.org/abs/1707.06347)
- Patrick Loeber's [RL Snake Game Tutorial](https://github.com/patrickloeber/snake-ai-pytorch)

## License

This project is licensed under the MIT License.
