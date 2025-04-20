import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning rate

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # controls exploration vs. exploitation
        self.gamma = 0.9  # discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # experience replay buffer
        self.model = Linear_QNet(11, 256, 3)     # input=11 features, output=3 actions
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Get current position of snake's head
        head = game.snake[0]
        # Define surrounding points
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Construct state vector: 11 binary features
        state = [
            # Danger straight
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # Danger left
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Save a single game step to memory
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Train on a batch from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # random subset
        else:
            mini_sample = self.memory  # use full memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train on most recent action
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Decide action based on ε-greedy strategy
        self.epsilon = 80 - self.n_games  # lower ε with more games
        final_move = [0, 0, 0]  # [straight, right, left]

        if random.randint(0, 200) < self.epsilon:
            # Exploration: random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation: use model prediction
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
