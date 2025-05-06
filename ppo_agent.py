
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Linear_QNet
from game import Point, Direction


class PPOAgent:
    def __init__(self, input_dim=11, action_dim=3, gamma=0.99, lr=0.0003, clip_epsilon=0.2):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.policy = Linear_QNet(input_dim, 256, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.old_log_probs = []
        self.states = []
        self.actions = []
        self.rewards = []

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)


        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def store_transition(self, state, action, log_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(log_prob)
        self.rewards.append(reward)

    def compute_returns(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float)

    def update(self):
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.stack(self.old_log_probs).detach()
        returns = self.compute_returns()

        logits = self.policy(states)
        new_probs = torch.softmax(logits, dim=-1)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        ratios = torch.exp(new_log_probs - old_log_probs)
        advantages = returns - returns.mean()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.states, self.actions, self.rewards, self.old_log_probs = [], [], [], []
