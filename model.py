import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# A simple feedforward neural network for Q-learning
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # First hidden layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Output layer (Q-values for each action)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass input through network
        x = F.relu(self.linear1(x))  # activation after first layer
        x = self.linear2(x)          # raw output (Q-values)
        return x

    def save(self, file_name='model.pth'):
        # Save the model’s parameters
        torch.save(self.state_dict(), file_name)


# Q-learning training class using mean squared error and Adam optimizer
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # loss between predicted and target Q-values

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle single sample input (unsqueeze adds batch dimension)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predicted Q-values with current state
        pred = self.model(state)

        # Clone predictions to apply target updates
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Bellman Equation: Q = r + γ * max(Q(s'))
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
