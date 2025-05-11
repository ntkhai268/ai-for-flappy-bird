import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_size: int = 64):
#         super(DQN, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_size = hidden_size
#         self.output_dim = output_dim

#         self.fc1 = nn.Linear(input_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_dim)

#         # Khởi tạo trọng số
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.uniform_(m.weight, -0.01, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size: int = 5):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing_steps=10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.beta_increment = (1.0 - beta) / beta_annealing_steps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
        self.epsilon = 1e-6

    def append(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = PrioritizedReplayBuffer(capacity=50000, alpha=0.6, beta=0.4)
        self.gamma = 0.99
        self.epsilon = 0.1  # Giảm từ 1.0
        self.epsilon_min = 0.01  # Giảm từ 0.1
        self.epsilon_decay = 0.995  # Giảm từ 0.9998
        self.learning_rate = 0.001
        self.update_targetnn_rate = 100  # Giảm từ 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.train_steps = 0

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        samples, indices, weights = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor([batch[0] for batch in samples]).to(self.device)
        action_batch = torch.LongTensor([batch[1] for batch in samples]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor([batch[2] for batch in samples]).to(self.device)
        next_state_batch = torch.FloatTensor([batch[3] for batch in samples]).to(self.device)
        terminal_batch = torch.FloatTensor([float(batch[4]) for batch in samples]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, indices, weights

    def train_main_network(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0.0

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, indices, weights = self.get_batch_from_buffer(batch_size)
        q_values = self.main_network(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q = reward_batch + self.gamma * max_next_q_values * (1 - terminal_batch)
            td_errors = q_values - target_q

        loss = (weights * self.loss_fn(q_values, target_q)).mean() 
        loss_value = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.update_targetnn_rate == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        self.replay_buffer.update_priorities(indices, td_errors.abs().cpu().numpy())
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss_value

    def make_decision(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.main_network(state_tensor)
        return torch.argmax(q_values).item()

    def save_model(self, filename):
        torch.save(self.main_network.state_dict(), filename)
        print(f"Model saved to {filename}")