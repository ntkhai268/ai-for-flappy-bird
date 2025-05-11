import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Neural network cho DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5)
        # self.fc2 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(5, output_dim)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        return self.fc2(x)

# class DQN(nn.Module):
#     def __init__(self, input_dim=5, hidden_dim=32, output_dim=2):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)     # 5 -> 32
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)    # 32 -> 32
#         self.fc3 = nn.Linear(hidden_dim, output_dim)    # 32 -> 2
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))  
#         x = torch.relu(self.fc2(x))   
#         x = self.fc3(x)          
#         return x


# Agent sử dụng PyTorch
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.1  # Giảm từ 1.0
        self.epsilon_min = 0.02  # Giảm từ 0.1
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
        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch = torch.FloatTensor([batch[0] for batch in exp_batch]).to(self.device)
        action_batch = torch.LongTensor([batch[1] for batch in exp_batch]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor([batch[2] for batch in exp_batch]).to(self.device)
        next_state_batch = torch.FloatTensor([batch[3] for batch in exp_batch]).to(self.device)
        terminal_batch = torch.FloatTensor([float(batch[4]) for batch in exp_batch]).to(self.device)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def train_main_network(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        # Tính Q(s, a)
        q_values = self.main_network(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q = reward_batch + self.gamma * max_next_q_values * (1 - terminal_batch)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.update_targetnn_rate == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

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