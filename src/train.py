import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Initialize environment
env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

# Replay Buffer Implementation
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)

# ProjectAgent class with enhancements
class ProjectAgent:
    def __init__(self):
        self.writer = SummaryWriter(log_dir="./logs")

    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        self.path = path + "/model_DQN.pt"
        torch.save(self.model.state_dict(), self.path)

    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() + "/model_DQN.pt"
        self.model = self.myDQN({}, device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()

    def myDQN(self, config, device):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = 256

        DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        ).to(device)

        return DQN

    def train(self):
        config = {
            'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.98,
            'buffer_size': 10000,
            'epsilon_min': 0.02,
            'epsilon_max': 1.0,
            'epsilon_decay_period': 30000,
            'epsilon_delay_decay': 120,
            'batch_size': 128,
            'gradient_steps': 3,
            'update_target_strategy': 'replace',
            'update_target_freq': 1000,
            'criterion': torch.nn.SmoothL1Loss()
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        self.model = self.myDQN(config, device)
        self.target_model = deepcopy(self.model).to(device)

        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']

        epsilon_max = config['epsilon_max']
        epsilon_min = config['epsilon_min']
        epsilon_decay_period = config['epsilon_decay_period']
        epsilon_delay_decay = config['epsilon_delay_decay']
        epsilon_step = (epsilon_max - epsilon_min) / epsilon_decay_period

        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        episode_returns = []
        max_episodes = 400
        epsilon = epsilon_max
        step = 0
        best_val_score = -np.inf

        state, _ = env.reset()
        for episode in range(max_episodes):
            episode_return = 0
            for t in range(200):
                if step > epsilon_delay_decay:
                    epsilon = max(epsilon_min, epsilon - epsilon_step)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.act(state)

                next_state, reward, done, _, _ = env.step(action)
                episode_return += reward
                reward /= 1e4  # Normalized reward

                self.memory.append(state, action, reward, next_state, done)
                state = next_state

                for _ in range(config['gradient_steps']):
                    self.gradient_step()

                if step % config['update_target_freq'] == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                step += 1
                if done:
                    break

            val_score = evaluate_HIV(self, nb_episode=1)
            print(f"Episode {episode + 1}/{max_episodes} | "
                  f"Return: {episode_return:.2e} | "
                  f"Validation Score: {val_score:.2e} | "
                  f"Epsilon: {epsilon:.4f}")

            self.writer.add_scalar("Validation/Score", val_score, episode)
            self.writer.add_scalar("Return/Episode", episode_return, episode)

            if val_score > best_val_score:
                best_val_score = val_score
                self.save(os.getcwd())
            episode_returns.append(episode_return)
            state, _ = env.reset()

        self.writer.close()
        return episode_returns

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Main execution starts here
if __name__ == "__main__":
    agent = ProjectAgent()
    returns = agent.train()
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Performance')
    plt.show()
    print("Training complete. Best model saved.")
