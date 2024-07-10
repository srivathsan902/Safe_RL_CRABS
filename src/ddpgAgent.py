from models import Actor, Critic, BarrierCertificate, DynamicsModel
from replayBuffer import ReplayBuffer, PERBuffer
from OUNoise import OUNoise
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from torch.utils.data import TensorDataset

def U(state, action, dynamics_model, barrier_certificate):
        num_samples = 100
        state = torch.Tensor(state).to(device)
        state = torch.clamp(state, -1000, 1000)
        dist = dynamics_model(state, action)
        # dist = torch.distributions.Normal(mean, std)
        samples = dist.sample((num_samples,))
        certificate_values = barrier_certificate(samples)
        # Return the max of these values
        return -1*certificate_values.max()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, min_action, hidden_size_1 = 64, hidden_size_2 = 128, priority_replay = True):
        
        self.max_action = np.array(max_action)
        self.min_action = np.array(min_action)
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.actor = Actor(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.barrier_certificate = BarrierCertificate(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.dynamics_model = DynamicsModel(state_dim, action_dim, num_models=10, hidden_size_1=256, hidden_size_2=256, learning_rate=1e-3).to(device)

        self.priority_replay = priority_replay

        if self.priority_replay:
            self.replay_buffer = PERBuffer(max_size=1_000_000)
        else:
            self.replay_buffer = ReplayBuffer(max_size=1_000_000)

        self.gamma = 0.99
        self.tau = 0.005

        self.ou_noise = OUNoise(action_dim)

        self.cnt = 0
        self.UPDATE_INTERVAL = 20

    def select_action(self, state, noise_enabled=True):
        self.actor.eval()
        if len(state.shape) == 1:
            state = torch.FloatTensor(state.reshape(28)).to(device).float()
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise_enabled:
            for _ in range(5):
                noisy_action = action + self.ou_noise.noise()
                if U(state, noisy_action, self.dynamics_model, self.barrier_certificate) < 0:
                    action = noisy_action
                    break
        # Scale action to [min_action, max_action]
        action = self.min_action + (action + 1.0) * 0.5 * (self.max_action - self.min_action)
        return action.clip(self.min_action, self.max_action)

    def train(self, batch_size):
        torch.autograd.set_detect_anomaly(True)
        if self.priority_replay:
            transitions, indices, weights = self.replay_buffer.sample(batch_size)

            # Unzip transitions into separate variables
            states, actions, rewards, next_states, costs, dones = zip(*transitions)
        else:
            states, actions, rewards, next_states, costs, dones = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.array(states)).to(device)
        action = torch.FloatTensor(np.array(actions)).to(device)
        next_state = torch.FloatTensor(np.array(next_states)).to(device)

        reward = torch.FloatTensor(np.array(rewards)).to(device).view(-1, 1)
        done = torch.FloatTensor(np.array(dones)).to(device).view(-1, 1)
        cost = torch.FloatTensor(np.array(costs)).to(device).view(-1, 1)

        # Train the Dyanmics Model
        dataset = TensorDataset(state, action, next_state)
        self.dynamics_model.train(dataset, batch_size = batch_size, num_epochs = 1)

        # Train the Barrier Certificate
        self.barrier_certificate.train(self.actor, self.dynamics_model, train_size = 1)

        # Train the Critic Network
        # Compute the target Q value for reward critic
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.gamma * target_Q

        target_Q = target_Q.view(-1)

        # Get current Q estimate for reward critic
        current_Q = self.critic(state, action).view(-1)

        # Compute TD errors for prioritization
        reward_errors = torch.abs(current_Q - target_Q).cpu().data.numpy()

        take_transition = np.zeros(batch_size, dtype=bool)
        # Take the states only when mean over actions at next_state gives -ve U value
        # next_state = next_state.cpu().data.numpy()

        # print('Reached Here')
        # Take average over actions at next_state
        all_U_values = None
        # print('All U values: ', all_U_values.shape)
        cnt = 1
        for nxt_state in next_state:
            # print('Next state: ', cnt)
            cnt += 1
            U_val_tot = torch.zeros(1)
            for i in range(1):
                # print('Next state ',type(nxt_state), nxt_state.shape)
                next_action = self.select_action(nxt_state, noise_enabled=True)
                # print('Action Chosen')
                U_value = U(nxt_state, next_action, self.dynamics_model, self.barrier_certificate).detach()
                # print('U value computed')
                # print('U value: ', U_value.shape)
                U_val_tot = U_val_tot + U_value
            if all_U_values is None:
                all_U_values = U_val_tot
            else:
                all_U_values = torch.cat((all_U_values, U_val_tot), dim=0)
                
        # print('U values: ', all_U_values.shape)
        take_transition = all_U_values < 0
        

        # for i in range(next_state.shape[0]):
        #     U_values = []
        #     for j in range(20):
        #         print(j)
        #         next_action = self.select_action(next_state[i], noise_enabled=True)
        #         U_values.append(U(next_state[i], next_action, self.dynamics_model, self.barrier_certificate).detach())
        #     if np.mean(U_values) < 0:
        #         # Consider this transition for loss computation
        #         take_transition[i] = True
        
        reward_errors = reward_errors[take_transition]

        errors = reward_errors

        if self.priority_replay:
            # Update priorities in PER buffer
            self.replay_buffer.update_priorities(indices, errors)

            # Compute critic loss with importance sampling weights
            critic_loss = (torch.FloatTensor(weights).to(device) * nn.MSELoss()(current_Q, target_Q)).mean()
        else:
            critic_loss = nn.MSELoss()(current_Q, target_Q).mean()


        # Optimize the reward critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train the Actor Network
        barrier_values = self.barrier_certificate(state)
        critic_values = self.critic(state, self.actor(state))
        actor_loss = torch.where(barrier_values <= 0 , -1*critic_values, -1e6 - barrier_values).mean()
        # actor_loss = -critic_values.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models every UPDATE_INTERVAL steps to stabilize training

        self.cnt = (self.cnt + 1) % self.UPDATE_INTERVAL

        if self.cnt == 0:
            self.update_target_network(self.critic, self.critic_target)
            self.update_target_network(self.actor, self.actor_target)


    def update_target_network(self, network, target_network):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        paths = {
            'actor': os.path.join(dir_path, f'actor.pth'),
            'critic': os.path.join(dir_path, f'critic.pth'),
            'actor_target': os.path.join(dir_path, f'actor_target.pth'),
            'critic_target': os.path.join(dir_path, f'critic_target.pth'),
            'replay_buffer': os.path.join(dir_path, f'replay_buffer.pickle')
        }

        for element, path in paths.items():
            getattr(self, element).save(path)


    def load(self, dir_path):
        
        if not os.path.exists(dir_path):
            raise ValueError(f'Path {dir_path} does not exist')
        
        paths = {
            'actor': os.path.join(dir_path, f'actor.pth'),
            'critic': os.path.join(dir_path, f'critic.pth'),
            'actor_target': os.path.join(dir_path, f'actor_target.pth'),
            'critic_target': os.path.join(dir_path, f'critic_target.pth'),
            'replay_buffer': os.path.join(dir_path, f'replay_buffer.pickle')
        }


        for element, path in paths.items():
            if os.path.exists(path):
                getattr(self, element).load(path)

        
        