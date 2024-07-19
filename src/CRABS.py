import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



from OUNoise import OUNoise
from replayBuffer import PERBuffer
from models import Policy, Critic, BarrierCertificate, TransitionDynamics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CRABS:
    def __init__(self, state_dim, min_obs, max_obs, action_dim, min_action, max_action, hidden_size_1, hidden_size_2):

        """Define the two Policy Networks"""
        self.policy = Policy(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.policy_target = Policy(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

        """Define the two Critic Networks"""
        self.critic = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        """Define the Barrier Certificate Network"""
        self.barrier_certificate = BarrierCertificate(state_dim, hidden_size_1, hidden_size_2).to(device)

        """Define the dynamics model"""
        self.dynamics = TransitionDynamics(state_dim, action_dim, num_models=10, hidden_size_1=256, hidden_size_2=256, learning_rate=1e-3).to(device)

        """Define the Prioritized Experience Replay Buffer"""
        self.replay_buffer = PERBuffer(max_size=1_000_000)

        """Define the Ornstein-Uhlenbeck Noise for exploration"""
        self.ou_noise = OUNoise(action_dim)
        
        """Define the hyperparameters"""
        self.gamma = 0.99                           # Discount factor
        self.tau = 0.005                            # Soft update parameter
        self.batch_size = 64                        # Batch size for training the networks Critic and Policy Networks
        self.batch_size_dynamics = 1024             # Batch size for training the dynamics model

        """Environment parameters"""
        self.min_action = np.array(min_action)      # Minimum action value
        self.max_action = np.array(max_action)      # Maximum action value

        self.min_obs = torch.tensor(min_obs)
        self.max_obs = torch.tensor(max_obs)
        
        """Miscellaneous variables"""
        self.cnt = 0                                # Counter for updating the target networks
        self.UPDATE_INTERVAL = 50                   # Update the target networks every 50 steps
        self.certificate_cnt = 0                    # Counter for updating the barrier certificate
        self.UPDATE_CERTIFICATE_INTERVAL = 5        # Update the barrier certificate every 50 steps
        self.C = 1000                               # Large Constant for the policy loss
        


    def U(self, state : torch.FloatTensor, action: np.ndarray):
        """
        Given a state and an action, check if the resulting state
        given by a calibrated dynamics model is certified safe by 
        the current barrier certificate.
        """
        num_samples = 5
        # print('State: ', state.shape)
        # print('Action: ', action.shape)
        dist = self.dynamics(state, action)
        # print('Dist: ', dist)
        possible_next_states = dist.sample((num_samples,))
        # print('Possible next states: ', possible_next_states.shape)
        possible_next_states = torch.tanh(possible_next_states)*0.5*(self.max_obs - self.min_obs) + 0.5*(self.max_obs + self.min_obs)
        possible_next_states = possible_next_states.float()
        # print(possible_next_states.shape)
        # possible_next_states = torch.clamp(possible_next_states, self.min_obs, self.max_obs).float()
        # print(possible_next_states.shape)
        certificate_values = self.barrier_certificate(possible_next_states)
        
        """
        Return the max of these values (if max is negative, then
        we can have some confidence all states are safe)
        """
        # certificate_values = certificate_values.cpu().data.numpy()
    
        # if len(certificate_values.shape) == 1:
        #     return -1*np.amax(certificate_values)
        # return -1*np.amax(certificate_values, axis = 1)

        if len(certificate_values.shape) == 1:
            max_certificate_value = torch.max(certificate_values)
        else:
            # print('Certificate values: ', certificate_values, certificate_values.dtype, certificate_values.shape)
            max_certificate_value = torch.max(certificate_values, dim=0).values
            # print('Max certificate value: ', max_certificate_value, max_certificate_value.dtype, max_certificate_value.shape)

        return -1*max_certificate_value


    def select_action(self, state: np.ndarray, exploration_noise = True, debug = False):
        """
        Get the next action given the current state
        """
        self.policy.eval()

        state = torch.Tensor(state).to(device)
        action = self.policy(state).detach().cpu().numpy()
    
        if debug:
            print('State in select action: ', state.shape)
            print('Action in select action: ', action.shape)

        if len(state.shape) == 1:
            if exploration_noise:
                cnt = 0
                while cnt < 100:
                    cnt += 1
                    noisy_action = action + [np.clip(np.random.normal(0,0.25), -1, 1),np.clip(np.random.normal(0,0.25),-1, 1)]
                    # print(noisy_action)
                    # noisy_action = action + np.random.normal(0, 0.1, size=action.shape)
                    # noisy_action = action + 0.1*self.ou_noise.noise()
                    is_action_safe = self.U(state, noisy_action)

                    if is_action_safe <= 0 :
                        if debug:
                            print('State in select action: ', state.shape)
                            print('Noisy Action in select action: ', noisy_action.shape)
                        # print(noisy_action)
                        return noisy_action

                return action
            else:
                return action

        if len(state.shape) == 2:
            if exploration_noise:
                cnt = 0
                mask = np.zeros(state.shape[0], dtype=bool)
                
                while cnt < 100:
                    cnt += 1
                    noisy_action = np.where(mask[:, None], action, action + [np.clip(np.random.normal(0,0.1), -1, 1),np.clip(np.random.normal(0,1),-1, 1)])
                    # noisy_action = np.where(mask[:, None], action, action + 0.1*self.ou_noise.noise())
                    if debug:
                        print('Noisy Action in select action line 147: ', noisy_action.shape)
                    is_action_safe = self.U(state, noisy_action).squeeze(1)
                    if debug:
                        print('is_action_safe dimension: ', is_action_safe.shape)
                    mask = is_action_safe <= 0
                    
                    if mask.all():
                        if debug:
                            print('State in select action: ', state.shape)
                            print('Noisy Action in select action: ', noisy_action.shape)
                        # print(noisy_action)
                        return noisy_action
                if debug:
                    print('State in select action: ', state.shape)
                    print('Noisy Action in select action: ', noisy_action.shape)
                # print(noisy_action)
                return noisy_action
            else:
                return action
    
    def train_dynamics(self):
        """
        Train the dynamics model
        """
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size_dynamics)
        states, actions, rewards, next_states, costs, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)

        rewards = torch.FloatTensor(np.array(rewards)).to(device).view(-1, 1)
        costs = torch.FloatTensor(np.array(costs)).to(device).view(-1, 1)
        dones = torch.FloatTensor(np.array(dones)).to(device).view(-1, 1)

        self.dynamics.train(states, actions, next_states, max_epochs=10, batch_size=64, shuffle=True, verbose=True)


    def train(self):
        torch.autograd.set_detect_anomaly(True)

        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, costs, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        
        rewards = torch.FloatTensor(np.array(rewards)).to(device).view(-1, 1)
        costs = torch.FloatTensor(np.array(costs)).to(device).view(-1, 1)
        dones = torch.FloatTensor(np.array(dones)).to(device).view(-1, 1)

        # print('Training the Dynamics model')
        # self.dynamics.train(states, actions, next_states, max_epochs=10, batch_size=64, shuffle=True, verbose=True)

        # self.certificate_cnt = (self.certificate_cnt + 1) % self.UPDATE_CERTIFICATE_INTERVAL

        # if self.certificate_cnt == 0:
        #     print('Training the Barrier Certificate')
        #     self.barrier_certificate.train(self.policy, self.dynamics, self.U, max_epochs=10)

        self.barrier_certificate.train(self.policy, self.dynamics, self.U, max_epochs=10)


        current_Q = self.critic(states, actions).view(-1)
        
        target__Q = self.critic_target(next_states, self.policy_target(next_states))
        target_Q = rewards + (1 - dones) * self.gamma * target__Q
        target__Q = target_Q.view(-1)

        reward_errors = torch.abs(current_Q - target__Q)
    
        # print('Selecting the samples for training the critic')
        U_values = None
        num_samples = 10
        for i in range(num_samples):
            if U_values is None:
                U_values = self.U(next_states, self.select_action(next_states, exploration_noise = True, debug=False))
            else:
                U_values = U_values + self.U(next_states, self.select_action(next_states, exploration_noise = True, debug=False))
        
        U_values = U_values / num_samples
        take_transition = (U_values <= 0).squeeze(1)
        
        reward_errors = reward_errors[take_transition]
        indices = indices[take_transition]

        errors = reward_errors
        self.replay_buffer.update_priorities(indices, errors.cpu().data.numpy())

        weights = weights[take_transition]
        critic_loss = (torch.FloatTensor(weights) * reward_errors**2).mean()
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        U_values = self.U(states, self.policy(states))
        critic_values = self.critic(states, self.policy(states))

        policy_loss = torch.where(U_values <= 0, -critic_values, -self.C - U_values).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        # MALA adversarial Training for Policy is not implemented yet

        






        self.cnt = (self.cnt + 1) % self.UPDATE_INTERVAL

        if self.cnt == 0:
            self.update_target_network(self.critic, self.critic_target)
            self.update_target_network(self.policy, self.policy_target)


    
    def update_target_network(self, model, target_model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)





    


