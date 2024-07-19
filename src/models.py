import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from safeStates import isStateSafe
from utils import print_args_decorator

'''
What are the models required?
1. Actor to learn the policy
2. Barrier Certificate to learn the safety constraint
3. Dynamics Model to learn the transition dynamics

What is the flow?
1. Assuming a initial safe policy, the agent interacts with the environment
   and collects data.
2. Train the dynamics model on the collected data.
3. Sample possible states from current state using the dynamics model.
4. Check if the sampled states are safe using the barrier certificate.
5. If the action is leading to a safe state


'''
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def save(self, path):
        '''
        Save the model, parameters, and optimizer state
        '''
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_size_1': self.hidden_size_1,
            'hidden_size_2': self.hidden_size_2
        }, path)

    def load(self, path):
        '''
        Load the model, parameters, and optimizer state
        '''
        checkpoint = torch.load(path)
        self.hidden_size_1 = checkpoint.get('hidden_size_1', 64)
        self.hidden_size_2 = checkpoint.get('hidden_size_2', 128)
        self.load_state_dict(checkpoint.get('model_state_dict', checkpoint))


class Policy(BaseModel):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(Policy, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, action_dim)

        # Initialize weights and biases of layer3 to output [0, 1] initially
        nn.init.zeros_(self.layer3.weight)
        nn.init.constant_(self.layer3.bias, 0)

    # @print_args_decorator
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        # print(x)
        return x

class Critic(BaseModel):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(Critic, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

class ProbabilisticTransitionDynamics(BaseModel):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2, learning_rate = 0.001):
        super(ProbabilisticTransitionDynamics, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)

        self.mean = nn.Linear(hidden_size_2, state_dim)
        self.log_std = nn.Linear(hidden_size_2, state_dim)

        self.max_log_std = nn.Parameter(torch.full((1, state_dim), 5.0, dtype=torch.float).squeeze(0), requires_grad=True)
        self.min_log_std = nn.Parameter(torch.full((1, state_dim), -5.0, dtype=torch.float).squeeze(0), requires_grad=True)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def set_env_specs(self, env_specs: dict):
        """
        env_specs: dict
        Keys:
        - initial_state: np.ndarray
        - state_space_low: np.ndarray
        - state_space_high: np.ndarray
        """
        self.env_specs = env_specs
        self.env_specs['initial_state'] = torch.FloatTensor(self.env_specs['initial_state'])
        self.env_specs['state_space_high'] = torch.tensor(self.env_specs['state_space_high']).float()
        self.env_specs['state_space_low'] = torch.tensor(self.env_specs['state_space_low']).float()

    # @print_args_decorator
    def forward(self, state : torch.FloatTensor, action : np.ndarray):
        dim = 1
        if len(state.shape) == 1:
            dim = 0
        x = torch.cat([state, torch.FloatTensor(action)], dim)
        x = torch.relu(self.layer_1(x))
        x = torch.tanh(self.layer_2(x))

        mean = self.mean(x)
        mean = torch.tanh(mean)*(self.env_specs['state_space_high'] - self.env_specs['state_space_low'])/4 + (self.env_specs['state_space_high'] + self.env_specs['state_space_low'])/2
        log_std = self.log_std(x)
        # print('max_log_std: ', self.max_log_std, 'min_log_std: ', self.min_log_std)
        # print('log_std: ', log_std)
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)

        std = torch.exp(log_std)
        std = std.clamp(1e-2, 10)

        # print(std)
        return torch.distributions.Normal(mean, std)


class BarrierCertificate(BaseModel):
    def __init__(self, state_dim, hidden_size_1, hidden_size_2):
        super(BarrierCertificate, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)


        self.m = 500                        # Number of samples to find the adversarial state
        self.prev_network = None
        self.lambd = 0.01                   # Regularization parameter
        self.mala_temperature = 0.1         # Temperature for the Metropolis Adjusted Langevin Algorithm
        self.mala_step_size = 0.01          # Step size for the Metropolis Adjusted Langevin Algorithm
        self.pos = None

    def set_env_specs(self, env_specs: dict):
        """
        env_specs: dict
        Keys:
        - initial_state: np.ndarray
        - state_space_low: np.ndarray
        - state_space_high: np.ndarray
        """
        self.env_specs = env_specs
        self.env_specs['initial_state'] = torch.FloatTensor(self.env_specs['initial_state'])
        self.env_specs['state_space_high'] = torch.tensor(self.env_specs['state_space_high']).float()
        self.env_specs['state_space_low'] = torch.tensor(self.env_specs['state_space_low']).float()
    
    def set_dynamics_model(self, dynamics_model):
        self.dynamics_model = dynamics_model
    
    def set_policy(self, policy):
        self.policy = policy
    
    def set_U(self, U):
        self.U = U
    
    def set_prev_network(self):
        self.prev_network = copy.deepcopy(self)
    
    def current_pos(self, pos):
        self.pos = pos

    def is_state_safe(self, state, pos):
        # if abs(pos[0]) <= 1:
        #     return 0.05
        # else:
        #     return 1.05
        return isStateSafe(state)
    
    def forward(self, state):
        
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))

        if self.env_specs is not None:
            x0 = torch.relu(self.layer1(self.env_specs['initial_state']))
            x0 = torch.relu(self.layer2(x0))
            x0 = torch.sigmoid(self.layer3(x0))
        else:
            raise ValueError("Environment specs not set!")
        
        x = 1 - torch.log(1 + torch.exp(x - x0))
        
        if len(x.shape) == 2:
            x = x.squeeze()
        
        x = x - self.is_state_safe(state, self.pos)

        
        return x
    # @print_args_decorator
    def step(self, states: torch.FloatTensor):
        """
        For each state, use MALA (Metropolis Adjusted Langevin Algorithm)
        to move to a new state.

        
        state: torch.FloatTensor
        """
        if not states.requires_grad:
            states.requires_grad_(True)
        state_certificates = self(states)
        indicator_state = (state_certificates >= 0).float()
        
        log_prob_a = self.mala_temperature * (self.U(states, self.policy(states)).squeeze() - indicator_state)
        grad_a = torch.autograd.grad(log_prob_a.sum(), states, allow_unused=True, create_graph=True)[0]
        if grad_a is None:
            grad_a = torch.zeros_like(states)

        x_prime = states + self.mala_step_size*grad_a + torch.randn_like(states)*(2*self.mala_step_size)**0.5

        proposed_state_certificates = self(x_prime)
        indicator_proposed_state = (proposed_state_certificates >= 0).float()

        log_prob_b = self.mala_temperature * (self.U(x_prime, self.policy(x_prime)).squeeze() - indicator_proposed_state)
        grad_b = torch.autograd.grad(log_prob_b.sum(), x_prime, allow_unused=True, create_graph=True)[0]
        if grad_b is None:
            grad_b = torch.zeros_like(x_prime)

        proposal_correction_ratio = torch.norm((states - x_prime)*(grad_b - grad_a) / 2, dim = 1)

        acceptance_ratio = torch.exp(self.mala_temperature*(log_prob_b - log_prob_a + proposal_correction_ratio))

        mask = (torch.rand_like(acceptance_ratio) < acceptance_ratio).unsqueeze(1)

        states = torch.where(mask, x_prime, states)

        return states

    
    def compute_loss(self, states):
        """
        states: torch.FloatTensor
        """

        C_loss = self.U(states, self.policy(states)).mean()

        if self.prev_network is not None:
            R_loss = torch.relu(self.prev_network(states) - self(states)).mean()
        else:
            R_loss = torch.tensor(0)
        
        loss = C_loss + self.lambd * R_loss
        return loss
        
    def train(self, policy, dynamics, U, max_epochs=100, verbose=True):
        self.set_policy(policy)
        self.set_dynamics_model(dynamics)
        self.set_U(U)

        """Initialize m candidates from the state space randomly"""
        """Ensure that the states are within the state space bounds"""
        candidate_states = torch.rand((self.m, self.env_specs['state_space_low'].shape[0]), dtype=torch.float32).float()
        # candidate_states[-16:] = torch.zeros_like(candidate_states[-16:])

        """Get 3 indices between 16 and 28"""
        # indices = torch.randint(16, 28, (3,))
        # candidate_states[indices] = torch.rand((3, self.env_specs['state_space_low'].shape[0]), dtype=torch.float32).float()

        candidate_states = candidate_states*(self.env_specs['state_space_high'] - self.env_specs['state_space_low']) + self.env_specs['state_space_low']


        # print(candidate_states)

        for epoch in range(max_epochs):
            candidate_states = self.step(candidate_states)

            mask = self(candidate_states) >= 0
            
            trainable_states = candidate_states[mask]

            if len(trainable_states) == 0:
                continue
            
            loss = self.compute_loss(trainable_states)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        
    

class TransitionDynamics(BaseModel):
    def __init__(self, state_dim, action_dim, num_models, hidden_size_1, hidden_size_2, learning_rate = 0.001):
        super(TransitionDynamics, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        
        self.models = nn.ModuleList([ProbabilisticTransitionDynamics(state_dim, action_dim, hidden_size_1, hidden_size_2, learning_rate) for _ in range(num_models)])

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.lambd = 0.01           # Regularization parameter
        self.__initialize_models()


    def __initialize_models(self):
        for model_idx, model in enumerate(self.models):
            seed = model_idx + torch.initial_seed()
            torch.manual_seed(seed)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, seed)
                else:
                    nn.init.constant_(param, 0)
    def set_env_specs(self, env_specs: dict):
        for model in self.models:
            model.set_env_specs(env_specs)
        
    # @print_args_decorator
    def forward(self, state : torch.FloatTensor, action : np.ndarray):
        mean_mean, mean_std = 0, 0
        for model in self.models:
            dist = model(state, action)
            mean_mean += dist.mean
            mean_std += dist.stddev
        
        mean_mean /= len(self.models)
        mean_std /= len(self.models)

        return torch.distributions.Normal(mean_mean, mean_std)
    
    # @print_args_decorator
    def compute_loss(self, state, action, next_state):
        prediction_loss = 0
        regularization_loss = 0

        for model in self.models:
            dist = model(state, action)
            
            prediction_loss += -dist.log_prob(next_state).mean()
            regularization_loss += torch.norm(model.max_log_std - model.min_log_std)
        # print('Prediction Loss: ', prediction_loss, 'Regularization Loss: ', regularization_loss)
        
        loss = (prediction_loss + self.lambd*regularization_loss)/len(self.models)
        return loss

    # @print_args_decorator
    def train(self, states, actions, next_states, max_epochs=100, batch_size=64, shuffle=True, verbose=True):
        dataset = TensorDataset(states, actions, next_states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(max_epochs):
            epoch_loss = 0
            for state, action, next_state in dataloader:
                self.optimizer.zero_grad()
                loss = self.compute_loss(state, action, next_state)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            if verbose and epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")
        