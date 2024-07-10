import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from safeStates import isStateSafe

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


class Actor(BaseModel):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(Actor, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, action_dim)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
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
    
class MALA:
    def __init__(self, step_size, num_steps,lambd):
        self.step_size = step_size
        self.num_steps = num_steps
        self.lambd = lambd

    def set_dynamics_model(self, dynamics_model):
        self.dynamics_model = dynamics_model

    def set_barrier_certificate(self, barrier_certificate):
        self.barrier_certificate = barrier_certificate
    
    def U(self, state, action):
        num_samples = 100
        state = state
        # print('IN U', ' state', state.shape, 'action', action.shape)
        state = torch.clamp(state, -1000, 1000)
        dist = self.dynamics_model(state, action)
        # print('distribution ', dist)
        # dist = torch.distributions.Normal(mean, std)
        samples = dist.sample((num_samples,))
        # print('samples : ', samples)
        certificate_values = self.barrier_certificate(samples)
        # print('Max Certificate value ',-1*certificate_values.max())
        # Return the max of these values
        return -1*certificate_values.max()

    def step(self, initial_state, action):
        state = initial_state.clone().detach()
        # print('Initial State', state.shape)
        
        for _ in range(self.num_steps):
            if not state.requires_grad:
                state.requires_grad_(True)

            barrier_certificate = self.barrier_certificate(state)
            barrier_condition = (barrier_certificate < 0).float()
            U_value = self.U(state, action)
            log_prob = self.step_size*(U_value - self.lambd*barrier_condition)
            log_prob_sum = log_prob.sum()
            
            grad_log_prob = torch.autograd.grad(log_prob_sum, state, allow_unused=True)[0]
            # Replace None with zeros
            grad_log_prob = torch.zeros_like(state) if grad_log_prob is None else grad_log_prob
            proposed_state = state + self.step_size*grad_log_prob + (2*self.step_size)**0.5*torch.randn_like(state)

            U_value_proposed = self.U(proposed_state, action)
            barrier_certificate_proposed = self.barrier_certificate(proposed_state)
            barrier_condition_proposed = (barrier_certificate_proposed < 0).float()
            log_prob_proposed = self.step_size*(U_value_proposed - self.lambd*barrier_condition_proposed)

            random_tensor = torch.rand_like(log_prob)
            acceptance_criterion = random_tensor < torch.exp(log_prob_proposed - log_prob)
            # print('Acceptance Criterion', acceptance_criterion.shape, 'proposed_state', proposed_state.shape, 'state', state.shape)
            state = torch.where(acceptance_criterion.unsqueeze(1), proposed_state, state)
            # print(random_tensor.shape, torch.exp(log_prob_proposed - log_prob).shape)
            # if random_tensor < torch.exp(log_prob_proposed - log_prob):
            #     state = proposed_state

        return state
    

class BarrierCertificate(BaseModel):

    def __init__(self, state_dim, hidden_size_1 = 256, hidden_size_2 = 256, initial_state = None):
        super(BarrierCertificate, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, 1)

    def _set_env_properties(self, initial_state, state_dim, low, high):
        self.initial_state = torch.FloatTensor(initial_state).float()
        self.state_dim = state_dim
        self.low = low
        self.high = high


    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))

        x0 = torch.relu(self.layer1(self.initial_state))
        x0 = torch.relu(self.layer2(x0))
        x0 = torch.sigmoid(self.layer3(x0))

        # print('In Barrier Certificate', 'state', state.shape, 'x0', x0.shape, 'x', x.shape)
        x = 1 - (torch.log(1 + torch.exp(x - x0)))
        # print('Inside Barrier Certificate', x.shape)
        if len(x.shape) == 2:
            x = x.squeeze()
        x = x - isStateSafe(state)
        # print('Inside Barrier Certificate', x.shape)

        return x

    def train(self, policy, dynamics_model, train_size = 10000):
        max_epochs = 1

        # states = np.random.uniform(self.low, self.high, size=(train_size, self.state_dim))
        low = torch.tensor(self.low, dtype=torch.float32)
        high = torch.tensor(self.high, dtype=torch.float32)
        # Generate uniform random states directly in PyTorch
        states = ((high - low) * torch.rand(train_size, self.state_dim) + low)
        # print('states in start of train', states)
        # print('states in start of train', states.shape)
    
        mala = MALA(step_size=0.1, num_steps=10, lambd=0.1)
        mala.set_barrier_certificate(self)
        mala.set_dynamics_model(dynamics_model)

        print('\nTraining Barrier Certificate')
        for epoch in range(max_epochs):
            # print('Training Barrier Certificate Epoch', epoch + 1)
            actions = policy(states)
            dataset = TensorDataset(states, actions)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

            train_states = []
            # cnt = 0
            for state, action in dataloader:
                # print('Training Barrier Certificate Batch', cnt + 1)
                # cnt += 1
                '''
                Get the adversarial state for a given state, action pair
                '''
                # print('state in barrier certificate train', state.shape)
                # print('action in barrier certificate train', action.shape)
                state = (mala.step(state, action))
                # print('state after sampling in barrier certificate train', state.shape)
                
                barrier_values = self(state)
                condition = barrier_values >= 0
                train_states.extend(state[condition])
            
            if len(train_states) == 0:
                # print('No safe states found in the training data')
                continue
            
            train_states = torch.stack(train_states)
            # print('Train States', train_states.shape)
            actions = policy(train_states)
            # print('Actions', actions.shape)
            loss = mala.U(train_states, actions)
            print(f'Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item()} ', end='\r')
            # print('U_values', U_values.shape)
            # loss = (mala.U(train_states, policy(train_states)))

            if loss < 0:
                # print('Converged to a safe policy as loss is negative')
                continue

            # print(f'Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item()}')
            if loss < 1e-3:
                # print('Converged to a safe policy as loss is negligible')
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    

class ProbabilisticDynamicsModel(BaseModel):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(ProbabilisticDynamicsModel, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        
        self.mean = nn.Linear(hidden_size_2, state_dim)
        self.log_std = nn.Linear(hidden_size_2, state_dim)

    def forward(self, state, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)

        action = action.float()
        state = state.float()

        state = torch.clamp(state, -1000, 1000)
        # print('In PDM : ','state', state.shape, 'action', action.shape)

        if len(state.shape) == 1:
            x = torch.cat([state, action], 0)
        else:
            x = torch.cat([state, action], 1)

        x = torch.relu(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        # print('mean, std', mean, std)
        
        return mean, std
    

class DynamicsModel(BaseModel):
    def __init__(self, state_dim, action_dim, num_models=10, hidden_size_1=256, hidden_size_2=256, learning_rate=1e-3):
        super(DynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.models = nn.ModuleList([ProbabilisticDynamicsModel(self.state_dim, self.action_dim, hidden_size_1, hidden_size_2) for _ in range(num_models)])
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Apply custom weight initialization to each model
        self._initialize_weights()

    def _initialize_weights(self):
        for i, model in enumerate(self.models):
            seed = torch.initial_seed() + i
            torch.manual_seed(seed)
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    
    def get_dynamics(self, state, action):
        means = []
        stds = []
        state = torch.clamp(state, -1000, 1000)
        for model in self.models:
            mean, std = model(state, action)
            means.append(mean)
            stds.append(std)
        
        mean_mean = torch.stack(means).mean(dim=0)
        mean_std = torch.stack(stds).mean(dim=0)
        
        return mean_mean, mean_std

    def forward(self, state, action):
        mean_mean, mean_std = self.get_dynamics(state, action)
        dist = torch.distributions.Normal(mean_mean, mean_std)
        return dist
    
    def train(self, dataset, batch_size=64, num_epochs=200):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print('\nTraining Dynamics Model')
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for state, action, next_state in dataloader:
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state)
                if isinstance(action, np.ndarray):
                    action = torch.FloatTensor(action)
                if isinstance(next_state, np.ndarray):
                    next_state = torch.FloatTensor(next_state)

                self.optimizer.zero_grad()
                
                dist = self(state, action)
                loss = -dist.log_prob(next_state).sum(dim=1).mean()
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}', end='\r')


    def save(self, path):
        '''
        Save the model, parameters, and optimizer state for all probabilistic models
        '''
        torch.save({
            'model_states': [model.state_dict() for model in self.models],
            'hidden_size_1': self.hidden_size_1,
            'hidden_size_2': self.hidden_size_2
        }, path)

    def load(self, path):
        '''
        Load the model, parameters, and optimizer state for all probabilistic models
        '''
        checkpoint = torch.load(path)
        self.hidden_size_1 = checkpoint.get('hidden_size_1', 64)
        self.hidden_size_2 = checkpoint.get('hidden_size_2', 128)
        model_states = checkpoint.get('model_states', [])
        for model, state_dict in zip(self.models, model_states):
            model.load_state_dict(state_dict)
