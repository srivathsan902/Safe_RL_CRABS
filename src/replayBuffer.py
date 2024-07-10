from collections import deque
import pickle
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.size = 0

    def add(self, state, action, reward, next_state, cost, done):
        self.buffer.append((state, action, reward, next_state, cost, done))
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, cost, dones = zip(*batch)
        return states, actions, rewards, next_states, cost, dones
    
    def save(self, path):
        np.save(path, np.array(self.buffer, dtype=object))

    def load(self, path):
        self.buffer = deque(np.load(path, allow_pickle=True).tolist(), maxlen=self.buffer.maxlen)
    
# Define the PER replay buffer
class PERBuffer:
    def __init__(self, max_size, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, cost, done):
        transition = (state, action, reward, next_state, cost, done)
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        if self.size < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == 0:
            return [], [], np.array([], dtype=np.float32)
        
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        if probs.sum() == 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + self.epsilon
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'buffer': self.buffer,
                'priorities': self.priorities,
                'pos': self.pos,
                'size': self.size
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = data['buffer']
            self.priorities = data['priorities']
            self.pos = data['pos']
            self.size = data['size']
