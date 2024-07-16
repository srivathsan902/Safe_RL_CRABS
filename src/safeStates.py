import torch
import numpy as np
from utils import get_coords

def isStateSafe(state):
    """
    Here comes the human input.
    
    Implement what determines safety here.
    If the state is safe, return a number close to 0
    If the state is unsafe, return a number > 1 
    
    """
    # print('in safestates: ',state.shape)
    max_lidar_distance = 6
    lidar_resolution = 16
    states = state.detach().numpy()
    print('state', state.shape)
    x,y,theta_local,info_lidar = get_coords(states, max_lidar_distance, lidar_resolution)

    print('x', x.shape)
    
    mask = abs(x) >= 1.125

    is_state_safe = torch.full_like(state[:,0], 0.05)
    is_state_safe[mask] = 1.05

    # if len(is_state_safe.shape) == 1:
    #     return is_state_safe.unsqueeze(1)
    return is_state_safe


