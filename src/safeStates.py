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
    # print('state', state.shape)

    # if len(states.shape) == 1:
    #     if max(states[-16:]) >= 0.95:
    #         return torch.tensor([0.05])
    #     else:
    #         return torch.tensor([1.05])
    # else:
    #     mask = np.any(states[:,-16:] >= 0.95, axis=1)
    #     is_state_safe = torch.full_like(state[:,0], 0.05)
    #     is_state_safe[mask] = 1.05
    #     return is_state_safe

    if len(state.shape) == 1:
        if torch.any(state[-16:]) >= 0.995:
            return torch.tensor(1.05)
        else:
            return torch.tensor(0.05)
    elif len(state.shape) == 2:
        # print('state', state.shape)
        # print(state[0][-16:])
        mask = torch.any(state[:,-16:] >= 0.995, dim=1)
        # print(mask)
        is_state_safe = torch.full_like(state[:,0], 0.05)
        is_state_safe[mask] = 1.05
        return is_state_safe
    else:
        mask = torch.any(state[:,:,-16:] >= 0.995, dim=2)
        is_state_safe = torch.full_like(state[:,:,0], 0.05)
        is_state_safe[mask] = 1.05
        is_state_safe = is_state_safe.unsqueeze(2)
        
        return is_state_safe


    x,y,theta_local,info_lidar = get_coords(states, max_lidar_distance, lidar_resolution)

    # print('x', x.shape)
    
    mask = abs(x) >= 1.125

    is_state_safe = torch.full_like(state[:,0], 0.05)
    is_state_safe[mask] = 1.05

    # if len(is_state_safe.shape) == 1:
    #     return is_state_safe.unsqueeze(1)
    return is_state_safe


