import torch
def isStateSafe(state):
    """
    Here comes the human input.
    
    Implement what determines safety here.
    If the state is safe, return a number close to 0
    If the state is unsafe, return a number > 1 
    
    """
    # print('in safestates: ',state.shape)
    if len(state.shape) == 2:
        second_feature = state[:, 1]
    else:
        second_feature = state[:,:,1]

    expanded_second_feature = second_feature.unsqueeze(-1) if len(state.shape) == 3 else second_feature

    # Check if the 2nd feature is less than 0.5
    safety_tensor = torch.where(expanded_second_feature < 0.5, torch.tensor(0.05), torch.tensor(1.05))
    # safety_tensor = torch.where(second_feature < 0.5, 0.05, 1.05)
    # print('safety_tensor: ', safety_tensor.shape)
    return safety_tensor


