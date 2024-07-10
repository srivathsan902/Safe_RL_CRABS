import safety_gymnasium
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def select_safe_action(env, agent_action):
    '''
    Select an action that does not have a cost.
    To do so, select random actions till an 
    action with zero cost is selected.

    Action space is continuous, so not
    possible to use action_space.sample()
    '''
    agent_action_safe = True
    safe_actions = []

    max_try = 100
    try_num = 0

    for _ in range(10):
        cost = 1
        while cost > 0:
            if try_num >= max_try:
                break
            try_num += 1
            copy_env = copy.deepcopy(env)

            if agent_action_safe:
                next_state, reward, cost, done, truncated, _ = copy_env.step(agent_action)
                if cost == 0:
                    copy_env.close()
                    del copy_env
                    return agent_action, True
                agent_action_safe = False

            else:
                action = copy_env.action_space.sample()
                next_state, reward, cost, done, truncated, _ = copy_env.step(action)
                copy_env.close()
                del copy_env
        safe_actions.append(action)
    

    '''
    Choose the safe action that has least KL divergence with the agent action
    '''
    if len(safe_actions) == 0:
        return agent_action, False
    else:
        safe_actions = np.array(safe_actions)
        kl_divergence = np.sum(np.abs(safe_actions - agent_action), axis=1)
        safe_action = safe_actions[np.argmin(kl_divergence)]

        return safe_action, True

def generate_random_states_uniform(env, low, high, num_samples=10):
    states = np.random.uniform(low, high, size=(num_samples, state_dim))
    return states

if __name__ == '__main__':
    # env = safety_gymnasium.make('SafetyPointCircle1-v0', render_mode='human')

    # max_action = float(max((env.action_space.high)))
    # min_action = float(min((env.action_space.low)))
    # env.reset()
    # action = env.action_space.sample()
    # safe_action = select_safe_action(env, action)
    # print(f'Agent action: {action}, Safe action: {safe_action}')
    # state, reward, cost, done, truncated, _ = env.step(action)
    
    # copy_env = copy.deepcopy(env)

    # if hasattr(copy_env, 'render_mode'):
    #     copy_env.render_parameters.mode = 'None'
    #     print(copy_env.render_mode)
    # else:
    #     print('No render_mode attribute')

    # if hasattr(copy_env.env, 'state'):
    #     print(copy_env.env.state)
    # else:
    #     print('No state attribute')

    # for episode in range(10):
    #     env.reset()
    #     for steps in range(250):
    #         if episode % 2:
    #             action = [0,0.5]
    #         else:
    #             action = [-1,0]
    #         next_state, reward, cost, done, truncated, info = env.step(action)
    #         print(info)
        
    env_id = 'SafetyPointCircle1-v0'
    env = safety_gymnasium.make(env_id)
    # env = safety_gymnasium.make(env_id, render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    # print(max_action, min_action)

    low = env.observation_space.low
    high = env.observation_space.high
    inf = 1e6

    for i in range(len(low)):
        if low[i] <= -inf:
            low[i] = -1000
        if high[i] >= inf:
            high[i] = 1000
    # Generate 10 random states with a uniform distribution
    random_states_uniform = generate_random_states_uniform(env, low, high, num_samples=10000)
    # for state in random_states_uniform:
        # print(state)




