from tqdm import tqdm
import numpy as np
import os

import wandb
from safetyPolicy import select_safe_action


def train(env, agent, dir_name, params, start_episode = 0):
    
    num_episodes = params['train'].get('num_episodes', 1000)
    batch_size = params['train'].get('batch_size', 64)
    max_steps_per_episode = params['train'].get('max_steps_per_episode', 250)

    SAVE_EVERY = params['train'].get('save_every', 100)

    for episode in tqdm(range(start_episode, start_episode + num_episodes), desc='Training'):
        state, info = env.reset()
        state_dim = env.observation_space.shape[0]
        low = env.observation_space.low
        high = env.observation_space.high

        low = np.clip(low, -1000, 1000)
        high = np.clip(high, -1000, 1000)
        agent.barrier_certificate._set_env_properties(state, state_dim, low, high)
        agent.ou_noise.reset()
        episode_reward = 0
        episode_cost = 0
        num_safe_actions = 0
        for t in range(max_steps_per_episode):
            # print('Episode:', episode, 'Step:', t, end= "\r")
            action = agent.select_action(np.array(state))
            next_state, reward, cost, done, truncated, _ = env.step(action)

            
            if cost == 0:
                num_safe_actions += 1
            else:
                print('Cost incurred: ', cost)
            agent.replay_buffer.add(state, action, reward, next_state, cost, done)

            state = next_state
            episode_reward += reward
            episode_cost += cost

            if agent.replay_buffer.size > batch_size:
                agent.train(batch_size)

            if done or truncated:
                break

        # os.system('cls' if os.name == 'nt' else 'clear')

        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(os.path.join(dir_name , f'{episode + 1}'))

            wandb_enabled = params['base']['wandb_enabled']

            if wandb_enabled:

                local_path = os.path.join(dir_name, f'{episode + 1}')
                run_name = dir_name.replace('/','-').replace('\\','-').replace('artifacts-', "")

                for file in os.listdir(os.path.join(dir_name, f'{episode + 1}')):
                    model_path = os.path.join(local_path, file)
                    file_name = os.path.splitext(file)[0]

                    artifact = wandb.Artifact(file_name, type="model")
                    artifact.add_file(model_path)
                    artifact.metadata = {
                        "root": local_path
                    }
                    wandb.log_artifact(artifact, aliases=[run_name + f"-{episode + 1}"])
                

        percent_safe_actions = num_safe_actions / max_steps_per_episode * 100
        yield episode, episode_reward, episode_cost, percent_safe_actions
    
    env.close()
    
