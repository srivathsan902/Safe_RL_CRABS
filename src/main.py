import os
import sys
import time
import yaml
import wandb
import shutil
import numpy as np
import safety_gymnasium
from dotenv import load_dotenv

from train import train
# from ddpgAgent import DDPGAgent
from CRABS import CRABS

load_dotenv()

artifacts_folder = 'artifacts'
os.environ['WANDB_MODE'] = 'offline'

def main(dir_name, params):

    env_id = params['main'].get('env_id', 'SafetyPointCircle1-v0')
    render_mode = params['main'].get('render_mode', None)
    if render_mode == 'None':
        env = safety_gymnasium.make(env_id)
    else:
        env = safety_gymnasium.make(env_id, render_mode = render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    low = env.observation_space.low
    high = env.observation_space.high

    low = np.clip(low, -6, 6)
    high = np.clip(high, -6, 6)

    hidden_size_1 = params['main']['agent'].get('hidden_size_1', 64)
    hidden_size_2 = params['main'].get('hidden_size_2', 128)

    agent = CRABS(state_dim, low, high, action_dim, min_action, max_action, hidden_size_1 = hidden_size_1, hidden_size_2 = hidden_size_2)

    wandb_enabled = params['base']['wandb_enabled']

    if wandb_enabled:
        try:
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)

        except Exception as e:
            print(f"Error occurred while logging into wandb: {e}")
            sys.exit(1)

        run_name = dir_name.replace('/','-').replace('\\','-').replace('artifacts-', "")
        config = {
            'env_id': env_id,
            'render_mode': render_mode,
            'hidden_size_1': hidden_size_1,
            'hidden_size_2': hidden_size_2,
            'run_name': f'{env_id}-{run_name}',
            'save_every': params['train']['save_every'],
        }

        if params['main']['update']:
            config['update'] = True
            config['update_from'] = params['main']['update_from']

        wandb.init(project='Safe Reinforcement Learning', name = f'{env_id}-{run_name}', config = config)

        if params['main']['update']:

            download_name = config['update_from'].replace('/','-').replace('\\','-').replace('artifacts-', "")

            artifact_names = ['actor', 'critic', 'safety_critic', 'actor_target', 'critic_target', 'safety_critic_target', 'replay_buffer']

            for artifact_name in artifact_names:
                root = os.path.join(dir_name,config['update_from'].split('/')[-1])
                artifact = wandb.use_artifact(f'{artifact_name}:{download_name}', type="model")
                artifact_dir = artifact.download(root = root)
                print(f"Artifact downloaded to: {artifact_dir}")

            # Remove the episode number from end of download name
            download_name = download_name.split('-')[:-1]
            download_name = '-'.join(download_name)
            data_artifacts = ['episode_rewards', 'episode_costs', 'episode_percent_safe_actions']

            for artifact_name in data_artifacts:
                root = dir_name
                artifact = wandb.use_artifact(f'{artifact_name}:{download_name}', type="data")
                artifact_dir = artifact.download(root = root)
                print(f"Artifact downloaded to: {artifact_dir}")
        else:
            os.makedirs(dir_name, exist_ok=True)
    
    else:
        try:
            if params['main'].get('update', False):
                update_from = params['main'].get('update_from', False)

                if update_from:
                    if os.path.exists(update_from):
                        # Create new directory if it doesn't exist
                        os.makedirs(dir_name, exist_ok=True)

                        # Copy contents from old_dir_name to new dir_name
                        for item in os.listdir(update_from):
                            old_item_path = os.path.join(update_from, item)
                            new_item_path = os.path.join(dir_name, item)
                            if os.path.isdir(old_item_path):
                                shutil.copytree(old_item_path, new_item_path)
                            else:
                                shutil.copy2(old_item_path, new_item_path)

                        print(f"Contents from '{update_from}' copied to '{dir_name}'.")
                    else:
                        raise FileNotFoundError(f"Directory '{update_from}' does not exist.")

            else:
                os.makedirs(dir_name, exist_ok=True)

        except Exception as e:
            print(f"Error occurred: {e}")
            sys.exit(1)

    episode_rewards = []
    episode_costs = []

    # Load the latest models if they exist
    start_episode = 0
    if os.path.exists(dir_name) and len(os.listdir(dir_name)) > 0:
        latest_episode = max(int(ep) for ep in os.listdir(dir_name) if ep.isdigit())
        agent.load(dir_name)
        start_episode = latest_episode
    # Load existing reward and cost logs if they exist
    if os.path.exists(os.path.join(dir_name,'episode_rewards.npy')):
        episode_rewards = np.load(os.path.join(dir_name,'episode_rewards.npy')).tolist()
        episode_rewards = episode_rewards[:start_episode]
    else:
        episode_rewards = [0]*start_episode

    if os.path.exists(os.path.join(dir_name,'episode_costs.npy')):
        episode_costs = np.load(os.path.join(dir_name,'episode_costs.npy')).tolist()
        episode_costs = episode_costs[:start_episode]
    else:
        episode_costs = [0]*start_episode

    if os.path.exists(os.path.join(dir_name,'episode_percent_safe_actions.npy')):
        episode_percent_safe_actions = np.load(os.path.join(dir_name,'episode_percent_safe_actions.npy')).tolist()
        episode_percent_safe_actions = episode_percent_safe_actions[:start_episode]
    else:
        episode_percent_safe_actions = [0]*start_episode


    if wandb_enabled:

        for i in range(len(episode_rewards)):

            wandb.log({
                'Reward': episode_rewards[i],
                'Cost': episode_costs[i],
                '% Safe Actions': episode_percent_safe_actions[i],  
            })

    training_gen = train(env, agent, dir_name, params, start_episode)

    try:
        for episode, reward, cost, percent_safe_actions in training_gen:
            episode_rewards.append(reward)
            episode_costs.append(cost)
            episode_percent_safe_actions.append(percent_safe_actions)
            
            if wandb_enabled:
                wandb.log({
                    'Reward': reward,
                    'Cost': cost,
                    '% Safe Actions': percent_safe_actions,
                })

    except ValueError as e:
        print(f"Error occurred during training: {e}")

    np.save(os.path.join(dir_name,'episode_rewards.npy'), np.array(episode_rewards))
    np.save(os.path.join(dir_name,'episode_costs.npy'), np.array(episode_costs))
    np.save(os.path.join(dir_name,'episode_percent_safe_actions.npy'), np.array(episode_percent_safe_actions))

    if wandb_enabled:
        artifact_names = ['episode_rewards', 'episode_costs', 'episode_percent_safe_actions']
        for artifact_name in artifact_names:
            artifact = wandb.Artifact(artifact_name, type="data")
            artifact.add_file(os.path.join(dir_name, f'{artifact_name}.npy'))
            
            artifact.metadata = {
                "root": dir_name
            }
            wandb.log_artifact(artifact, aliases=[run_name])
        
        wandb.finish()


if __name__ == '__main__':

    with open('src/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    '''
    Create the dir name based on the current time: dd_mm_yyyy_hh_mm_ss
    artifacts/yyyy/mm/dd/hh_mm should be the structure
    '''
    dir_name = os.path.join(artifacts_folder,
                        time.strftime('%Y'),
                        time.strftime('%m'),
                        time.strftime('%d'))
    
    run = 1
    while os.path.exists(os.path.join(dir_name, f'Run_{run}')):
        run += 1
    dir_name = os.path.join(dir_name, f'Run_{run}')

    main(dir_name, params)
    


