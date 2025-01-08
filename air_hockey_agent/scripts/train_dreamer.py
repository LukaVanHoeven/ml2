import os
import numpy as np
import torch
from torch import nn
from torch import distributions as torchd
import gym
from gym import spaces
from types import SimpleNamespace

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.agent_base import AgentBase as BaselineAgent
from air_hockey_agent.dreamer.dreamer import Dreamer

from dreamer import tools as Tools

class MinimalHockeyEnv(gym.Env):
    def __init__(self, env_str='3dof-hit'):
        super().__init__()
        # Create base environment
        self.env = AirHockeyChallengeWrapper(env=env_str, custom_reward_function=None)
        
        # Create baseline opponent
        self.opponent = BaselineAgent(self.env.env_info, agent_id=2)
        
        # Store action indices for both agents
        self.action_idx = (
            np.arange(self.env.base_env.action_shape[0][0]),
            np.arange(self.env.base_env.action_shape[1][0]) if len(self.env.base_env.action_shape) > 1 else np.array([])
        )

        # Define observation space
        # Assuming standard air hockey observations including:
        # - Participant end-effector position (3D)
        # - Opponent end-effector position (3D)
        # - Puck position (3D)
        # - Puck velocity (3D)
        # Adjust these dimensions based on your actual observation space
        obs_dim = 12  # 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Define action space
        # For 3DOF robot, actions are typically position commands
        action_dim = self.env.base_env.action_shape[0][0]  # Usually 3 for 3DOF
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        obs1, obs2 = np.split(obs, 2)
        self._previous_obs2 = obs2
        return self._process_obs(obs1)

    def step(self, action1):
        # Get opponent action
        action2 = self.opponent.draw_action(self._previous_obs2)
        
        # Combine actions
        combined_action = (action1[self.action_idx[0]], action2[self.action_idx[1]])
        
        # Take environment step
        obs, reward, done, info = self.env.step(combined_action)
        
        # Split observations
        obs1, obs2 = np.split(obs, 2)
        self._previous_obs2 = obs2
        
        return self._process_obs(obs1), reward, done, info

    def _process_obs(self, obs):
        # Process raw observation into the format expected by DreamerV3
        # Combine relevant observations into a flat array
        # Adjust this based on your actual observation structure
        processed_obs = np.concatenate([
            obs[0:3],    # End-effector position
            obs[3:6],    # Opponent end-effector position
            obs[6:9],    # Puck position
            obs[9:12]    # Puck velocity
        ]).astype(np.float32)
        
        return processed_obs
    
    
def make_dataset(episodes, config):
    generator = Tools.sample_episodes(episodes, config.batch_length)
    dataset = Tools.from_generator(generator, config.batch_size)
    return dataset



def train_dreamer():
    # Environment setup
    env = MinimalHockeyEnv()
    
    # Create DreamerV3 agent
    config = dict(
        # Core training settings
        device='cpu', #use cuda if you have cuda
        batch_size=50,
        batch_length=50,
        train_steps=1000000,
        
        # Model architecture
        deter_size=200,
        stoch_size=30,
        num_units=400,
        
        # Learning rates
        model_lr=3e-4,
        actor_lr=8e-5,
        critic_lr=8e-5,
        train_ratio= 512,
        pretrain= 100,
        should_pretrain= True,
        opt_eps= 1e-8,
        grad_clip= 1000,
        dataset_size= 1000000,
        pt= "adam",
        reset_every= 0,
        # Other parameters
        expl_behavior= "greedy",
        expl_until= 3,
        expl_extr_scale= 0.0,
        expl_intr_scale= 1.0,
        disag_target= "stoch",
        disag_log= True,
        disag_models= 10,
        disag_offset= 1,
        disag_layers= 4,
        disag_units= 400,
        disag_action_cond= False,
        action_repeat= 2,
        step= 1,

        dyn_hidden= 512,
        dyn_deter= 512,
        dyn_stoch= 32,
        dyn_discrete= 32,
        dyn_rec_depth= 1,
        dyn_mean_act= "none",
        dyn_std_act= "sigmoid2",
        dyn_min_std= 0.1,
        grad_heads= ["decoder", "reward", "cont"],
        units= 512,
        act= "SiLU",
        norm= True,

        precision= 32,

        dyn_scale= 0.5,
        rep_scale= 0.1,
        kl_free= 1.0,
        weight_decay= 0.0,
        unimix_ratio= 0.01,
        initial= "learned",

        log_every=1e4,

        encoder=
            dict(
            mlp_keys= "$^",
            cnn_keys= "$^",
            act= "SiLU",
            norm= True,
            cnn_depth= 32,
            kernel_size= 4,
            minres= 4,
            mlp_layers= 5,
            mlp_units= 1024,
            symlog_inputs= True,
            ),
        decoder=
            dict(
            mlp_keys= "$^",
            cnn_keys= "$^",
            act= "SiLU",
            norm= True,
            cnn_depth= 32,
            kernel_size= 4,
            minres= 4,
            mlp_layers= 5,
            mlp_units= 1024,
            cnn_sigmoid= False,
            image_dist= "mse",
            vector_dist= "symlog_mse",
            outscale= 1.0,
            ),
        actor=
            dict(
            layers= 2,
            dist= "normal",
            entropy= 3e-4,
            unimix_ratio= 0.01,
            std= "learned",
            min_std= 0.1,
            max_std= 1.0,
            temp= 0.1,
            lr= 3e-5,
            eps= 1e-5,
            grad_clip= 100.0,
            outscale= 1.0,
            ),
        critic=
            dict(
            layers= 2,
            dist= "symlog_disc",
            slow_target= True,
            slow_target_update= 1,
            slow_target_fraction= 0.02,
            lr= 3e-5,
            eps= 1e-5,
            grad_clip= 100.0,
            outscale= 0.0,
            ),

        reward_head=dict( layers= 2, dist= "symlog_disc", loss_scale= 1.0, outscale= 0.0 ),
        cont_head= dict( layers= 2, loss_scale= 1.0, outscale= 1.0 ),
        opt= "adam",
        reward_EMA= True,
        compile= True,
    )
    config = SimpleNamespace(**config)
    config.num_actions = 0

    logdir = os.path.join(os.getcwd(), 'logs')

    logger = Tools.Logger(logdir, config.action_repeat * config.step)
    
    
    directory = os.path.join(os.getcwd(), 'data')
    train_eps = Tools.load_episodes(directory, limit=config.dataset_size)
    
    
    train_dataset = make_dataset(train_eps, config)
    
    agent = Dreamer(obs_space=env.observation_space, act_space=env.action_space, config=config, logger=logger, dataset=train_dataset)
    
    # Training loop
    total_episodes = 0
    total_steps = 0
    
    while total_steps < config.train_steps:
        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Get action from agent
            action = agent(obs, training=True, reset=env.reset)['action']
            
            # Take step in environment
            next_obs, reward, done, _ = env.step(action)
            
            # Store transition for training
            agent.experience(obs, action, reward, done)
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            # Train agent
            if len(agent.buffer) > config.batch_size:
                agent.train()
        
        # Episode complete
        total_episodes += 1
        
        # Log progress
        if total_episodes % 10 == 0:
            print(f"Episode {total_episodes}, Steps {total_steps}")
            print(f"Episode Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    train_dreamer()