from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
import gym
import numpy as np
from copy import deepcopy

class AirHockeyChallengeDreamerWrapper(AirHockeyChallengeWrapper):
    def __init__(self, env, custom_reward_function=None, interpolation_order=3, agent=None, **kwargs):
        super().__init__(env, custom_reward_function, interpolation_order, **kwargs)
        self.agent = agent
    
    
    @property
    def observation_space(self):
        obs_space_size = len(self.env_info['puck_pos_ids']) + len(self.env_info['puck_vel_ids']) + \
                         len(self.env_info['joint_pos_ids']) + len(self.env_info['joint_vel_ids'])
        observation_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32)

        spaces = {}
        """
        puck_pos_ids_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.env_info['puck_pos_ids']) ,), dtype=np.float32)
        puck_vel_ids_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.env_info['puck_vel_ids']) ,), dtype=np.float32)
        joint_pos_ids_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.env_info['joint_pos_ids']) ,), dtype=np.float32)
        joint_vel_ids_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.env_info['joint_vel_ids']) ,), dtype=np.float32)

        spaces["puck_pos_ids_box"] = puck_pos_ids_box
        spaces["puck_vel_ids_box"] = puck_vel_ids_box
        spaces["joint_pos_ids_box"] = joint_pos_ids_box
        spaces["joint_vel_ids_box"] = joint_vel_ids_box
        """
        #spaces["image"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        spaces["player"] = observation_box
        spaces["is_first"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        spaces["is_last"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        spaces["is_terminal"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        return gym.spaces.Dict(spaces)


    @property
    def action_space(self):
        # The desired XY position of the mallet
        #  - pos_x
        #  - pos_y
        action_dim = 2 * self.env_info['robot']['n_joints']
        box = gym.spaces.Box(
            low=-1,
            high=1,
            shape= (action_dim,),
            #shape=(2, 3),
            dtype=np.float32,
        )
        return box
    
    def reset(self, state=None):
        obs = self.base_env.reset(state)
        self.agent.reset()
        #print(f'obs in reset {obs}')
        obs = {
            #"image": [0, 0, 0],
            "player": obs,
            "is_first": True,
            "is_last": False,
            "is_terminal": False
            }
        return obs

    def step(self, action):
        #print(f'action in step {action}')

        obs, reward, done, info = self.base_env.step(action)

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            info["jerk"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2): (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(deepcopy(self.env_info['constraints'].fun(
                    obs_agent[self.env_info['joint_pos_ids']], obs_agent[self.env_info['joint_vel_ids']])))
                info["jerk"].append(
                    self.base_env.jerk[i * self.env_info['robot']['n_joints']:(i + 1) * self.env_info['robot'][
                        'n_joints']])

            info["score"] = self.base_env.score
            info["faults"] = self.base_env.faults

        else:
            info["constraints_value"] = deepcopy(self.env_info['constraints'].fun(obs[self.env_info['joint_pos_ids']],
                                                                                  obs[self.env_info['joint_vel_ids']]))
            info["jerk"] = self.base_env.jerk
            info["success"] = self.check_success(obs)
        #print(f'obs in step {obs}')

        obs = {
            #"image": [0, 0, 0],
            "player": obs,
            "is_first": False,
            "is_last": done,
            "is_terminal": self.check_success(obs) > 0,
        }
        return obs, reward, done, info
