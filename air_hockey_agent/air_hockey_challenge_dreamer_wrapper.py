from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
import gym
import numpy as np

class AirHockeyChallengeDreamerWrapper(AirHockeyChallengeWrapper):
    @property
    def observation_space(self):
        obs_space_size = len(self.env_info['puck_pos_ids']) + len(self.env_info['puck_vel_ids']) + \
                         len(self.env_info['joint_pos_ids']) + len(self.env_info['joint_vel_ids'])
        observation_box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32)
        return gym.spaces.Dict({"image": observation_box})


    @property
    def action_space(self):
        # The desired XY position of the mallet
        #  - pos_x
        #  - pos_y
        action_dim = 2 * self.env_info['robot']['n_joints']
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(action_dim,),
            dtype=np.float32,
        )