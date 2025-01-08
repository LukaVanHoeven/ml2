from air_hockey_challenge.framework import AgentBase, AirHockeyChallengeWrapper
from dreamer import dreamer
import numpy as np

def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    raise NotImplementedError


class DreamerAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        self.env_info = env_info
        self.kwargs = kwargs
        self.agent = dreamer.Dreamer(self.env_info, act_space, config, logger, dataset) #observation space, act_space, config, logger?, dataset????)
        

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        velocity = np.zeros_like(self.hold_position)
        action = np.vstack([self.hold_position, velocity])
        return action
    
    def draw_action2(self, observation):
        # Implement action to be taken by the agent
        return None

    def reset(self):
        # Implement reset of the agent
        return None

    @property
    def observation_space(self):
        # Implement observation space of the agent
        return None
    
    @property
    def action_space(self):
        # Implement action space of the agent
        return None
    
    @property
    def infer_action(self):
        # Implement infer action of the agent
        return None 
    
