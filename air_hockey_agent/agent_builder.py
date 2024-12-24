# def build_agent(env_info, **kwargs):
#     """
#     Function where an Agent that controls the environments should be returned.
#     The Agent should inherit from the mushroom_rl Agent base env.

#     Args:
#         env_info (dict): The environment information
#         kwargs (any): Additionally setting from agent_config.yml
#     Returns:
#          (AgentBase) An instance of the Agent
#     """

#     raise NotImplementedError


import numpy as np
import mujoco as mujoco

from air_hockey_challenge.framework import AgentBase


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.

@@ -9,5 +15,41 @@ def build_agent(env_info, **kwargs):
    Returns:
         (AgentBase) An instance of the Agent
    """
    # return DummyAgent(env_info, **kwargs)
    # return StaticDummyAgent(env_info, **kwargs)
    return moveForwardAgent(env_info, **kwargs)


class moveForwardAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.new_position = None
        self.episode = 1


    def reset(self):
        self.new_start = True
        self.new_position = None

    def draw_action(self, observation):
        
        if self.new_start:
            self.new_start = False
            
            if self.episode > 0:
                self.new_position = np.array([-1.15570723, 1.40024401, 1.44280414])
            else:
                self.new_position = self.get_joint_pos(observation)
                
            self.episode += 1
            
            
            print("\n newPos", self.new_position)
            print("Joint pos", self.get_joint_pos(observation))
            print("Joint vel", self.get_joint_vel(observation))
            print("Puck pos", self.get_puck_pos(observation))
            print("Puck vel", self.get_puck_vel(observation))
            print("Puck state", self.get_puck_state(observation))
            print("ee pos", self.get_ee_pose(observation))
            
            
            print(observation)


        velocity = np.zeros_like(self.new_position)
        action = np.vstack([self.new_position, velocity])

        return action
    
     
# class StaticDummyAgent(AgentBase):
#     def __init__(self, env_info, **kwargs):
#         super().__init__(env_info, **kwargs)
#         self.new_start = True
#         self.hold_position = None

#     def reset(self):
#         self.new_start = True
#         self.hold_position = None

#     def draw_action(self, observation):
#         if self.new_start:
#             self.new_start = False
#             self.hold_position = self.get_joint_pos(observation)

#         velocity = np.zeros_like(self.hold_position)
#         action = np.vstack([self.hold_position, velocity])
#         return action

