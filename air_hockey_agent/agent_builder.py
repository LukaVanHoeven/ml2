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
from air_hockey_challenge.framework import AirHockeyChallengeWrapper


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
        
        # Available Environments [3dof, 3dof-hit, 3dof-defend],
        # [7dof, 7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be released at the beginning of
        # the stage.
        self.env = AirHockeyChallengeWrapper("3dof-hit")
        
        # Print env name and env dict
        print(self.env.env_name)
        for k, v in self.env.env_info.items():
            print(k, v)
        
        # # Get the keys of the available constraint
        # print(env_info['constraints'].keys())                      
            
            
            
            
        
        # instantiate dict (only for readability/ease of use. Could've just used self.env.env_info["robot"]["joint_pos_limit"])
        self.robot_dict = self.env.env_info["robot"] 
        self.joint_pos_limit = self.robot_dict["joint_pos_limit"]
        
        self.lower_joint_pos_limit = self.robot_dict["joint_pos_limit"][0]
        self.upper_joint_pos_limit = self.robot_dict["joint_pos_limit"][1]
        # print(self.lower_joint_pos_limit) # [-2.96705973 -1.8        -2.0943951 ]

    def reset(self):
        self.new_start = True
        self.new_position = None

    def draw_action(self, observation):
                
                
        
        # joint_pos_limit = self.env.env_info["joint_pos_limit"]
        # print(joint_pos_limit)
        
        
        if self.new_start:
            self.new_start = False
            
            
            # Get the joint position and velocity from the observation
            q = observation[self.env_info['joint_pos_ids']]
            dq = observation[self.env_info['joint_vel_ids']]
            
            print("AAAAAAAAAAAA")
            print(q)
            print(dq)
            
            # Get a dictionary of the constraint functions {"constraint_name": ndarray}
            c = self.env_info['constraints'].fun(q, dq)
            print(c)
            
            # TODO: Why is there a difference between pos_limit and pos_constraint? And how to properly read x,y value coordinates???
            
            if self.episode > 1:
                self.new_position = self.lower_joint_pos_limit
                # self.new_position = np.array([-1.15570723, 1.30024401, 1.44280414]) # Start position of the robot
            else:
                self.new_position = self.upper_joint_pos_limit
                # self.new_position = self.get_joint_pos(observation)
                
            self.episode += 1
            
            
            
            # print("\n newPos", self.new_position)
            # print("Joint pos", self.get_joint_pos(observation))
            # print("Joint vel", self.get_joint_vel(observation))
            # print("Puck pos", self.get_puck_pos(observation))
            # print("Puck vel", self.get_puck_vel(observation))
            # print("Puck state", self.get_puck_state(observation))
            # print("ee pos", self.get_ee_pose(observation))
            # print("obs", observation)
            
            # print("env_info", self.env_info)


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

