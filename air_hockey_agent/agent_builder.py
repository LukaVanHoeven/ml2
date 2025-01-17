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
import torch

from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.utils import inverse_kinematics, world_to_robot



def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.

@@ -9,5 +15,41 @@ def build_agent(env_info, **kwargs):
    Returns:
         (AgentBase) An instance of the Agent
    """
    # return DummyAgent(env_info, **kwargs)
    # return StaticDummyAgent(env_info, **kwargs)
    return myTestAgent(env_info, **kwargs)


class myTestAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.new_position = None
        self.episode = 1
        self.i = 0
        
        # Available Environments [3dof, 3dof-hit, 3dof-defend],
        # [7dof, 7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be released at the beginning of
        # the stage.
        self.env = AirHockeyChallengeWrapper("3dof-hit")
        
        # Print env name and env dict
        # print(self.env.env_name)
        # for k, v in self.env.env_info.items():
        #     print(k, v)
        
        # # Get the keys of the available constraint
        # print(env_info['constraints'].keys())                                
        
        # instantiate dict (only for readability/ease of use. Could've just used self.env.env_info["robot"]["joint_pos_limit"])
        self.robot_dict = self.env.env_info["robot"] 
        self.joint_pos_limit = self.robot_dict["joint_pos_limit"]

    def reset(self):
        self.new_start = True
        self.new_position = None

    def draw_action(self, observation):
        
        if self.new_start:
            self.new_start = False       
            
            # TODO: Currently working on getting the mallet to the puck
            puck_pos = self.get_puck_pos(observation)
            ee_pos = self.get_ee_pose(observation) # End effector position is the mallet pos + mallet orientation
            mallet_pos = ee_pos[0]
            
            target_position = np.array([puck_pos[0], puck_pos[1], 1.00000000e-01])
            
            direction = target_position - mallet_pos
            # We only want the direction, not the magnitude. So we normalize the vector
            direction_unit = direction / np.linalg.norm(direction)  # Normalize direction            

            # # TODO: Implement step size adjustment based on constraint violations
            # # TODO: Change step size, check hitting_agent, check constraints, check how draw_action is called. Problem: Currently, the mallet moves once towards a position, but the position is also off.
            # step_size = 0.5
            # step_position = mallet_pos + direction_unit * step_size
            
            
            # The inverse_kinematics function is a method for calculating the joint angles
            # of the robot arm that will position its end-effector at a desired position
            # success, joint_positions = inverse_kinematics(self.robot_model, self.robot_data, step_position) 
            
            # Get the joint position and velocity from the observation
            joint_pos_ids = observation[self.env_info['joint_pos_ids']]
            joint_vel_ids = observation[self.env_info['joint_vel_ids']]
            
            print("mall_pos", mallet_pos)
            print("puck_pos", puck_pos)
            
            self.ee_height = self.env_info['robot']["ee_desired_height"]
            print("ee_height", self.ee_height)
            print(self.env_info['robot'])
            # print("joint_pos_ids", q)
            # print("joint_vel_ids", dq)
            
            
            print("joint_pos", joint_pos_ids)
            print("Get Joint pos2", self.get_joint_pos(observation))
            print("joint_vel", joint_vel_ids)
            print("Get Joint vel", self.get_joint_vel(observation))
            
            #TODO VIOLATIONS
            # constraint_violations = self.env_info['constraints'].fun(joint_positions, dq)
            # print("constraint_violations", constraint_violations)
            
            # print("constraint_violations", constraint_violations)
            # if np.any(constraint_violations > 0):
            #     print("Constraint violated!")
            #     # Handle violation (e.g., adjust step size or target position)

            
            # # Get a dictionary of the constraint functions {"constraint_name": ndarray}
            # c = self.env_info['constraints'].fun(q, dq)
            
            # jac = self.env_info['constraints'].jacobian(q, dq)
            
            # # Get value of the constraint function by name
            # c_ee = self.env_info['constraints'].get('ee_constr').fun(q, dq)

            # # Get jacobian of the constraint function by name
            # jac_vel = self.env_info['constraints'].get('joint_vel_constr').jacobian(q, dq)      
            # print("jac_vel", jac_vel)
            
            # Joint pos limits:
            # [-2.96705973 -1.8        -2.0943951 ]
            # [ 2.96705973  1.8         2.0943951 ]
            
            
            # joint_pos_constr: 
            # [-3.97441397, -0.40975599, -0.54687121,
            # -1.66299951, -3.01024401, -3.43247949]
            
            # Start position of the robot
            # [-1.15570723, 1.30024401, 1.44280414]
            
            if self.episode > 1:
                self.new_position = np.array([-1.15570723, 1.30024401, 1.44280414]) # Start position of the robot
                # self.new_position = joint_positions
            else:
                self.new_position = np.array([-1.15570723, 1.30024401, -1.44280414]) # Start position of the robot
                # self.new_position = joint_positions
                # self.new_position = target_position # Start position of the robot
                # self.new_position = self.get_joint_pos(observation)
                
            self.episode += 1
            print("BBBBBBBBBBBBBb", self.episode)      
            
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
        
        if self.i < 5:
            print(self.i, "action", action)
            self.i += 1
            
        # print("action", action)

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




##########################################################################################################################################################
##################### CODE BELOW FROM https://air-hockey-challenges-docs.readthedocs.io/en/latest/agent.html to save agent variables #####################
##########################################################################################################################################################

# class DummyAgent(AgentBase):
#     def __init__(self, env_info, value, **kwargs):
#         super().__init__(env_info, **kwargs)
#         self.new_start = True
#         self.hold_position = None

#         self.primitive_variable = value  # Primitive python variable
#         self.numpy_vector = np.array([1, 2, 3]) * value  # Numpy array
#         self.list_variable = [1, 'list', [2, 3]]  # Numpy array

#         # Dictionary
#         self.dictionary = dict(some='random', keywords=2, fill='the dictionary')

#         # Building a torch object
#         data_array = np.ones(3) * value
#         data_tensor = torch.from_numpy(data_array)
#         self.torch_object = torch.nn.Parameter(data_tensor)

#         # A non serializable object
#         self.object_instance = object()

#         # A variable that is not important e.g. a buffer
#         self.not_important = np.zeros(10000)

#         # Here we specify how to save each component
#         self._add_save_attr(
#             primitive_variable='primitive',
#             numpy_vector='numpy',
#             list_variable='primitive',
#             dictionary='pickle',
#             torch_object='torch',
#             object_instance='none',
#             # The '!' is to specify that we save the variable only if full_save is True
#             not_important='numpy!',
#         )

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


# if __name__ == '__main__':
#     env = AirHockeyChallengeWrapper("3dof-hit")

#     # Construct Agent
#     args = {'value': 1.1}
#     agent_save = build_agent(env.env_info, **args)

#     print("######################################################")
#     print("Save Agent Variables")
#     print("######################################################")
#     print("agent_save.primitive_variable: ", agent_save.primitive_variable)
#     print("agent_save.numpy_vector: ", agent_save.numpy_vector)
#     print("agent_save.list_variable: ", agent_save.list_variable)
#     print("agent_save.dictionary: ", agent_save.dictionary)
#     print("agent_save.torch_object: ", agent_save.torch_object)

#     # The not_important variable will not be saved unless the full_save is set True
#     agent_save.save("agent.msh", full_save=False)

#     agent_load = DummyAgent.load_agent("agent.msh", env.env_info)
#     print("######################################################")
#     print("Load the Agent")
#     print("######################################################")
#     print("agent_load.primitive_variable: ", agent_load.primitive_variable)
#     print("agent_load.numpy_vector: ", agent_load.numpy_vector)
#     print("agent_load.list_variable: ", agent_load.list_variable)
#     print("agent_load.dictionary: ", agent_load.dictionary)
#     print("agent_load.torch_object: ", agent_load.torch_object)
#     print("agent_load.object_instance: ", agent_load.object_instance)

#     print("------------------------------------------------------")
#     print("These variable will not be saved while full_save is False")
#     print("agent_load.not_important: ", agent_load.not_important)

#     print("------------------------------------------------------")
#     print("These variable will be parsed from env_info:")
#     print("agent_load.env_info.keys()s: ", agent_load.env_info.keys())
#     print("agent_load.agent_id: ", agent_load.agent_id)
#     print("agent_load.robot_model: ", agent_load.robot_model)
#     print("agent_load.robot_data: ", agent_load.robot_data)
