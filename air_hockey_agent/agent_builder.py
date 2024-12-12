import numpy as np
import mujoco as mujoco

from air_hockey_challenge.framework import AgentBase


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
    # return DummyAgent(env_info, **kwargs)
    return CircularAgent(env_info, **kwargs)


class CircularAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.radius = 0.1  # Radius of the circle
        self.center = np.array([0.5, 0.5, 0.0])  # Center of the circle in 3D space
        self.speed = 0.05  # Speed of movement (radians per timestep)
        self.angle = 0.0  # Initial angle in radians
        self.num_joints = 3  # The environment expects 3 joints

    def reset(self):
        # Reset the angle at the start of each episode
        self.angle = 0.0

    def draw_action(self, observation):
        # Calculate the next position on the circle (2D plane)
        x = self.center[0] + self.radius * np.cos(self.angle)
        y = self.center[1] + self.radius * np.sin(self.angle)
        z = self.center[2]  # Keep the third dimension constant for now

        # Update the angle for the next step
        self.angle += self.speed
        self.angle = self.angle % (2 * np.pi)  # Keep the angle within [0, 2π]

        # Desired position for 3 joints (add z dimension)
        desired_position = np.array([x, y, z])

        # Desired velocity (set to zero for all joints)
        desired_velocity = np.zeros_like(desired_position)

        # Combine position and velocity into the required shape (2, 3)
        action = np.vstack([desired_position, desired_velocity])
        return action


