import numpy as np

#from air_hockey_challenge.framework import AgentBase

import threading
import time
from scipy.linalg import lstsq

import numpy as np
from scipy.interpolate import CubicSpline

from air_hockey_challenge.framework.agent_base import AgentBase
#from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.utils import inverse_kinematics, world_to_robot, robot_to_world
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer, PuckTracker
import gym

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
    #print(env_info)
    return DreamerV3HittingAgent(env_info, **kwargs)


class DummyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        #print(f"observation in drawaction {observation}")
        velocity = np.zeros_like(self.hold_position)
        #action = np.vstack([self.hold_position, velocity])
        action = np.append(self.hold_position, velocity)
        return action


class DreamerV3HittingAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None
        #self.agent = agent
        self.last_velocity = np.zeros(3)
        self.q_lower, self.q_upper = self.env_info['robot']['joint_pos_limit']
        self.dq_lower, self.dq_upper = self.env_info['robot']['joint_vel_limit']
        self.ddq_lower, self.ddq_upper = self.env_info['robot']['joint_acc_limit']
        self.dt = 1 / self.env_info['robot']['control_frequency']

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation, *args, **kwargs):
        """
        obs = self.agent.wm.preprocess(observation)
        embed = self.agent._wm.encoder(obs)
        latent, _ = self.wm.dynamics.obs_step(None, None, embed, obs["is_first"])
        feat = self.wm.dynamics.get_feat(latent)

        # Get action from Dreamer's policy
        actor = self.task_behavior.actor(feat)
        action = actor.sample().detach().cpu().numpy()
        """
        is_training = kwargs.get("is_training", True)  
        if is_training:
            return self.process_action(observation)
        else:
            #print(f"observation in drawaction {observation}")
            velocity = np.zeros_like(self.hold_position)
            #action = np.vstack([self.hold_position, velocity])
            action = np.append(self.hold_position, velocity)
        return action

        # Map action to [position, velocity]
        #position = action[:len(action)//2]
        #velocity = action[len(action)//2:]
    
    def process_action(self, action):
        #joint_position_constraints = self.env_info['constraints'].get('joint_pos_constr').fun(q, dq)
        # Get static joint limits
        """
        print(f"puck_pos_ids: {self.env_info['puck_pos_ids']}")
        print(f"puck_vel_ids: {self.env_info['puck_vel_ids']}")
        print(f"joint_pos_ids: {self.env_info['joint_pos_ids']}")
        print(f"joint_vel_ids: {self.env_info['joint_vel_ids']}")
        """

        q_cmd = self._unnormalize_value(action[:3], self.q_lower, self.q_upper)
        dq_cmd = self._unnormalize_value(action[3:6], self.dq_lower, self.dq_upper)

        q_cmd = np.clip(q_cmd, self.q_lower, self.q_upper)
        dq_cmd = np.clip(dq_cmd, self.dq_lower, self.dq_upper)

        ddq_cmd = (dq_cmd - self.last_velocity) / self.dt   # Compute acceleration
        new_ddq_cmd = np.clip(ddq_cmd, self.ddq_lower, self.ddq_upper)  # Clip acceleration

        # Only modify dq_cmd if clipping actually changed acceleration
        #if not np.allclose(new_ddq_cmd, ddq_cmd):  # Check if clipping changed ddq_cmd
            #dq_cmd = self.last_velocity + new_ddq_cmd * self.dt
            

        """
        # Apply runtime (dynamic) constraints if available
        if 'joint_pos_constr' in self.env_info['constraints'].keys():
            q_dyn = self.env_info['constraints'].get('joint_pos_constr').fun(q_cmd, dq_cmd)
            print(f"q_dyn: {q_dyn}")
            q_dyn_lower, q_dyn_upper = np.split(q_dyn, 2)
            q_cmd = np.clip(q_cmd, q_dyn_lower, q_dyn_upper)  # Apply dynamic position constraint
        
        if 'joint_vel_constr' in self.env_info['constraints'].keys():
            dq_dyn = self.env_info['constraints'].get('joint_vel_constr').fun(q_cmd, dq_cmd)
            dq_dyn_lower, dq_dyn_upper = np.split(dq_dyn, 2)
            print(f"dq_dyn: {dq_dyn}")
            dq_cmd = np.clip(dq_cmd, dq_dyn_lower, dq_dyn_upper)  # Apply dynamic position constraint

        constraints = self.env_info.get('constraints', {})

        constraint_violations = constraints.get('fun', lambda q, dq: {})(q_cmd, dq_cmd)

        if constraint_violations:
            print(f"Dynamic Constraint Violations Detected: {constraint_violations}")

            for name, violation in constraint_violations.items():
                if np.any(violation > 0):  # Constraint violated
                    print(f"Adjusting for {name} constraint...")

                    # Get the Jacobian for this constraint
                    J = constraints.get('jacobian', lambda q, dq: {})(q_cmd, dq_cmd).get(name)

                    if J is not None:
                        # Compute correction using pseudo-inverse
                        correction = -np.linalg.pinv(J) @ violation  # Moves state back inside constraint

                        # Apply correction to q_cmd and dq_cmd
                        q_cmd += correction * self.dt
                        dq_cmd += correction  

                        print(f"Applied correction for {name}: {correction}")
        """
            # 5️⃣ Retrieve and evaluate dynamic constraints (if available)
        """
        c = self.env_info['constraints'].fun(q_cmd, dq_cmd)
        jac = self.env_info['constraints'].jacobian(q_cmd, dq_cmd)

        c.popitem()
        jac.popitem()
        # 6️⃣ Apply corrections using Jacobian
        for name, violation in c.items():
            jacobian = jac.get(name)

            if jacobian.shape[1] > 3:  
                jacobian = jacobian[:, :3]  # Keep only the first 3 columns

            if np.any(violation > 0):  # Constraint is violated (positive means violation)
                print(f"Warning: {name} constraint violated! Applying correction.")

                if jacobian is not None:
                    correction = -np.linalg.pinv(jacobian) @ violation  # Compute correction
                    q_cmd += correction * self.dt  # Adjust position
                    dq_cmd += correction  # Adjust velocity

        c = self.env_info['constraints'].fun(q_cmd, dq_cmd)
        jac = self.env_info['constraints'].jacobian(q_cmd, dq_cmd)


        for name, violation in c.items():
            jacobian = jac.get(name)

            if jacobian.shape[1] > 3:  
                jacobian = jacobian[:, :3]  # Keep only the first 3 columns

            if np.any(violation > 0):  # Constraint is violated (positive means violation)
                print(f"Warning: {name} constraint violated AGAAINNNNASNDFNASDFN! Applying correction.")

                if jacobian is not None:
                    correction = -np.linalg.pinv(jacobian) @ violation  # Compute correction
                    q_cmd += correction * self.dt  # Adjust position
                    dq_cmd += correction  # Adjust velocity


        ee_constr = 'ee_constr'
            # 7️⃣ Enforce End-Effector Constraint (ee_constr)
        if ee_constr in self.env_info['constraints'].keys():
            ee_violation = self.env_info['constraints'].get('ee_constr').fun(q_cmd, dq_cmd)
            ee_jacobian = self.env_info['constraints'].get('ee_constr').jacobian(q_cmd, dq_cmd)

             # Fix incorrect Jacobian dimensions if necessary
            if ee_jacobian.shape[1] > 3:  
                ee_jacobian = ee_jacobian[:, :3]  # Keep only the first 3 columns


            if np.any(ee_violation > 0):  # If EE constraint is violated
                print("Warning: End-Effector constraint violated! Applying correction.")
                jac_inv = -np.linalg.pinv(ee_jacobian)
                correction = jac_inv @ ee_violation

                q_cmd += correction * self.dt  # Adjust position
                dq_cmd += correction  # Adjust velocity

        if ee_constr in self.env_info['constraints'].keys():
            ee_violation = self.env_info['constraints'].get('ee_constr').fun(q_cmd, dq_cmd)
            ee_jacobian = self.env_info['constraints'].get('ee_constr').jacobian(q_cmd, dq_cmd)

             # Fix incorrect Jacobian dimensions if necessary
            if ee_jacobian.shape[1] > 3:  
                ee_jacobian = ee_jacobian[:, :3]  # Keep only the first 3 columns


            if np.any(ee_violation > 0):  # If EE constraint is violated
                print("Warning: End-Effector constraint violated againnnn!!!!! Applying correction.")
                jac_inv = -np.linalg.pinv(ee_jacobian)
                correction = jac_inv @ ee_violation

                q_cmd += correction * self.dt  # Adjust position
                dq_cmd += correction  # Adjust velocity
        """
        
        MAX_ITERS = 10  # Prevent infinite loops
        for constraint_name in ['joint_pos_constr', 'joint_vel_constr', 'ee_constr']:
            if constraint_name in self.env_info['constraints'].keys():
                for i in range(MAX_ITERS):  # Try multiple correction steps
                    violation = self.env_info['constraints'].get(constraint_name).fun(q_cmd, dq_cmd)
                    jacobian = self.env_info['constraints'].get(constraint_name).jacobian(q_cmd, dq_cmd)

                    if jacobian.shape[1] > 3:  
                        jacobian = jacobian[:, :3]  # Keep only the first 3 columns

                    if np.linalg.cond(jacobian) > 1e3:
                        pass  
                        #print("Warning: Jacobian is poorly conditioned! Consider damping.")

                    if np.any(violation > 0):  
                        #print(f"Warning: {constraint_name} violated (iteration {i+1})! Applying correction.")

                        jac_inv = -np.linalg.pinv(jacobian)
                        correction = jac_inv @ violation
                        max_correction = 1.0  # Tune this
                        correction = np.clip(correction, -max_correction, max_correction)
                        # Apply correction
                        q_cmd += correction * self.dt  # Adjust position
                        dq_cmd += correction  # Adjust velocity
                    else:
                        break  # Exit loop if violation is gone
        
        # 7️⃣ Final safety clipping
        q_cmd = np.clip(q_cmd, self.q_lower, self.q_upper)
        dq_cmd = np.clip(dq_cmd, self.dq_lower, self.dq_upper)
        self.last_velocity = dq_cmd.copy()

        new_action = np.append(q_cmd, dq_cmd)
        return new_action



    def custom_reward(env, state, action, next_state, absorbing):
        puck_pos, puck_vel = env.get_puck(next_state)
        puck_pos_world, _ = robot_to_world(env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        
        # Reward for scoring a goal
        if puck_pos_world[0] > env.env_info['table']['length'] / 2 and \
        abs(puck_pos_world[1]) < env.env_info['table']['goal_width'] / 2:
            return 10  # Reward for scoring
        
        return 0
    
    def _unnormalize_value(self, value, low, high):
        return (high - low) * (value + 1) * 0.5 + low
    

def main():
    from air_hockey_agent.air_hockey_challenge_dreamer_wrapper import AirHockeyChallengeDreamerWrapper
    plot_trajectory = False
    env = AirHockeyChallengeWrapper(env="3dof-hit", interpolation_order=3, debug=False)
    
    agent = DummyAgent(env.base_env.env_info)

    
    obs = env.reset()
    agent.reset()

    steps = 0
    while True:
        steps += 1
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        print("State:", obs)
        print("Action:", action)
        print("rewards", reward)
        env.render()

        if done or steps > env.info.horizon:
            if plot_trajectory:
                import matplotlib.pyplot as plt
                trajectory_record = np.array(env.base_env.controller_record)
                nq = env.base_env.env_info['robot']['n_joints']

                fig, axes = plt.subplots(3, nq)
                for j in range(nq):
                    axes[0, j].plot(trajectory_record[:, j])
                    axes[0, j].plot(trajectory_record[:, j + nq])
                    axes[1, j].plot(trajectory_record[:, j + 2 * nq])
                    axes[1, j].plot(trajectory_record[:, j + 3 * nq])
                    axes[2, j].plot(trajectory_record[:, j + 4 * nq])
                    axes[2, j].plot(trajectory_record[:, j + nq] - trajectory_record[:, j])
                plt.show()

            steps = 0
            obs = env.reset()
            agent.reset()

def main2():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    plot_trajectory = False
    env = AirHockeyChallengeWrapper(env="3dof-hit", interpolation_order=3, debug=plot_trajectory)

    agent = HittingAgentExample(env.base_env.env_info)

    obs = env.reset()
    agent.reset()

    steps = 0
    while True:
        steps += 1
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        if steps > env.info.horizon:
            if plot_trajectory:
                import matplotlib.pyplot as plt
                trajectory_record = np.array(env.base_env.controller_record)
                nq = env.base_env.env_info['robot']['n_joints']

                fig, axes = plt.subplots(3, nq)
                for j in range(nq):
                    axes[0, j].plot(trajectory_record[:, j])
                    axes[0, j].plot(trajectory_record[:, j + nq])
                    axes[1, j].plot(trajectory_record[:, j + 2 * nq])
                    axes[1, j].plot(trajectory_record[:, j + 3 * nq])
                    axes[2, j].plot(trajectory_record[:, j + 4 * nq])
                    axes[2, j].plot(trajectory_record[:, j + nq] - trajectory_record[:, j])
                plt.show()

            steps = 0
            obs = env.reset()
            agent.reset()


if __name__ == '__main__':
    main()