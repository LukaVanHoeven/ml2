from air_hockey_challenge.utils import inverse_kinematics, world_to_robot, robot_to_world
import numpy as np

class Reward:
    def __init__(self):
        self.has_hit = False
    def setAgent(self, agent):
        self.agent = agent

    def custom_reward(self, env, state, action, next_state, absorbing):
        r = 0
        puck_pos, puck_vel = env.get_puck(next_state)
        puck_pos_world, _ = robot_to_world(env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        
        if not self.has_hit:
            self.has_hit = self._has_hit(env, state)

        if absorbing or env._data.time < env.env_info["dt"] * 2:
            # If the hit scores
            if (
                (puck_pos[0] - env.env_info["table"]["length"] / 2)
                > 0
                > (np.abs(puck_pos[1]) - env.env_info["table"]["goal_width"] / 2)
            ):
                r = 10000
            self.has_hit = False
            return r
        else:
            # If the puck has not yet been hit, encourage the robot to get closer to the puck
            if not self.has_hit:
                ee_pos = self.agent.get_ee_pose(state)[0][:2]  # Get end-effector (x, y)

                dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos)  # Euclidean distance

                #vec_ee_puck = (puck_pos - ee_pos) / (dist_ee_puck + 1e-6)  # Normalized direction vector
                # Encourage movement towards the puck
                r = -dist_ee_puck
        
        return r
    
    def _has_hit(self, mdp, state):
        ee_pos, ee_vel = mdp.get_ee()
        puck_cur_pos, _ = mdp.get_puck(state)
        if (
            np.linalg.norm(ee_pos[:2] - puck_cur_pos[:2])
            < mdp.env_info["puck"]["radius"] + mdp.env_info["mallet"]["radius"] + 5e-3
        ):
            return True
        else:
            return False
