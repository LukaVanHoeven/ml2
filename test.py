from air_hockey_challenge.framework import AirHockeyChallengeWrapper

# Available Environments [3dof, 3dof-hit, 3dof-defend],
# [7dof, 7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be released at the beginning of
# the stage.
env = AirHockeyChallengeWrapper("3dof-hit")

print(env.env_info)
