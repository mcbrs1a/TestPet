#__init__.py
from gym.envs.registration import register

register(
    id='PET-v0',
    entry_point='gym_PET.envs:PETEnv',
)

#register(
#    id='PET-extrahard-v0',
#    entry_point='gym_PET.envs:PETExtraHardEnv',
#)
