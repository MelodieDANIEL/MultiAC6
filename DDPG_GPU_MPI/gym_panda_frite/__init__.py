import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='PandaFriteEnvComplete-v0',
    entry_point='gym_panda_frite.envs:PandaFriteEnvComplete',
    kwargs={'database': None, 'json_decoder' : None, 'env_pybullet': None, 'gui': None, 'env_rank': None}
)

register(
    id='PandaFriteEnvRotationGripper-v0',
    entry_point='gym_panda_frite.envs:PandaFriteEnvRotationGripper',
    kwargs={'database': None, 'json_decoder' : None, 'env_rank': None}
)
