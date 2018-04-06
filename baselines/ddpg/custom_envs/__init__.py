##
# @file   __init__.py
# @author Yibo Lin
# @date   Apr 2018
#

from gym.envs.registration import register

register(
    id='LQG-v2',
    entry_point='custom_envs.envs:LQGEnv',
)
