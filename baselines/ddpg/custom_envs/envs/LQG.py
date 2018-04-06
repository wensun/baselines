##
# @file   LQG.py
# @author Yibo Lin
# @date   Apr 2018
#

import numpy as np
import gym 
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register

#class LinearQuadGausEnv(mujoco_env.MujocoEnv, utils.EzPickle):
class LQGEnv(gym.Env):

    def __init__(self):
        #utils.EzPickle.__init__(self)
        #mujoco_env.MujocoEnv.__init__(self, 'LQG.xml', 2)

        self.A = np.zeros((3,3))
        self.A[0,0] = 1.01
        self.A[0,1] = 0.01
        self.A[1,0] = 0.01
        self.A[1,1] = 1.01
        self.A[1,2] = 0.01
        self.A[2,1] = 0.01
        self.A[2,2] = 1.01

        self.B = np.eye(3)
        self.Q = 1e-3*np.eye(3)
        self.R = np.eye(3)

        self.x_dim = 3
        self.a_dim = 3

        self.observation_space = np.array(np.ones([self.x_dim]))
        self.action_space = np.array(np.ones([self.a_dim]))

        self.init_state_mean = np.ones(self.x_dim)*5
        self.init_state_cov = np.eye(self.x_dim)*0.1

        self.state = None

        self.noise_cov = np.eye(self.x_dim)*0.001 

        print("==== LQG parameters ====")
        print("A = ", self.A)
        print("B = ", self.B)
        print("Q = ", self.Q)
        print("noise_cov = ", self.noise_cov)

    def reset(self):
        self.state = np.random.multivariate_normal(mean = self.init_state_mean, cov = self.init_state_cov)
        return self.state

    def step(self, a): 
        cost = self.state.dot(self.Q).dot(self.state) + a.dot(self.R).dot(a)
        next_state = self.A.dot(self.state.reshape((self.x_dim, 1))) + self.B.dot(a.reshape((self.a_dim, 1)));
        next_state = next_state.reshape(self.x_dim)
        self.state = next_state + np.random.multivariate_normal(mean = np.zeros(self.x_dim), cov = self.noise_cov)

        return self.state, -cost, False, None

    def seed(self, seed):
        np.random.seed(seed)

    def render(self, mode='human', close=False):
        pass 
