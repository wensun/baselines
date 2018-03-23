##
# @file   ddpg_rm.py
# @author Yibo Lin
# @date   Mar 2018
#

import tensorflow as tf
from ddpg import DDPG
from tf_jacobian_vector import *

def flatvar(var_list):
    vars = tf.concat(axis=0, values=[
        tf.reshape(v if v is not None else tf.zeros_like(v), [numel(v)])
        for v in var_list
    ])

    return tf.reshape(vars, [1, numel(vars)])

def flatshape(var_list): 
    num = 0
    for var in var_list:
        num += numel(var)
    return num 

class DDPG_RM(DDPG):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        adaptive_param_noise=True, adaptive_param_noise_policy_threshold=.1,
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):

        super(DDPG_RM, self).__init__(actor, critic, memory, observation_shape, action_shape, param_noise, action_noise,
                gamma, tau, normalize_returns, enable_popart, normalize_observations,
                batch_size, observation_range, action_range, return_range,
                adaptive_param_noise, adaptive_param_noise_policy_threshold,
                critic_l2_reg, actor_lr, critic_lr, clip_norm, reward_scale)

        # compute Jacobian 
        # setup J'Ju
        self.flat_actor_trainable_vars = flatvar(self.actor.trainable_vars)
        flat_actor_shape = self.flat_actor_trainable_vars.get_shape().as_list()[1]
        self.u_right = tf.placeholder(shape=[flat_actor_shape], dtype=tf.float32, name="u_right")
        self.JpJu = Jacobian_p_Jacobian_u_product(self.actor_loss, [self.flat_actor_trainable_vars], self.u_right)
        self.actor_grads_natural = tf.placeholder(shape=[flat_actor_shape], dtype=tf.float32, name="actor_grads_natural")
        # function, given vector u, compute J' J u
        self.actor_f_Ax = lambda u: JpJu_product(self.JpJu, self.flat_actor_trainable_vars, self.u_right, self.flat_actor_trainable_vars.eval(session=self.sess), u, self.sess)
        print("actor.trainable_vars = ", self.actor.trainable_vars)
        print("u_right = ", self.u_right)
        print("JpJu = ", self.JpJu)
        print("actor_grads_natural = ", self.actor_grads_natural)
        print("actor_f_Ax = ", self.actor_f_Ax)

    def train(self):
        # Get a batch.
        batch = self.memory.sample(batch_size=self.batch_size)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_Q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_Q], feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })
            self.ret_rms.update(target_Q.flatten())
            self.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                self.old_std : np.array([old_std]),
                self.old_mean : np.array([old_mean]),
            })

            # Run sanity check. Disabled by default since it slows down things considerably.
            # print('running sanity check')
            # target_Q_new, new_mean, new_std = self.sess.run([self.target_Q, self.ret_rms.mean, self.ret_rms.std], feed_dict={
            #     self.obs1: batch['obs1'],
            #     self.rewards: batch['rewards'],
            #     self.terminals1: batch['terminals1'].astype('float32'),
            # })
            # print(target_Q_new, target_Q, new_mean, new_std)
            # assert (np.abs(target_Q - target_Q_new) < 1e-3).all()
        else:
            target_Q = self.sess.run(self.target_Q, feed_dict={
                self.obs1: batch['obs1'],
                self.rewards: batch['rewards'],
                self.terminals1: batch['terminals1'].astype('float32'),
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
            self.obs0: batch['obs0'],
            self.actions: batch['actions'],
            self.critic_target: target_Q,
        })
        # ==== compute natural gradient 
        actor_grads_natural = cg(self.actor_f_Ax, actor_grads)
        # ====
        self.actor_optimizer.update(actor_grads_natural, stepsize=self.actor_lr)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        return critic_loss, actor_loss
