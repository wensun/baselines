##
# @file   ddpg_rm.py
# @author Yibo Lin
# @date   Mar 2018
#

import tensorflow as tf
from functools import reduce
from baselines import logger
from ddpg import DDPG, get_target_updates
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
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., 
        natural_update_target = False):
 
        self.nat_update_target = natural_update_target

        super(DDPG_RM, self).__init__(actor, critic, memory, observation_shape, action_shape, param_noise, action_noise,
                gamma, tau, normalize_returns, enable_popart, normalize_observations,
                batch_size, observation_range, action_range, return_range,
                adaptive_param_noise, adaptive_param_noise_policy_threshold,
                critic_l2_reg, actor_lr, critic_lr, clip_norm, reward_scale)

        # compute Jacobian 
        # setup J'Ju
        print('########### tau {0}, actor_lr {1}, critic_lr {2}'.format(tau, actor_lr, critic_lr))

        self.flat_actor_trainable_vars = flatvar(self.actor.trainable_vars)
        flat_actor_shape = self.flat_actor_trainable_vars.get_shape().as_list()[1]
        self.u_right = tf.placeholder(shape=[flat_actor_shape], dtype=tf.float32, name="u_right")
        #self.JpJu = Jacobian_p_Jacobian_u_product(self.actor_loss, [self.flat_actor_trainable_vars], self.u_right)

        #instead, we setup hessian vector product here:
        self.mean_actor_tf_p_actor_tf = tf.reduce_mean(tf.reduce_sum(tf.square(self.actor_tf),axis=1))   # define g(theta) = (\sum_{i=1}^N pi(x_i;\theta)'\pi(x_i;\theta))/N
        self.Hu = hessian_vector_product(y = self.mean_actor_tf_p_actor_tf, xs = [self.flat_actor_trainable_vars], u = self.u_right) # \nabla^2 g(\theta) u
        # function, given vector u, compute J' J u
        #self.actor_f_Ax = lambda u: JpJu_product(self.JpJu, self.flat_actor_trainable_vars, self.u_right, self.flat_actor_trainable_vars.eval(session=self.sess), u, self.sess)
        # function, given vector u, compute H u
        self.actor_f_Ax = lambda u: Hv_product(self.Hu, self.flat_actor_trainable_vars, self.u_right, 
                                              self.flat_actor_trainable_vars.eval(session=self.sess), u, self.sess)

        #do the same thing for target_actor:
        if self.nat_update_target is True:
            self.flat_target_actor_trainable_vars = flatvar(self.target_actor.trainable_vars)
            self.mean_target_actor_tf_p_target_actor_tf = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_actor_tf),axis=1)) #define g(\theta') = \sum_{i=1}^N pi(x_i;\theta')^T \pi(x_i;\theta')/N
            self.targetHv = hessian_vector_product(self.mean_target_actor_tf_p_target_actor_tf, [self.flat_target_actor_trainable_vars], u = self.u_right) #\nabla^2 g(\theta') u
            self.target_actor_f_Ax = lambda u: Hv_product(self.targetHv, self.flat_target_actor_trainable_vars, 
                    self.u_right, self.flat_target_actor_trainable_vars.eval(session=self.sess), u, self.sess) 

            actor_minus_target_actor = self.actor_tf - tf.scalar_mul(0.5,self.target_actor_tf) #diff = pi(x_i;\theta) - 0.5pi(x_i;\theta')
            target_actor_p_diff = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.target_actor_tf, actor_minus_target_actor),axis=1)) # G(theta') = (1/N)\sum_{i=1}^N pi(x_i;\theta')^T (pi(x_i;\theta) - 0.5\pi(x_i;\theta'))
            self.target_actor_J_diff = flatgrad(target_actor_p_diff, [self.flat_target_actor_trainable_vars])   #nabla G(\theta') = (1/N)sum_{i=1}^N J_{theta'}(x_i)'(\pi(x_i;\theta) - \pi(x_i;\theta')), [d]
            self.target_actor_optimizer = PlainDescent(var_list = self.target_actor.trainable_vars)

        #print("actor.trainable_vars = ", self.actor.trainable_vars)
        #print("u_right = ", self.u_right)
        #print ("Hv = ", self.Hu)
        #print("actor_f_Ax", self.actor_f_Ax)
        #print("JpJu = ", self.JpJu)
        #print("actor_f_Ax = ", self.actor_f_Ax)

    def setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = PlainDescent(var_list=self.actor.trainable_vars)


    def setup_target_network_updates(self):
        print('Set up target updates ####################')
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        if self.nat_update_target is True:
            self.target_soft_updates = [critic_soft_updates]
        else:
            self.target_soft_updates = [actor_soft_updates, critic_soft_updates]


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
        actor_grads_natural = cg(self.actor_f_Ax, actor_grads, cg_iters=10)
        # use actor_lr for delta  
        mu = np.sqrt(self.actor_lr / (actor_grads.dot(actor_grads_natural) + np.finfo(np.float32).eps)) # avoid zero division 
        # ====
        self.actor_optimizer.update(actor_grads_natural, stepsize=mu)
        self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr)

        #update target_actor:
        if self.nat_update_target is True:
            target_actor_J_diff = self.sess.run(self.target_actor_J_diff, feed_dict={self.obs0:batch['obs0']})
            target_actor_J_diff_natural = cg(self.target_actor_f_Ax, target_actor_J_diff, cg_iters = 10) # ~= (J'J)^{-1} J' (\pi - pi_target)
            target_mu = np.sqrt(self.tau / (target_actor_J_diff.dot(target_actor_J_diff_natural)+np.finfo(np.float32).eps))
            self.target_actor_optimizer.update(-target_actor_J_diff_natural, stepsize = target_mu)  #target_theta = target_theta + mu * (J'J)^{-1}J' (pi - pi_targe) , so it's like doing ascent

        return critic_loss, actor_loss
