##
# @file   ddpg_rm.py
# @author Yibo Lin
# @date   Mar 2018
#

import tensorflow as tf
import tensorflow.contrib as tc
from functools import reduce
from baselines import logger
from ddpg import DDPG, get_target_updates, normalize
from tf_jacobian_vector import *
from baselines.common.mpi_adam import MpiAdam
from IPython import embed

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
        natural_update_target = False, natural_critic_update = False):
 
        self.nat_update_target = natural_update_target
        self.nat_update_critic = natural_critic_update
        super(DDPG_RM, self).__init__(actor, critic, memory, observation_shape, action_shape, param_noise, action_noise,
                gamma, tau, normalize_returns, enable_popart, normalize_observations,
                batch_size, observation_range, action_range, return_range,
                adaptive_param_noise, adaptive_param_noise_policy_threshold,
                critic_l2_reg, actor_lr, critic_lr, clip_norm, reward_scale)

        # compute Jacobian 
        # setup J'Ju
        print('########### tau {0}, actor_lr {1}, critic_lr {2}'.format(tau, actor_lr, critic_lr))

        #set up natural gradient stuff for actor:
        self.flat_actor_trainable_vars = flatvar(self.actor.trainable_vars)
        flat_actor_shape = self.flat_actor_trainable_vars.get_shape().as_list()[1]
        self.u_right = tf.placeholder(shape=[flat_actor_shape], dtype=tf.float32, name="u_right")

        self.mean_actor_tf_p_actor_tf = tf.reduce_mean(tf.reduce_sum(tf.square(self.actor_tf),axis=1))   # define g(theta) = (\sum_{i=1}^N pi(x_i;\theta)'\pi(x_i;\theta))/N
        tmp_1 = U.flatgrad(loss = self.mean_actor_tf_p_actor_tf, var_list = self.actor.trainable_vars)
        tmp_1_u = tf.matmul(tf.reshape(tmp_1,(1,-1)), tf.reshape(self.u_right, (-1,1)))
        self.Hu = U.flatgrad(loss = tmp_1_u, var_list = self.actor.trainable_vars)
        #self.Hu = hessian_vector_product(y = self.mean_actor_tf_p_actor_tf, xs = self.actor.trainable_vars, u = self.u_right) # \nabla^2 g(\theta) u
        #self.actor_f_Ax = lambda u: Hv_product(self.Hu, self.flat_actor_trainable_vars, self.u_right,
        #                                      self.flat_actor_trainable_vars.eval(session=self.sess), u, self.sess)

        #minus something to recover the (1/N)\sum J'J:
        tmp_g_theta_one = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.actor_tf, self.actions),axis=1)) #scalar
        tmp_g_theta_two = U.flatgrad(loss = tmp_g_theta_one, var_list=self.actor.trainable_vars) #d-vector
        tmp_g_theta_three = tf.matmul(tf.reshape(tmp_g_theta_two, (1,-1)), tf.reshape(self.u_right, (-1,1))) #scalar
        self.Hhat_u = U.flatgrad(loss = tmp_g_theta_three, var_list=self.actor.trainable_vars) #d-vector

        if self.nat_update_critic is True:
            self.flat_critic_trainable_vars = flatvar(self.critic.trainable_vars)
            flat_critic_shape = self.flat_critic_trainable_vars.get_shape().as_list()[1]
            self.u_right_c = tf.placeholder(shape = [flat_critic_shape], dtype=tf.float32, name='u_right_critic')
            self.mean_critic_square = tf.reduce_mean(tf.square(self.normalized_critic_tf))
            self.Hu_c = hessian_vector_product(y = self.mean_critic_square, xs = self.critic.trainable_vars, u = self.u_right_c)
            #self.critic_f_Ax = lambda u: Hv_product(self.Hu_c, self.flat_critic_trainable_vars, self.u_right_c,
            #                                self.flat_critic_trainable_vars.eval(session=self.sess), u, self.sess)
            self.critic_values = tf.placeholder(tf.float32, shape=(None, 1), name='critic_val')
            c_tmp_g_theta_one = tf.reduce_mean(tf.multiply(self.normalized_critic_tf, self.critic_values))
            c_tmp_g_theta_two = U.flatgrad(loss = c_tmp_g_theta_one, var_list=self.critic.trainable_vars)
            c_tmp_g_theta_three = tf.matmul(tf.reshape(c_tmp_g_theta_two,(1,-1)), tf.reshape(self.u_right_c,(-1,1)))
            self.Hhatu_c = U.flatgrad(loss = c_tmp_g_theta_three, var_list = tf.critic.trainable_vars)



        #do the same thing for target_actor and target_critic:
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


    def setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        if self.nat_update_critic is True:
            self.critic_optimizer = PlainDescent(var_list = self.critic.trainable_vars)
        else:
            self.critic_optimizer = MpiAdam(var_list=self.critic.trainable_vars,
                                                        beta1=0.9, beta2=0.999, epsilon=1e-08)



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
        random_rows = np.random.choice(batch['obs0'].shape[0], int(batch['obs0'].shape[0]*0.1), replace = False) #random choose half data from the mini-batch for jacobians
        #actor update:
        real_actions_from_curr_actor = self.sess.run(self.actor_tf, feed_dict={self.obs0:batch['obs0'][random_rows]})
        def actor_f_Ax(u):
            real_Hu = self.sess.run(self.Hu, feed_dict={self.obs0:batch['obs0'][random_rows], self.u_right:u})
            #real_Hhatu = self.sess.run(self.Hhat_u, feed_dict={self.obs0:batch['obs0'][random_rows], self.actions:real_actions_from_curr_actor, self.u_right:u})
            return real_Hu #- real_Hhatu

        # ==== compute natural gradient 
        actor_grads_natural = cg(actor_f_Ax, actor_grads, cg_iters=10)
        # use actor_lr for delta  
        #print (self.actor_lr / (actor_grads_natural.dot(actor_f_Ax(actor_grads_natural))))
        if actor_grads.dot(actor_grads_natural) < 0:
            pass
            #print ("non psd")
            #self.actor_optimizer.update(actor_grads, stepsize = self.actor_lr*1e-1)
        else:
            #print ('psd')
            mu = np.min([np.sqrt(self.actor_lr / (actor_grads.dot(actor_grads_natural) + np.finfo(np.float32).eps)),1.]) # avoid zero division 
            self.actor_optimizer.update(actor_grads_natural, stepsize=mu)

        #critic update:
        if self.nat_update_critic is False:
            critic_mu = self.critic_lr
            self.critic_optimizer.update(critic_grads, stepsize=critic_mu)
        else: #natural update for critic as well:
            real_norm_critic_values = self.sess.run(self.normalized_critic_tf, feed_dict={self.obs0:batch['obs0'][random_rows], self.actions:batch['actions'][random_rows]})
            def critic_f_Ax(u):
                real_Hu = self.sess.run(self.Hu_c, feed_dict={self.obs0:batch['obs0'][random_rows], self.actions:batch['actions'][random_rows], self.u_right_c:u})
                real_Hhatu = self.sess.run(self.Hhatu_c, feed_dict = {self.obs0:batch['obs0'][random_rows], self.actions:batch['actions'][random_rows], self.critic_values:real_norm_critic_values, self.u_right_c:u})
                return real_Hu - real_Hhatu
            # ==== compute natural gradient
            critic_grads_natural = cg(critic_f_Ax, critic_grads, cg_iters = 10)
            critic_mu = np.min([np.sqrt(self.critic_lr / (critic_grads.dot(critic_grads_natural) + np.finfo(np.float32).eps)),1.])
            self.critic_optimizer.update(critic_grads_natural, stepsize=critic_mu)

        #update target_actor:
        if self.nat_update_target is True:
            target_actor_J_diff = self.sess.run(self.target_actor_J_diff, feed_dict={self.obs0:batch['obs0']})
            target_actor_J_diff_natural = cg(self.target_actor_f_Ax, target_actor_J_diff, cg_iters = 10) # ~= (J'J)^{-1} J' (\pi - pi_target)
            target_mu = np.min([np.sqrt(self.tau / (target_actor_J_diff.dot(target_actor_J_diff_natural)+np.finfo(np.float32).eps)),1])
            self.target_actor_optimizer.update(-target_actor_J_diff_natural, stepsize = target_mu)  #target_theta = target_theta + mu * (J'J)^{-1}J' (pi - pi_targe) , so it's like doing ascent

        return critic_loss, actor_loss
