import numpy as np
import scipy as sp
from IPython import embed
import argparse

class ADAM(object):
    def __init__(self, dx, lr = None):
        self.dx = dx
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m = np.zeros(self.dx)
        self.v = np.zeros(self.dx)
        if lr is None: 
            self.alpha = 0.001
        else:
            self.alpha = lr
        self.t = 0
        self.epislon = 1e-8

    def update(self, theta, grad, alpha = None):
        self.t += 1
        self.m = (1.-self.beta1)*grad + self.beta1*self.m
        self.v = (1.-self.beta2)*(grad**2) + self.beta2*self.v
        hat_m = self.m / (1. - self.beta1**self.t)
        hat_v = self.v / (1. - self.beta2**self.t)
        if alpha is None:
            return theta - self.alpha*hat_m/(np.sqrt(hat_v) + self.epislon)
        else:
            return theta - alpha*hat_m/(np.sqrt(hat_v) + self.epislon)

class PlainGD(object):
    def __init__(self, dx, lr = None):
        self.dx = dx
        
    def update(self, theta, grad, alpha):
        return theta - alpha * grad

class LQG_env(object):
    def  __init__(self, method_name):

        self.method_name = method_name

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

        self.dx = 3
        self.du = 3

        self.init_state_mean = np.ones(self.dx)*5
        self.init_state_cov = np.eye(self.dx)*0.1

        self.state = None

        self.noise_cov = np.eye(self.dx)*0.001 

        self.K = 1e-3*np.random.randn(self.du, self.dx) #np.zeros((3,3))  #u = Kx
        self.P_K = None
 
        self.gamma = 0.998

        self.ADAM_optimizer = ADAM(dx = self.dx*self.du, lr = None)
        #self.Plain_GD = PlainGD(dx = self.dx*self.du)


    def reset(self):
        self.state = np.random.multivariate_normal(mean = self.init_state_mean, cov = self.init_state_cov)
        return self.state

    def step(self, x, u):
        cost = x.dot(self.Q).dot(x) + u.dot(self.R).dot(u)
        next_state = self.A.dot(x) + self.B.dot(u)
        next_state += np.random.multivariate_normal(mean = np.zeros(self.dx), cov = self.noise_cov)
        return next_state, cost, False

    def Fixed_point_iteration_Raccati_optimal_equation(self, max_iter):
        max_iter = int(max_iter)
        P_K = np.zeros((self.dx, self.dx))
        current_P_K = np.zeros((self.dx, self.dx))
        #max_iter = 200
        for i in range(2*max_iter):
            new_P_K = (self.Q + self.A.T.dot(current_P_K).dot(self.A) - 
                    self.A.T.dot(current_P_K).dot(self.B).dot(np.linalg.inv(self.B.T.dot(current_P_K).dot(self.B)+self.R)).dot(self.B.T).dot(current_P_K).dot(self.A))
            if np.linalg.norm(new_P_K - current_P_K) < 1e-5:
                break;
            current_P_K = np.copy(new_P_K)
        self.P_K = np.copy(current_P_K)
        
    def optimal_K(self):
        self.Fixed_point_iteration_Raccati_optimal_equation(max_iter = 500)
        self.K = -np.linalg.inv(self.B.T.dot(self.P_K).dot(self.B) + self.R).dot(self.B.T).dot(self.P_K).dot(self.A)

    def Fixed_point_iteration_Raccati_equation(self, max_iter):
        max_iter = int(max_iter)
        P_K = np.zeros((self.dx, self.dx))
        current_P_K = np.zeros((self.dx, self.dx))
        #max_iter = 200
        for i in range(2*max_iter):
            new_P_K = self.Q + self.K.T.dot(self.R).dot(self.K) + self.gamma*(self.A+self.B.dot(self.K)).T.dot(current_P_K).dot(self.A+self.B.dot(self.K))
            if np.linalg.norm(new_P_K - current_P_K) < 1e-7:
                break;
            current_P_K = np.copy(new_P_K)
        self.P_K = np.copy(current_P_K)

    def Jacobian_with_vectorized_K(self, x):
        dim_K = self.du*self.dx
        J = np.zeros((self.du, dim_K))
        for i in range(self.du):
            J[i, i*self.dx:i*self.dx+self.dx] = x
        return J

    def empircal_off_policy_gradient(self, xs):
        batch_size = xs.shape[0]
        us = (self.K.dot(xs.T)).T #u = Kx
        grad_vectorized_K = np.zeros(self.dx*self.du)
        for i in range(xs.shape[0]):
            g_u = 2*self.R.dot(us[i]) + 2*self.gamma*self.B.T.dot(self.P_K).dot(self.A.dot(xs[i])+self.B.dot(us[i]))
            grad_vectorized_K += self.Jacobian_with_vectorized_K(xs[i]).T.dot(g_u)     
        return grad_vectorized_K/(batch_size*1.)
    
    def empircal_off_policy_newton_gradient(self, xs):
        batch_size = xs.shape[0]
        us = (self.K.dot(xs.T)).T #u = Kx
        HinvGrad_vecotirzed_K = np.zeros(self.dx*self.du)
        H_u = 2*self.R + 2*self.gamma*self.B.T.dot(self.P_K).dot(self.B)
        for i in range(xs.shape[0]):
            g_u = 2*self.R.dot(us[i]) + 2*self.gamma*self.B.T.dot(self.P_K).dot(self.A.dot(xs[i])+self.B.dot(us[i]))    
            HinvGrad_vecotirzed_K += self.Jacobian_with_vectorized_K(xs[i]).T.dot(np.linalg.lstsq(a = H_u + 1e-3*np.eye(self.dx), b = g_u, rcond=-1)[0])
        return HinvGrad_vecotirzed_K/(batch_size*1)

    def empircal_off_policy_natural_gradient(self, xs, kl_threshold):
        batch_size = xs.shape[0]
        off_policy_grad = self.empircal_off_policy_gradient(xs)
        JpJ = np.zeros((self.dx*self.du, self.dx*self.du))
        for i in range(xs.shape[0]):
            J = self.Jacobian_with_vectorized_K(xs[i])
            JpJ += J.T.dot(J)
        JpJ /= batch_size
        
        natural_grad = np.linalg.lstsq(a = JpJ + 1e-3*np.eye(JpJ.shape[0]), b = off_policy_grad, rcond = -1)[0]
        #print np.linalg.norm(off_policy_grad), np.linalg.norm(natural_grad)
        #compute learning_rate too:
        natural_grad_lr = np.sqrt(kl_threshold/(off_policy_grad.dot(natural_grad)+1e-7))
        return natural_grad,natural_grad_lr

    def empircal_off_policy_natural_newton(self, xs, kl_threshold):
        batch_size = xs.shape[0]
        off_policy_newton = self.empircal_off_policy_newton_gradient(xs)
        JpJ = np.zeros((self.dx*self.du, self.dx*self.du))
        for i in range(xs.shape[0]):
            J = self.Jacobian_with_vectorized_K(xs[i])
            JpJ += J.T.dot(J)
        JpJ /= batch_size
        
        natural_newton = np.linalg.lstsq(a = JpJ + 1e-3*np.eye(JpJ.shape[0]), b = off_policy_newton, rcond=-1)[0]
        #compute learning_rate too:
        natural_newton_lr = np.sqrt(kl_threshold/(off_policy_newton.dot(natural_newton)+1e-7))
        return natural_newton,natural_newton_lr
    
    def plain_gradient_descent(self, xs, batch_size = 64, lr = None):
         #sample: 
         #xs = np.random.multivariate_normal(self.init_state_mean, self.init_state_cov, size = batch_size)
         #approximately compute P_K based on the current K:
         self.Fixed_point_iteration_Raccati_equation(max_iter = 1./(1-self.gamma))
         #compute plain "DPG":
         vectorized_grad = self.empircal_off_policy_gradient(xs = xs)
         vectorized_grad = vectorized_grad/np.linalg.norm(vectorized_grad)
         #print np.linalg.norm(vectorized_grad)
         #[U,S] = np.linalg.eig(self.P_K)
    
         vectorized_new_K = self.ADAM_optimizer.update(theta = self.K.reshape(self.du*self.dx), grad = vectorized_grad, alpha = lr)
         #vectorized_new_K = self.Plain_GD.update(theta = self.K.reshape(self.du*self.dx), grad = vectorized_grad, alpha = lr)
         self.K = np.reshape(vectorized_new_K, (self.du, self.dx))
        
    def natural_gradient_descent(self, xs, batch_size = 64, kl_threshold = 0.001):
        #xs = np.random.multivariate_normal(self.init_state_mean, self.init_state_cov, size = batch_size)
        #approximately compute P_K based on the current K:
        self.Fixed_point_iteration_Raccati_equation(max_iter = 1./(1-self.gamma))
        vectorized_nat_grad, nat_grad_lr = self.empircal_off_policy_natural_gradient(xs = xs, kl_threshold = kl_threshold)
        #print np.linalg.norm(vectorized_nat_grad), nat_grad_lr
        vectorized_new_K = self.K.reshape(self.du*self.dx) - nat_grad_lr*vectorized_nat_grad
        self.K = np.reshape(vectorized_new_K, (self.du, self.dx))
    
    def natural_newton_descent(self, xs, batch_size = 64, kl_threshold = 0.001):
        #xs = np.random.multivariate_normal(self.init_state_mean, self.init_state_cov, size = batch_size)
        #approximately compute P_K based on the current K:
        self.Fixed_point_iteration_Raccati_equation(max_iter = 1./(1-self.gamma))
        vectorized_nat_newton, nat_newton_lr = self.empircal_off_policy_natural_newton(xs = xs, kl_threshold = kl_threshold)
        vectorized_new_K = self.K.reshape(self.du*self.dx) - nat_newton_lr*vectorized_nat_newton
        self.K = np.reshape(vectorized_new_K, (self.du, self.dx))

    def exact_evaluate(self):
        #evaluate the off-policy objective with the current K.
        #obtain P_K:
        self.Fixed_point_iteration_Raccati_equation(max_iter = 1./(1-self.gamma))
        return self.init_state_mean.dot(self.P_K).dot(self.init_state_mean) + np.trace(self.P_K.dot(self.init_state_cov)) #+ np.trace(self.P_K.dot(self.noise_cov))

    def train(self, epoch = 200, batch_size = 64, lr_or_KL = 0.001):
        epoch_cost = []
        for e in range(epoch):
            #evaluate:
            curr_cost =  self.exact_evaluate()
            print ('at epoch {0}, the current policy has cost {1}'.format(e, curr_cost))
            epoch_cost.append(curr_cost)

            xs = np.random.multivariate_normal(self.init_state_mean, self.init_state_cov, size = batch_size)
            if self.method_name == 'Plain GD':
                self.plain_gradient_descent(xs, batch_size=batch_size, lr = lr_or_KL)
            elif self.method_name == 'Natural GD':
                self.natural_gradient_descent(xs = xs, batch_size= batch_size, kl_threshold=lr_or_KL)
            elif self.method_name == 'Newton Natural GD':
                self.natural_newton_descent(xs = xs, batch_size=batch_size, kl_threshold=lr_or_KL)
        
        return epoch_cost

    def rollout(self, T = 100):
        x = np.random.multivariate_normal(mean = self.init_state_mean, cov = self.init_state_cov, size = 1)[0]
        traj_x = []
        traj_a = []
        traj_c = []
        for t in range(T):
            a = self.K.dot(x)
            traj_x.append(x)
            traj_a.append(a)
            x, c, done = self.step(x = x, u = a)
            traj_c.append(c)
        
        return traj_x, traj_a, traj_c
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1337)
    parser.add_argument('--alg', type=str, default='Natural_GD') 
    args = parser.parse_args()
    np.random.seed(args.seed)
    method_name = args.alg.replace("_", " ") #it also supports Natural GD and Newton Natural GD
    model = LQG_env(method_name = method_name)  
    epoches_costs = model.train(epoch = 50, batch_size = 64, lr_or_KL=1e-2)


    
                

