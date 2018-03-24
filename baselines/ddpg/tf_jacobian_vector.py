from mpi4py import MPI
import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U

'''
implementation of jacobian vector (left and right) multiplication, and J'Jp in the order of Jp first and then J'(Jp)
'''
def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def intprod(x):
    return int(np.prod(x))

def numel(x):
    return intprod(var_shape(x))

def flatgrad(loss, var_list, grad_ys, clip_norm=None):
    grads = tf.gradients(loss, var_list, grad_ys)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    flatten_grad = tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])
    return tf.reshape(flatten_grad, [numel(flatten_grad)])

#implementation of u J
def v_jacobian_product(ys, xs, u):
    g = flatgrad(ys, xs, grad_ys = u) # sum_i v[i] * dys[i]/dxs
    return g

#implementation of Ju' (uJ')
def jacobian_v_product(ys, xs, u):
    v = tf.placeholder(ys.dtype, shape = ys.get_shape()) #a dummy variable. 
    #g = tf.gradients(ys, xs, grad_ys = v)  #  v' nabla_{xs}(ys)
    g = flatgrad(ys, xs, grad_ys = v)  #  v' nabla_{xs}(ys)
    return tf.gradients(g, v, grad_ys = u)  # u' nabla_{v} [v'nabla_{xs}ys] => u' nabla_{xs}[ys]' = (nabla_{xs}[ys] u)'

#implementation of J'Ju 
def Jacobian_p_Jacobian_u_product(ys, xs, u):
    #compute Ju:
    Ju = jacobian_v_product(ys, xs, u) #Ju
    JpJu = v_jacobian_product(ys, xs, u = Ju)
    return JpJu

#create a function (not in symbolic fashion) that returns the real values of J'Jv,
#so that we can keep call it in Conjugate Gradient (CG)
def JpJu_product(symbolic_jpjv, symbolic_x, symbolic_u,
                x_val, u_val, session):
    jpjv_result = session.run(symbolic_jpjv, feed_dict={symbolic_x:x_val, symbolic_u:u_val})
    return jpjv_result

#use cg to solve (A+lambdaI)x = b (in defualt, )
def cg(f_Ax, b, damping = 1e-3, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    f_Ax is a function that computes the matrix-vector product: Ax, for any vector x as input. 
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: 
        print (titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: 
            print (fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p) + damping*p  #(A+lambda I)p
        pdotz = p.dot(z)
        if pdotz < np.finfo(np.float32).eps: 
            v = (rdotr + np.finfo(np.float32).eps) / (pdotz + np.finfo(np.float32).eps) # add epsilon to avoid zero division 
        else:
            v = rdotr / pdotz
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        if rdotr < np.finfo(np.float32).eps: 
            mu = (newrdotr + np.finfo(np.float32).eps) / (rdotr + np.finfo(np.float32).eps) # add epsilon to avoid zero division 
        else:
            mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: 
        print (fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x

class PlainDescent(object):
    def __init__(self, var_list, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.scale_grad_by_procs = scale_grad_by_procs
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, gradients, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = gradients.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        self.setfromflat(self.getflat() - stepsize*gradients)

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

if __name__ == '__main__':
    np.random.seed(1337)
    Theta = tf.Variable(np.random.randn(2,3), dtype = tf.float32)
    reshaped_theta = tf.reshape(Theta, [numel(Theta)])
    #A = tf.constant(np.random.randn(5, 3), dtype=tf.float32)
    x = tf.placeholder(tf.float32, [3]) #input 5 dim
    y =  tf.matmul(Theta, tf.reshape(x, shape=[3, 1])) #tf.tanh(tf.matmul(x, Theta)) #output 3 dim:  y = f(x;Theta) = tanh(x.dot(Theta))
    u_left = tf.placeholder(tf.float32, [2, 1])
    u_right = tf.placeholder(tf.float32, [6])

    vjp = v_jacobian_product(y, [Theta], u_left)  #u.dot(\nabla_{theta}f(x;Theta))
    jvp = jacobian_v_product(y, [Theta], u_right) # \nabla_{theta}f(x;Theta).dot(u)
    jjv = Jacobian_p_Jacobian_u_product(y, [Theta], u_right)

    x_val = np.random.randn(3)
    u_val_right = np.random.randn(6)
    u_val_left = np.random.randn(2, 1)

    Jtheta = [tf.reshape(tf.gradients(y[0], Theta), shape=[6]), tf.reshape(tf.gradients(y[1], Theta), shape=[6])]

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        vjp_result = sess.run(vjp, feed_dict={x:x_val, u_left:u_val_left})
        jvp_result = sess.run(jvp,  feed_dict={x: x_val, u_right: u_val_right})
        jjv_results = sess.run(jjv, feed_dict={x:x_val, u_right:u_val_right})
        jpjv = JpJu_product(symbolic_jpjv = jjv, symbolic_x = x, symbolic_u = u_right,
                            x_val = x_val, u_val = u_val_right, session = sess)
        Jtheta_results = np.array(sess.run(Jtheta, feed_dict={x: x_val}))

        # f(x) = J'Jx, b = J'Ju 
        # golden solution is x = u 
        f_Ax = lambda u: JpJu_product(symbolic_jpjv = jjv, symbolic_x = x, symbolic_u = u_right,
                            x_val = x_val, u_val = u, session = sess)
        cgresult = cg(f_Ax, jpjv, cg_iters=10, verbose=True, damping=1e-3)

        print ("Theta = ", Theta.eval())
        print ("reshaped_theta = ", reshaped_theta.eval())
        print ("Jtheta = ", Jtheta_results)
        print ("==============================")
        print ("vjp         = ", vjp_result) # should match 
        print ("v' * Jtheta = ", np.transpose(u_val_left).dot(Jtheta_results))
        print ("==============================")
        print ("jvp        = ", jvp_result) # should match 
        print ("Jtheta * u = ", Jtheta_results.dot(u_val_right))
        print ("==============================")
        print ("jjv                  = ", jjv_results) # should match 
        print ("Jtheta' * Jtheta * u = ", np.transpose(Jtheta_results).dot(Jtheta_results).dot(u_val_right))
        print ("==============================")
        print ("cg x        = ", cgresult) # may not match u_right
        print ("u_right     = ", u_val_right)
        print ("sol f(x)    = ", f_Ax(cgresult)) # must match golden 
        print ("golden f(x) = ", f_Ax(u_val_right))

