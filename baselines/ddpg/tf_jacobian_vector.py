import tensorflow as tf
import numpy as np

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
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
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


np.random.seed(1337)
Theta = tf.Variable(np.random.randn(3,2), dtype = tf.float32)
reshaped_theta = tf.reshape(Theta, [numel(Theta)])
#A = tf.constant(np.random.randn(5, 3), dtype=tf.float32)
x = tf.placeholder(tf.float32, [1, 3]) #input 5 dim
y =  tf.matmul(x,Theta) #tf.tanh(tf.matmul(x, Theta)) #output 3 dim:  y = f(x;Theta) = tanh(x.dot(Theta))
u_left = tf.placeholder(tf.float32, [1, 2])
u_right = tf.placeholder(tf.float32, [1,6])

vjp = v_jacobian_product(y, [Theta], u_left)  #u.dot(\nabla_{theta}f(x;Theta))
jpv = jacobian_v_product(y, [Theta], u_right) # \nabla_{theta}f(x;Theta).dot(u)
jjv = Jacobian_p_Jacobian_u_product(y, [Theta], u_right)

x_val = np.random.randn(1, 3)
u_val_right = np.random.randn(1, 6)
u_val_left = np.random.randn(1,2)


init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    vjp_result = sess.run(vjp, feed_dict={x:x_val, u_left:u_val_left})
    jpv_result = sess.run(jpv,  feed_dict={x: x_val, u_right: u_val_right})
    jjv_results = sess.run(jjv, feed_dict={x:x_val, u_right:u_val_right})
    jpjv = JpJu_product(symbolic_jpjv = jjv, symbolic_x = x, symbolic_u = u_right,
                        x_val = x_val, u_val = u_val_right, session = sess)

    print (Theta.eval())
    print (reshaped_theta.eval())
    print (vjp_result)
    print (jpv_result)
    print (jjv_results)

