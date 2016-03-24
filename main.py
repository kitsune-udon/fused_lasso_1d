import numpy as np
import matplotlib.pyplot as plt
import math

# min (0.5 * || Y - T ||_2^2 + lambda1 * ||Z||_2) s.t. D * T = Z
# Y : (p, 1) matrix
# D : (M, p) matrix
# T0 : (p, 1) matrix
# lambda1 : real number
def fused_lasso_cd(Y, D, T0, lambda1, max_iter=100, eps=1e-5):
    def assert_args_type(Y, D, T0, lambda1):
        assert type(Y) == np.ndarray and \
                type(D) == np.ndarray and \
                type(T0) == np.ndarray and \
                type(lambda1) == float and \
                D.shape[1] == T0.shape[0]

    def assert_ret_type(R, T0):
        assert type(R) == type(T0) and \
                R.shape == T0.shape

    assert_args_type(Y, D, T0, lambda1)
    i = 0
    D_T = D.transpose()
    D_T_inv = np.linalg.pinv(D_T)
    U_cur = np.dot(D_T_inv, Y-T0)
    ERROR = U_cur.copy()
    ERROR.fill(eps)

    while max_iter and i < max_iter:
        U_succ = np.array(U_cur)
        for j in xrange(D.shape[0]):
            adjust = (D_T[:,j] * U_cur[j,0])
            adjust = adjust.reshape((len(adjust),1))
            R = Y - np.dot(D_T, U_succ) + adjust
            assert R.shape == Y.shape

            axis = np.dot(D_T_inv, R)[j,0]
            if axis < -lambda1:
                U_succ[j,0] = -lambda1
            elif axis > lambda1:
                U_succ[j,0] = lambda1
            else:
                U_succ[j,0] = axis


        if np.all(np.absolute(U_succ - U_cur) < ERROR):
            U_cur = U_succ
            break

        U_cur = U_succ
        i += 1

    print "end at {} th iter".format(i)
    T = Y - np.dot(D_T, U_cur)

    assert_ret_type(T, T0)
    return T

def fused_lasso_dp(Y, lambda1):
    assert Y.shape[1] == 1

    T = np.zeros(Y.shape)
    n = Y.shape[0]
    T[n-1,0] = Y[n-1,0]

    for i in xrange(n-2, -1, -1):
        k = Y[i] - T[i+1]

        if k > lambda1:
            z = k - lambda1
        elif k < -lambda1:
            z = k + lambda1
        else:
            z = 0.

        T[i] = z + T[i+1]

    return T

# on/off curve
def make_input(size, sigma=None, loss_rate=0.):
    Y = np.zeros((size,1))
    for i in xrange(size):
        if (i / (size/4)) % 2 == 0:
            Y[i,0] = 1.0
        else:
            Y[i,0] = -1.0
    if sigma:
        for i in xrange(size):
            Y[i,0] += sigma * np.random.randn()

    for i in xrange(size):
        if np.random.rand() < loss_rate:
            Y[i,0] = 0.

    return Y

# sine curve
def make_input2(size, sigma=None, loss_rate=0.):
    Y = np.zeros((size,1))
    X = np.linspace(0., 2. * math.pi, size)
    for i in xrange(size):
        Y[i,0] = math.sin(X[i])

    if sigma:
        for i in xrange(size):
            Y[i,0] += sigma * np.random.randn()

    for i in xrange(size):
        if np.random.rand() < loss_rate:
            Y[i,0] = 0.

    return Y

def make_diff_matrix(size):
    D = np.zeros((size-1, size), dtype=myfloat)
    for i in xrange(size-1):
        D[i,i] = 1.
        D[i,i+1] = -1.
    return D

def make_init_point(Y):
    return np.zeros(Y.shape, dtype=myfloat)

def calc_and_plot_cd(gen_func, size, lambda1, sigma=0., loss_rate=0.):
    Y = gen_func(size, sigma, loss_rate=loss_rate)
    D = make_diff_matrix(size)
    T0 = make_init_point(Y)
    T = fused_lasso_cd(Y, D, T0, lambda1)
    xs = np.linspace(0.0, 1.0, size)
    plt.plot(xs, Y, "bo")
    plt.plot(xs, T, "ro")

def calc_and_plot_dp(gen_func, size, lambda1, sigma=0., loss_rate=0.):
    Y = gen_func(size, sigma, loss_rate=loss_rate)
    T = fused_lasso_dp(Y, lambda1)
    xs = np.linspace(0.0, 1.0, size)
    plt.plot(xs, Y, "bo")
    plt.plot(xs, T, "ro")

myfloat = np.float64
np.random.seed(0)
size = 100
lambda1 = 0.1
loss_rate = 0.15
sigma = 0.05
gen_func = make_input
#calc_and_plot_cd(gen_func, size, lambda1, sigma, loss_rate)
calc_and_plot_dp(gen_func, size, lambda1, sigma, loss_rate)
plt.savefig("image.png")

