from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy
from scipy.optimize import minimize, rosen, rosen_der
from multiprocessing import Pool

def MSE(x, y):
    return ((x - y)) ** 2
def wrap_minimize(args):
    x0 = args[0]
    x_input = args[1]
    real = args[2]
    def loss_func(x):
        loss = 0
        for k in range(x_input.shape[0]):
            t = x_input[k]
            phi_e = sum(x_input[x_input <= t]) / sum(x_input)
            if phi_e == 1:
                continue
            # pdb.set_trace()
            phi_p = norm.cdf(t, x[0], x[1])
            # wt = (phi_e * (1 - phi_e)) ** (-0.5)
            wt = 1
            loss_single = wt * ((phi_p - phi_e) ** 2)
            loss += loss_single
        return loss
    res = minimize(loss_func, x0, method='Nelder-Mead', tol=1e-10)
    mse_c = MSE(real, x0[0])
    mse_n = MSE(real, res.x[0])
    # print(mse_c, mse_n)
    result = np.zeros(3)
    result[0] = mse_c
    result[1] = mse_n
    result[2] = args[3]
    return result


# if __name__ == '__main__':
# for test_iter in range(150)
test_iter = 150
mse_sum_n = 0
mse_sum_c = 0
real = np.random.randint(0, 255)
inputs = []
before_clip = []
for i in range(test_iter):
    mu = 0
    sigma = 43
    noise = norm.rvs(mu, sigma, size=i + 1)
    xj = noise.__add__(real)
    mean_before_clip = np.mean(xj)
    before_clip.append(mean_before_clip)
    x_clip = xj.clip(0, 255) / 255.0
    mean_after_clip = np.mean(x_clip)
    var_after_clip = np.var(x_clip)
    # x_input = x_clip[x_clip > 0]
    # x_input = x_input[x_input < 1]
    x0 = np.zeros(2)
    x0[0] = mean_after_clip
    x0[1] = var_after_clip
    inputs.append((x0, xj, real / 255.0, i))

pool = Pool()
results = pool.map(wrap_minimize, inputs)
pool.close()
pool.join()
mse_c = 0
mse_n = 0
# for i in range(len(results)):
for i in range(len(results)):
    result = results[i]
    mse_b = MSE(real / 255.0, before_clip[i] / 255.0)
    mse_c = result[0]
    mse_n = result[1]
    idx = result[2]
    print('mse before WLS after: %d %.7f  %.7f  %.7f' % (idx, mse_b, mse_n, mse_c))
# print('final %.5f  %.5f' %(mse_c / test_iter, mse_n / test_iter))



















