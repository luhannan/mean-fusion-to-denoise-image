from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy
from scipy.optimize import minimize, rosen, rosen_der
import os
import cv2
import time
from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool

def wrap_minimize(args):
    i = args[0]
    j = args[1]
    x0 = args[2]
    x_input = args[3]
    def loss_func(x):
        loss = 0
        for k in range(x_input.shape[0]):
            t = x_input[k]
            phi_e = sum(x_input[x_input <= t]) / sum(x_input)
            if phi_e == 1:
                continue
            # pdb.set_trace()
            phi_p = norm.cdf(t, x[0], x[1])
            wt = (phi_e * (1 - phi_e)) ** (-0.5)
            loss_single = wt * ((phi_p - phi_e) ** 2)
            loss += loss_single
        return loss

    res = minimize(loss_func, x0, method='Nelder-Mead', tol=1e-6)
    print(i, j, res.x[0])
    return (i, j, res.x[0])

im_path = './s_image/'
im_num = 150
im = cv2.imread(im_path + '1.png')
im_noise = np.zeros((im_num, im.shape[0], im.shape[1]))
result_nelder = np.zeros((im.shape[0], im.shape[1]))
for i in range(im_num):
    im_name = os.path.join(im_path, str(i) + '.png')
    im = cv2.imread(im_name).astype(np.float64)
    im = im[:, :, 0]
    im_noise[i, :, :] = im
clip_mean = cv2.imread(im_path + 'clip_mean_im.png').astype(np.float64)
# clip_mean = clip_mean[250:300, 250:300, 0] / 255.0
# im_noise = im_noise[:, 250:300, 250:300] / 255.0
clip_mean = clip_mean[:, :, 0] / 255.0
im_noise = im_noise / 255.0




# def wrap_loss_func(args):
#     return loss_func(args[0], args[1])
pool = Pool()
h = im_noise.shape[1]
w = im_noise.shape[2]
inputs = []
for i in range(h):
    for j in range(w):
        xj = im_noise[:, i, j]
        x_input = xj[xj < 1]
        x_input = x_input[x_input > 0]
        mean = clip_mean[i, j]
        var = 1
        x0 = np.zeros(2)
        x0[0] = mean
        x0[1] = var
        inputs.append((i, j, x0, x_input))


start = time.time()
results = pool.map(wrap_minimize, inputs)
pool.close()
pool.join()
end = time.time() - start
print('elaspe: ', end)
for res in results:
    i = res[0]
    j = res[1]
    mu = res[2]
    result_nelder[i, j] = mu

np.savetxt('result.txt', result_nelder * 255)
result_im = (result_nelder * 255).astype(np.uint8)
cv2.imwrite('result.png', result_im)

























