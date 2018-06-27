import cv2
import numpy as np
import pdb

def MSE(im1, im2):
    return np.mean((im1 - im2)**2)

mu = 1
sigma = 43
im_num = 150
im = cv2.imread('origin_gray.png')
im = im.astype(np.float32)[:, :, 0]
# fp = open('./s_image/real_mean.txt')

im_before_clip = np.zeros((im_num, im.shape[0], im.shape[1]))
im_after_clip = np.zeros((im_num, im.shape[0], im.shape[1]))
for i in range(im_num):
    noise = np.random.normal(mu, sigma, size=(im.shape[0], im.shape[1]))
    im_noise = noise + im
    im_before_clip[i, :, :] = im_noise
    im_noise_clip = im_noise.clip(0, 255).astype(np.uint8)
    im_after_clip[i, :, :] = im_noise_clip
    mse_single = MSE(im, im_noise_clip)
    print(mse_single)
    cv2.imwrite('./s_image/' + str(i) + '.png', im_noise_clip)

real_mean_im = np.mean(im_before_clip, axis=0)
clip_mean_im = np.mean(im_after_clip, axis=0)
np.savetxt('./s_image/real_mean.txt', real_mean_im)
np.savetxt('./s_image/clip_mean.txt', clip_mean_im)

cv2.imwrite('./s_image/clip_mean_im.png', clip_mean_im.astype(np.uint8))

print('ave_mse: ', MSE(im / 255.0, clip_mean_im.astype(np.uint8) / 255.0))






































