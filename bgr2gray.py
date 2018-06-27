import cv2
import numpy as np

im_name = 'butterfly_GT.bmp'
im = cv2.imread(im_name)
im_ycbcr = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
im_ycbcr = im_ycbcr[:, :, 0]
cv2.imwrite('origin_gray.png', im_ycbcr)
cv2.imshow('test', im_ycbcr)
cv2.waitKey(0)