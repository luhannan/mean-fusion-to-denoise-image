# Robust mean image estimation
Implementation the mean fusion part of [A High-Quality Denoising Dataset for Smartphone Cameras](https://www.eecs.yorku.ca/~mbrown/pdf/sidd_cvpr2018.pdf). The result is quite different from what the author presented in the paper.
- sys_image.py synsthesize n images with gaussian noise from a noise free image.
- nelder_mead_mProcess.py use nelder_mead minimize the objective function and get the mean image.
- nelder_mead_simu.py uses 1 pixel to simulate. 