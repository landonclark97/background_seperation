#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../../admm')

from admm import admm
import laplace

import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

from robustpca.general import DATADIR

from PIL import Image

import glob


H = 150
W = 200

SIGMA = 5e-3



vid_name = 'two_persons_walking'
# vid_name = 'robot_reach'
# vid_name = 'human_rob_int'

# # two person walking crop
# if vid_name == 'two_persons_walking':
#     crop_left = 0
#     crop_top = 0
#     crop_right = 0
#     crop_bottom = 0

# # robot reach crop
# elif vid_name == 'robot_reach':
#     crop_left = 15
#     crop_top = 500
#     crop_right = 0
#     crop_bottom = 300

# # human robot interaction
# elif vid_name == 'human_rob_int':
#     crop_left = 20
#     crop_top = 507
#     crop_right = 0
#     crop_bottom = 320


b_name = '../../data/'+vid_name+'_motionless/frame'

b_img = cv2.imread('../../data/'+vid_name+'_motionless/b_img.png')
b_img = b_img.mean(-1)
b_img /= 255.0
# b_img /= np.amax(b_img)


f_name = '../../data/'+vid_name+'_motionless/'
f_im_list = glob.glob(f_name+'frame*.png')

frames = len(f_im_list)
data_mat = np.empty((H,W,frames))
for f in range(1,frames+1):
    f_im = cv2.imread(f_name+'frame'+str(f)+'.png')
    im = cv2.resize(f_im.mean(-1), (W,H), None, None)
    data_mat[:,:,f-1] = im

data_mat /= np.amax(data_mat)


small_frames = np.copy(data_mat)

data_mat = np.reshape(data_mat,(H*W,-1))
data_mat += np.random.normal(0.0,SIGMA,data_mat.shape)

fig, axs = plt.subplots(3, 3, figsize=(13, 10))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(small_frames[:,:,i], cmap="gray")
    ax.axis("off")

fig.tight_layout()




# from robustpca.pcp import PCP
# from robustpca.pcp import StablePCP



# print('start PCP')
# pcp_alm = PCP()
# mu = pcp_alm.default_mu(data_mat)
# L_pcp, S_pcp = pcp_alm.decompose(data_mat, mu, tol=1e-5, max_iter=500, verbose=True)
# print('finish PCP')


# L = np.reshape(L_pcp, (H,W,-1))
# S = np.reshape(S_pcp, (H,W,-1))

# with open('../../examples/python/L/pcp_' + vid_name + '_L.npy', 'wb') as f:
#     np.save(f, L)
# with open('../../examples/python/S/pcp_' + vid_name + '_S.npy', 'wb') as f:
#     np.save(f, S)



# f'intrisic rank: {np.linalg.matrix_rank(L_pcp)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_pcp != 0).mean():.3f}'




# from matplotlib import pyplot as plt

# ncols = 6
# fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

# for ax in axs.flatten():
#     ax.axis('off')

# for i in range(ncols):
#     ind = i*(frames//ncols)
#     background = L_pcp[:, ind].reshape(small_frames[:,:,ind].shape)
#     foreground = S_pcp[:, ind].reshape(small_frames[:,:,ind].shape)
#     axs[0, i].imshow(small_frames[:,:,ind], cmap='gray')

#     axs[1, i].imshow(background, cmap='gray')
#     axs[1, i].set_title("PCP, L")

#     axs[2, i].imshow(np.abs(foreground), cmap='gray')
#     axs[2, i].set_title("PCP, S")

# fig.tight_layout()
# plt.savefig("./figs/pcp_" + vid_name + ".pdf")


# print('start stable PCP')
# st_pcp = StablePCP()
# mu = st_pcp.default_mu(data_mat, sigma=0.01)
# L_st_pcp, S_st_pcp = st_pcp.decompose(data_mat, mu, tol=1e-6, max_iter=500, verbose=True)
# print('finish stable PCP')


# L = np.reshape(L_st_pcp, (H,W,-1))
# S = np.reshape(S_st_pcp, (H,W,-1))

# with open('../../examples/python/L/st_pcp_' + vid_name + '_L.npy', 'wb') as f:
#     np.save(f, L)
# with open('../../examples/python/S/st_pcp_' + vid_name + '_S.npy', 'wb') as f:
#     np.save(f, S)

# f'intrisic rank: {np.linalg.matrix_rank(L_st_pcp)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_st_pcp != 0).mean():.3f}'


# from matplotlib import pyplot as plt

# ncols = 6
# fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

# for ax in axs.flatten():
#     ax.axis('off')

# for i in range(ncols):
#     ind = i*(frames//ncols)
#     axs[0, i].imshow(small_frames[:,:,ind], cmap='gray')

#     background = L_st_pcp[:, ind].reshape(small_frames[:,:,ind].shape)
#     foreground = S_st_pcp[:, ind].reshape(small_frames[:,:,ind].shape)
#     axs[1, i].imshow(background, cmap='gray')
#     axs[1, i].set_title("Stable PCP, L")
#     axs[2, i].imshow(np.abs(foreground), cmap='gray')
#     axs[2, i].set_title("Stable PCP, S")

# fig.tight_layout()
# plt.savefig("./figs/stable_pcp_" + vid_name + ".pdf")


from robustpca.ircur import IRCUR


ITERS = 5

L_ircur = np.zeros(data_mat.shape)
S_ircur = np.zeros(data_mat.shape)


print('start IRCUR')
pcp_alm = IRCUR()
rank = 2
c = 4
nrows, ncols = int(c * rank * np.log(data_mat.shape[0])), int(c * rank * np.log(data_mat.shape[1]))

for i in range(ITERS):
    L_curr, S_curr = pcp_alm.decompose(data_mat, rank, nrows, ncols, thresholding_decay=0.65, initial_threshold=100, verbose=True, max_iter=500, tol=1e-9)
    L_ircur += L_curr
    S_ircur += S_curr

L_ircur /= float(ITERS)
S_ircur /= float(ITERS)

print('finish IRCUR')


L = np.reshape(L_ircur, (H,W,-1))
S = np.reshape(S_ircur, (H,W,-1))

with open('../../examples/python/L/ircur_' + vid_name + '_L.npy', 'wb') as f:
    np.save(f, L)
with open('../../examples/python/S/ircur_' + vid_name + '_S.npy', 'wb') as f:
    np.save(f, S)


f'intrisic rank: {np.linalg.matrix_rank(L_ircur)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_ircur != 0).mean():.3f}'



from matplotlib import pyplot as plt

ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[:,:,ind], cmap='gray')

    background = L_ircur[:, ind].reshape(small_frames[:,:,ind].shape)
    foreground = S_ircur[:, ind].reshape(small_frames[:,:,ind].shape)
    axs[1, i].imshow(background, cmap='gray')
    axs[1, i].set_title("IRCUR, L")
    axs[2, i].imshow(np.abs(foreground), cmap='gray')
    axs[2, i].set_title("IRCUR, S")

fig.tight_layout()
plt.savefig("./figs/ircur_" + vid_name + ".pdf")

quit()

# rho = 5e-3
# gam = 5e-6
# lam1 = 2.0
# lam2 = 1e-3


rho = 5e-3
gam = 5e-6
lam1 = 2.0
lam2 = 3e-3


print('start ADMM')
L_admm, S_admm = admm(np.reshape(data_mat,(H,W,-1)),
                      laplace.s_laplace(np.reshape(data_mat,(H,W,-1))),
                      rho,gam,lam1,lam2,thresh=0.0001,iters=24)
L_admm = np.where(np.abs(L_admm) > 20.0, L_admm, 0.0)
S_admm = np.where(np.abs(S_admm) > 20.0, S_admm, 0.0)
print('finish ADMM')




# f'intrisic rank: {np.linalg.matrix_rank(L_admm)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_admm != 0).mean():.3f}'



from matplotlib import pyplot as plt

ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[:,:,ind], cmap='gray')

    background = L_admm[:, ind].reshape(small_frames[ind].shape)
    foreground = S_admm[:, ind].reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray')
    axs[1, i].set_title("ADMM, L")
    axs[2, i].imshow(np.abs(foreground), cmap='gray')
    axs[2, i].set_title("ADMM, S")

fig.tight_layout()
plt.savefig("./figs/admm_" + vid_name + ".pdf")
