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


H = 150
W = 200

# cut_ind = 135
cut_ind = 150

# vid_name = 'two_persons_walking'
vid_name = 'one_person_walking'

vid = cv2.VideoCapture('./data/' + vid_name + '.mp4')
small_frames = []
while True:
    ret, im = vid.read() # .convert('RGB')
    if ret == False:
        break
    im = cv2.resize(im.mean(-1), (W,H), None, None)
    small_frames.append(im)
    # data_mat = np.concatenate((data_mat,np.reshape(im,(H*W,1))), axis=1)

small_frames = small_frames[cut_ind:-5]
frames = len(small_frames)
data_mat = np.stack(list(map(lambda x: np.reshape(x,(H*W,1)),small_frames)),axis=1)[:,:,0]



fig, axs = plt.subplots(3, 3, figsize=(13, 10))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(small_frames[i], cmap="gray")
    ax.axis("off")

fig.tight_layout()




from robustpca.pcp import PCP
from robustpca.pcp import StablePCP



print('start PCP')
pcp_alm = PCP()
mu = pcp_alm.default_mu(data_mat)
L_pcp, S_pcp = pcp_alm.decompose(data_mat, mu, tol=1e-5, max_iter=500, verbose=True)
print('finish PCP')




f'intrisic rank: {np.linalg.matrix_rank(L_pcp)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_pcp != 0).mean():.3f}'




from matplotlib import pyplot as plt

ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    background = L_pcp[:, ind].reshape(small_frames[ind].shape)
    foreground = S_pcp[:, ind].reshape(small_frames[ind].shape)
    axs[0, i].imshow(small_frames[ind], cmap='gray')

    axs[1, i].imshow(background, cmap='gray')
    axs[1, i].set_title("PCP, L")

    axs[2, i].imshow(foreground, cmap='gray')
    axs[2, i].set_title("PCP, S")

fig.tight_layout()
plt.savefig("./figs/pcp_" + vid_name + ".pdf")


print('start stable PCP')
st_pcp = StablePCP()
mu = st_pcp.default_mu(data_mat, sigma=10)
L_st_pcp, S_st_pcp = st_pcp.decompose(data_mat, mu, tol=1e-5, max_iter=500, verbose=True)
print('finish stable PCP')



f'intrisic rank: {np.linalg.matrix_rank(L_st_pcp)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_st_pcp != 0).mean():.3f}'



from matplotlib import pyplot as plt

ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[ind], cmap='gray')

    # background = L_pcp[:, i].reshape(small_frames[i].shape)
    # foreground = S_pcp[:, i].reshape(small_frames[i].shape)
    # axs[1, i].imshow(background, cmap='gray')
    # axs[1, i].set_title("PCP, L")
    # axs[2, i].imshow(foreground, cmap='gray')
    # axs[2, i].set_title("PCP, S")

    background = L_st_pcp[:, ind].reshape(small_frames[ind].shape)
    foreground = S_st_pcp[:, ind].reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray')
    axs[1, i].set_title("Stable PCP, L")
    axs[2, i].imshow(foreground, cmap='gray')
    axs[2, i].set_title("Stable PCP, S")

fig.tight_layout()
plt.savefig("./figs/stable_pcp_" + vid_name + ".pdf")


from robustpca.ircur import IRCUR



print('start IRCUR')
pcp_alm = IRCUR()
rank = 2
c = 4
nrows, ncols = int(c * rank * np.log(data_mat.shape[0])), int(c * rank * np.log(data_mat.shape[1]))
L_ircur, S_ircur = pcp_alm.decompose(data_mat, rank, nrows, ncols, thresholding_decay=0.65, initial_threshold=100, verbose=True, max_iter=500, tol=1e-9)
print('finish IRCUR')




f'intrisic rank: {np.linalg.matrix_rank(L_ircur)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_ircur != 0).mean():.3f}'



from matplotlib import pyplot as plt

ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[ind], cmap='gray')

    # background = L_pcp[:, i].reshape(small_frames[i].shape)
    # foreground = S_pcp[:, i].reshape(small_frames[i].shape)
    # axs[1, i].imshow(background, cmap='gray')
    # axs[1, i].set_title("PCP, L")
    # axs[2, i].imshow(foreground, cmap='gray')
    # axs[1, i].set_title("PCP, S")

    # background = L_st_pcp[:, i].reshape(small_frames[i].shape)
    # foreground = S_st_pcp[:, i].reshape(small_frames[i].shape)
    # axs[3, i].imshow(background, cmap='gray')
    # axs[3, i].set_title("Stable PCP, L")
    # axs[4, i].imshow(foreground, cmap='gray')
    # axs[4, i].set_title("Stable PCP, S")

    background = L_ircur[:, ind].reshape(small_frames[ind].shape)
    foreground = S_ircur[:, ind].reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray')
    axs[1, i].set_title("IRCUR, L")
    axs[2, i].imshow(foreground, cmap='gray')
    axs[2, i].set_title("IRCUR, S")

fig.tight_layout()
plt.savefig("./figs/ircur_" + vid_name + ".pdf")



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




f'intrisic rank: {np.linalg.matrix_rank(L_admm)}, original rank: {np.linalg.matrix_rank(data_mat)}, fraction of outliers: {(S_admm != 0).mean():.3f}'



from matplotlib import pyplot as plt

ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[ind], cmap='gray')

    background = L_admm[:, ind].reshape(small_frames[ind].shape)
    foreground = S_admm[:, ind].reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray')
    axs[1, i].set_title("ADMM, L")
    axs[2, i].imshow(foreground, cmap='gray')
    axs[2, i].set_title("ADMM, S")

fig.tight_layout()
plt.savefig("./figs/admm_" + vid_name + ".pdf")
