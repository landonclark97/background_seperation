#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '../admm')
sys.path.insert(0, '../tests/robust-pca')

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

# SIGMA = 5e-3



vid_name = 'two_persons_walking'
# vid_name = 'robot_reach'
# vid_name = 'human_rob_int'


b_name = '../data/'+vid_name+'_motionless/b_img.png'

b_img = cv2.imread(b_name)
b_img = b_img.mean(-1)
b_img /= 255.0
# b_img /= np.amax(b_img)


f_name = '../data/'+vid_name+'_motionless/'
f_im_list = glob.glob(f_name+'frame*.png')

frames = len(f_im_list)
data_mat = np.empty((H,W,frames))
for f in range(1,frames+1):
    f_im = cv2.imread(f_name+'frame'+str(f)+'.png')
    im = cv2.resize(f_im.mean(-1), (W,H), None, None)
    data_mat[:,:,f-1] = im

data_mat /= 255.0
# data_mat /= np.amax(data_mat)


small_frames = np.copy(data_mat)

data_mat = np.reshape(data_mat,(H*W,-1))
# data_mat += np.random.normal(0.0,SIGMA,data_mat.shape)

# fig, axs = plt.subplots(3, 3, figsize=(13, 10))

# for i, ax in enumerate(axs.flatten()):
#     ax.imshow(small_frames[:,:,i], cmap="gray")
#     ax.axis("off")

# fig.tight_layout()




from robustpca.pcp import PCP
from robustpca.pcp import StablePCP



# print('start PCP')
pcp_alm = PCP()
mu = pcp_alm.default_mu(data_mat)
L_pcp, S_pcp = pcp_alm.decompose(data_mat, mu, tol=1e-5, max_iter=500, verbose=True)
print('finish PCP')


L = np.reshape(L_pcp, (H,W,-1))
S = np.reshape(S_pcp, (H,W,-1))

with open('./L/pcp_' + vid_name + '_L.npy', 'wb') as f:
    np.save(f, L)
with open('./S/pcp_' + vid_name + '_S.npy', 'wb') as f:
    np.save(f, S)



from matplotlib import pyplot as plt

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


print('start stable PCP')
st_pcp = StablePCP()
mu = st_pcp.default_mu(data_mat, sigma=0.01)
L_st_pcp, S_st_pcp = st_pcp.decompose(data_mat, mu, tol=1e-6, max_iter=500, verbose=True)
print('finish stable PCP')


L = np.reshape(L_st_pcp, (H,W,-1))
S = np.reshape(S_st_pcp, (H,W,-1))

with open('./L/st_pcp_' + vid_name + '_L.npy', 'wb') as f:
    np.save(f, L)
with open('./S/st_pcp_' + vid_name + '_S.npy', 'wb') as f:
    np.save(f, S)



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

with open('./L/ircur_' + vid_name + '_L.npy', 'wb') as f:
    np.save(f, L)
with open('./S/ircur_' + vid_name + '_S.npy', 'wb') as f:
    np.save(f, S)




# from matplotlib import pyplot as plt

# ncols = 6
# fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

# for ax in axs.flatten():
#     ax.axis('off')

# for i in range(ncols):
#     ind = i*(frames//ncols)
#     axs[0, i].imshow(small_frames[:,:,ind], cmap='gray')

#     background = L_ircur[:, ind].reshape(small_frames[:,:,ind].shape)
#     foreground = S_ircur[:, ind].reshape(small_frames[:,:,ind].shape)
#     axs[1, i].imshow(background, cmap='gray')
#     axs[1, i].set_title("IRCUR, L")
#     axs[2, i].imshow(np.abs(foreground), cmap='gray')
#     axs[2, i].set_title("IRCUR, S")

# fig.tight_layout()
# plt.savefig("./figs/ircur_" + vid_name + ".pdf")


beta = 1.8
K = 15

# two person walking - these values are not fine-tuned
if vid_name == 'two_persons_walking':
    rho = 1e-1 #0.001 # 0.01e-1 # 0.005, 0.001
    gam = 1e-1 #0.005 # 0.05e-1 # 0.005, 0.005
    lam1 = 3.5 #1.0 # 1.0  # 3.0, 1.0
    lam2 = 0.005 #0.003 # 0.0029 # 0.007, 0.003

# robot reach
elif vid_name == 'robot_reach':
    rho = 1.2e-1
    gam = 1.2e-1
    lam1 = 2.0
    lam2 = 0.05

# human robot interaction
elif vid_name == 'human_rob_int':
    rho = 1.2e-1
    gam = 1.2e-1
    lam1 = 5.5
    lam2 = 0.0188


mu = 0.0001
alp = 1.0

b_img_c = np.copy(b_img)
L, S, scores = admm(np.reshape(data_mat,(H,W,-1)),
                    laplace.s_laplace(np.reshape(data_mat,(H,W,-1))),
                    rho,gam,lam1,lam2,beta,K,mu,alp,step=0.1,wrap_t=True,
                    normalized=False,b_img=b_img,thresh=1e-5,iters=250)

print(min(scores))

L = np.array(L)
L = L.reshape((H,W,-1))

S = np.array(S)
S = S.reshape((H,W,-1))


with open('./L/admm_'+vid_name+'_L.npy', 'wb') as f:
    np.save(f, L)

with open('./S/admm_'+vid_name+'_S.npy', 'wb') as f:
    np.save(f, S)

# from matplotlib import pyplot as plt

# ncols = 6
# fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

# for ax in axs.flatten():
#     ax.axis('off')

# for i in range(ncols):
#     ind = i*(frames//ncols)
#     axs[0, i].imshow(small_frames[:,:,ind], cmap='gray')

#     background = L_admm[:, ind].reshape(small_frames[ind].shape)
#     foreground = S_admm[:, ind].reshape(small_frames[ind].shape)
#     axs[1, i].imshow(background, cmap='gray')
#     axs[1, i].set_title("ADMM, L")
#     axs[2, i].imshow(np.abs(foreground), cmap='gray')
#     axs[2, i].set_title("ADMM, S")

# fig.tight_layout()
# plt.savefig("./figs/admm_" + vid_name + ".pdf")
