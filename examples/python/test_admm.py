
import sys
sys.path.insert(0, '../../admm')

from admm import admm
import laplace


import torch
import torch.sparse
from torchvision import transforms
import numpy as np
import scipy
import scipy.sparse
import scipy.linalg as lin

import cv2

import matplotlib.pyplot as plt

import glob
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# convert pytorch sparse matrix to scipy sparse matrix
def tsp_to_nsp(A):
    return scipy.sparse.csr_matrix((A.coalesce().values().detach().numpy(),A.coalesce().indices().detach().numpy()),shape=(A.shape[0],A.shape[1]))


SIGMA = 0e-3


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100.0
    max_pixel = 1.0
    psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



# Other Methods for RPCA
# https://github.com/sverdoot/robust-pca


vid_name = 'two_persons_walking'
# vid_name = 'robot_reach'
# vid_name = 'human_rob_int'

H = 150
W = 200


print('loading images')


im_list = glob.glob('../../data/'+vid_name+'_motionless/frame*.png')

frames = len(im_list)
small_frames = np.empty((H,W,frames))
# for f in range(frames):
for f in range(1,frames+1):
    img = cv2.imread('../../data/'+vid_name+'_motionless/frame'+str(f)+'.png')
    small_frames[:,:,f-1] = cv2.resize(img.mean(-1), (W,H), None, None)
data_mat = np.reshape(small_frames,(H*W,-1))
b_img = cv2.imread('../../data/'+vid_name+'_motionless/b_img.png')
b_img = cv2.resize(b_img.mean(-1), (W,H), None, None)

data_mat += (np.random.normal(0.0,SIGMA,data_mat.shape)*255.0)

beta = 1.8
K = 15

# two person walking
rho = 0.001 # 0.01e-1 # 0.005, 0.001
gam = 0.005 # 0.05e-1 # 0.005, 0.005
lam1 = 1.0 # 1.0  # 3.0, 1.0
lam2 = 0.003 # 0.0029 # 0.007, 0.003

# robot reach
# rho = 1.2e-1
# gam = 1.2e-1
# lam1 = 2.0
# lam2 = 0.05

# human robot interaction
# rho = 1.2e-1
# gam = 1.2e-1
# lam1 = 5.5
# lam2 = 0.0188


mu = 0.00001
alp = 1.0

b_img_c = np.copy(b_img)
L, S, scores = admm(np.reshape(data_mat,(H,W,-1)),
                    laplace.s_laplace(np.reshape(data_mat,(H,W,-1))),
                    rho,gam,lam1,lam2,beta,K,mu,alp,step=0.5,wrap_t=True,
                    normalized=False,b_img=b_img,thresh=1e-5,iters=172)


print(min(scores))

# format background images
L = np.array(L)
L = L.reshape((H,W,-1))

# format foreground images
S = np.array(S)
S = S.reshape((H,W,-1))


Dmin = np.amin(data_mat)
Dmax = np.amax(data_mat)

Smin = np.amin(S)
Smax = np.amax(S)

Lmin = np.amin(L)
Lmax = np.amax(L)

ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

fig.suptitle(f'Results with: rho: {rho}, gam: {gam}, lam1: {lam1}, lam2: {lam2}')

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[:,:,ind], cmap='gray', vmin=0.0, vmax=255.0)

    background = L[:,:, ind]#.reshape(small_frames[ind].shape)
    foreground = S[:,:, ind]#.reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray', vmin=Lmin, vmax=Lmax)
    axs[1, i].set_title("ADMM, L")
    axs[2, i].imshow(foreground, cmap='gray', vmin=Smin, vmax=Smax)
    axs[2, i].set_title("ADMM, S")

fig.tight_layout()
plt.savefig("./admm_" + vid_name + ".pdf")


with open('./L/admm_'+vid_name+'_L.npy', 'wb') as f:
    np.save(f, L)

with open('./S/admm_'+vid_name+'_S.npy', 'wb') as f:
    np.save(f, S)
