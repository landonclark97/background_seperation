
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





def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



# Other Methods for RPCA
# https://github.com/sverdoot/robust-pca


vid_name = 'two_persons_walking'

H = 150
W = 200 

cut_ind = 135


print('loading images')

#### MP4 Method #############
vid = cv2.VideoCapture('../../data/' + vid_name + '.mp4')
small_frames = []
while True:
    ret, im = vid.read() # .convert('RGB')
    if ret == False:
        break
    im = cv2.resize(im.mean(-1), (W,H), None, None)
    small_frames.append(im)

b_img = small_frames[50]
small_frames = small_frames[cut_ind:-5]
frames = len(small_frames)
data_mat = np.stack(list(map(lambda x: np.reshape(x,(H*W,1)),small_frames)),axis=2)[:,:,0]


fig, axs = plt.subplots(3, 3, figsize=(13, 10))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(small_frames[i], cmap='gray')
    ax.axis("off")

fig.tight_layout()


# original one person walking
# best: lam1: 5.0, lam2: 5e-2
# [0.05, 2.0, 0.0005, 1e-05]

# two person walking, 24 iters
rho = 5e-3
gam = 5e-6
lam1 = 2.0
lam2 = 1e-3

# one person walking, 24 iters
# rho = 5e-3
# gam = 5e-6
# lam1 = 2.0
# lam2 = 3e-3


L, S, scores = admm(np.reshape(data_mat,(H,W,-1)),
                    laplace.s_laplace(np.reshape(data_mat,(H,W,-1))),
                    rho,gam,lam1,lam2,b_img=b_img,thresh=0.0001,iters=24)


# format background images
L = np.array(L)
L = L.reshape((H,W,-1))

# format foreground images
S = np.array(S)
S = S.reshape((H,W,-1))



ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[ind], cmap='gray')

    background = L[:, ind].reshape(small_frames[ind].shape)
    foreground = S[:, ind].reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray')
    axs[1, i].set_title("ADMM, L")
    axs[2, i].imshow(foreground, cmap='gray')
    axs[2, i].set_title("ADMM, S")

fig.tight_layout()
plt.savefig("./admm_" + vid_name + ".pdf")






