
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
    im = cv2.resize(im, (W,H), None, None)
    small_frames.append(im)

b_img = small_frames[50]
small_frames = small_frames[cut_ind:-5]
frames = len(small_frames)
data_mat = np.stack(list(map(lambda x: np.reshape(x,(H*W,3,1)),small_frames)),axis=2)[:,:,0]


fig, axs = plt.subplots(3, 3, figsize=(13, 10))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(small_frames[i])
    ax.axis("off")

fig.tight_layout()



rd = np.reshape(data_mat[:,0,:], (H,W,3,-1))
gd = np.reshape(data_mat[:,0,:], (H,W,3,-1))
bd = np.reshape(data_mat[:,0,:], (H,W,3,-1))

Ls_r = laplace.s_laplace(rd)
print(torch.any(torch.isnan(Ls_r)))
print(torch.any(torch.isinf(Ls_r)))
#Ls_r = tsp_to_nsp(Ls_r)


Ls_g = laplace.s_laplace(gd)
print(torch.any(torch.isnan(Ls_g)))
print(torch.any(torch.isinf(Ls_g)))
# # Ls_g = tsp_to_nsp(Ls_g)


Ls_b = laplace.s_laplace(bd)
print(torch.any(torch.isnan(Ls_b)))
print(torch.any(torch.isinf(Ls_b)))
# # Ls_b = tsp_to_nsp(Ls_b)


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


L_r, S_r, scores_r = admm(rd,Ls_r,rho,gam,lam1,lam2,b_img=b_img[:,:,0],thresh=0.0001,iters=24)
L_g, S_g, scores_g = admm(gd,Ls_g,rho,gam,lam1,lam2,b_img=b_img[:,:,1],thresh=0.0001,iters=24)
L_b, S_b, scores_b = admm(bd,Ls_b,rho,gam,lam1,lam2,b_img=b_img[:,:,2],thresh=0.0001,iters=24)


# format background images
L_r = np.array(L_r)
L_r = L_r.reshape((H,W,-1))

L_g = np.array(L_g)
L_g = L_g.reshape((H,W,-1))

L_b = np.array(L_b)
L_b = L_b.reshape((H,W,-1))


L = np.empty((H,W,3,L_r.shape[2]))
L[:,:,0,:] = L_r
L[:,:,1,:] = L_g
L[:,:,2,:] = L_b


# format foreground images
S_r = np.array(S_r)
S_r = S_r.reshape((H,W,-1))

S_g = np.array(S_g)
S_g = S_g.reshape((H,W,-1))

S_b = np.array(S_b)
S_b = S_b.reshape((H,W,-1))


S = np.empty((H,W,3,S_r.shape[2]))
S[:,:,0,:] = S_r
S[:,:,1,:] = S_g
S[:,:,2,:] = S_b



ncols = 6
fig, axs = plt.subplots(3, ncols, figsize=(12, 5))

for ax in axs.flatten():
    ax.axis('off')

for i in range(ncols):
    ind = i*(frames//ncols)
    axs[0, i].imshow(small_frames[ind])

    background = L[:, ind].reshape(small_frames[ind].shape)
    foreground = S[:, ind].reshape(small_frames[ind].shape)
    axs[1, i].imshow(background)
    axs[1, i].set_title("ADMM, L")
    axs[2, i].imshow(foreground)
    axs[2, i].set_title("ADMM, S")

fig.tight_layout()
plt.savefig("./admm_col_" + vid_name + ".pdf")








# # extract test image
# L_r_mean = L_r[:,:,75]
# L_g_mean = L_g[:,:,75]
# L_b_mean = L_b[:,:,75]

# L_mean = np.zeros((H,W,3))
# L_mean[:,:,0] = L_r_mean
# L_mean[:,:,1] = L_g_mean
# L_mean[:,:,2] = L_b_mean


# diff = b_img-L_mean
# score = np.linalg.norm(diff,ord='fro')/np.linalg.norm(b_img,ord='fro')



# plt.subplot(2,2,1)
# plt.title('L mean')
# plt.imshow(L_mean, interpolation='nearest', vmin=0.0, vmax=1.0)

# plt.subplot(2,2,2)
# plt.title('Back img')
# plt.imshow(b_img, interpolation='nearest', vmin=0.0, vmax=1.0)

# plt.subplot(2,2,3)
# plt.title('Diff img')
# plt.imshow(np.abs(diff), interpolation='nearest', vmin=0.0, vmax=1.0) #np.min(np.abs(diff[:,:])), vmax=np.max(np.abs(diff[:,:])))

# plt.subplot(2,2,4)
# plt.title('Relative Errors')
# plt.plot([i for i in range(len(scores_r))],scores_r,'r-')
# plt.plot([i for i in range(len(scores_g))],scores_g,'g-')
# plt.plot([i for i in range(len(scores_b))],scores_b,'b-')
# plt.show()


# # normalize foreground between 0 and 1
# S_r = (S_r + 1.0)/2.0
# S_g = (S_g + 1.0)/2.0
# S_b = (S_b + 1.0)/2.0

# S_all = np.empty((H,W,3,S_r.shape[2]))
# S_all[:,:,0,:] = S_r
# S_all[:,:,1,:] = S_g
# S_all[:,:,2,:] = S_b


# plt.subplot(1,1,1)
# plt.imshow(S_all[:,:,:,50], interpolation='nearest',vmin=0.0,vmax=1.0)
# plt.show()
