
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


SIGMA = 1e-3


# convert pytorch sparse matrix to scipy sparse matrix
def tsp_to_nsp(A):
    return scipy.sparse.csr_matrix((A.coalesce().values().detach().numpy(),A.coalesce().indices().detach().numpy()),shape=(A.shape[0],A.shape[1]))





def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100.0
    max_pixel = 1.0
    psnr = 20.0 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



# Other Methods for RPCA
# https://github.com/sverdoot/robust-pca


vid_name = 'robot_reach'
# vid_name = 'two_persons_walking'
# vid_name = 'human_rob_int'

H = 150
W = 200
# H = 200
# W = 150

cut_ind = 135
# cut_ind = 7
# cut_ind = 0

# two persons
crop_left = 0
crop_top = 0
crop_right = 0
crop_bottom = 0

# robot reach crop
# crop_left = 15
# crop_top = 500
# crop_right = 0
# crop_bottom = 300

# human robot interaction
# crop_left = 20
# crop_top = 507
# crop_right = 0
# crop_bottom = 320



print('loading images')

#### MP4 Method #############
####

USE_PNG = True

### * MAKE SURE TO TAKE OUT MOTIONLESS FRAMES

if USE_PNG:
    im_list = glob.glob('../../data/'+vid_name+'_motionless/*.png')
    # im_list = sorted(im_list, key=lambda x: int(x.replace('.','').replace('/','').replace('png',''),base=36))
    # im_list = sorted(im_list)

    frames = len(im_list)
    small_frames = np.empty((H,W,frames))
    for f in range(1,frames+1):
        img = cv2.imread('../../data/'+vid_name+'_motionless/frame'+str(f)+'.png')
        small_frames[:,:,f-1] = cv2.resize(img.mean(-1), (W,H), None, None)
    data_mat = np.reshape(small_frames,(H*W,-1))
    frames = data_mat.shape[1]
    # b_img = small_frames[:,:,0]
    vid = cv2.VideoCapture('../../data/' + vid_name + '.mp4')
    _, im = vid.read() # .convert('RGB')
    b_img = im[crop_top:im.shape[0]-crop_bottom,crop_left:im.shape[1]-crop_right,:]
    b_img = cv2.resize(b_img.mean(-1), (W,H), None, None)
    del vid



else:
    # temp_thresh = 1450.0
    temp_thresh = 0.0

    vid = cv2.VideoCapture('../../data/' + vid_name + '.mp4')
    small_frames = []
    _, im = vid.read() # .convert('RGB')
    im = im[crop_top:im.shape[0]-crop_bottom,crop_left:im.shape[1]-crop_right,:]
    im = cv2.resize(im.mean(-1), (W,H), None, None)
    small_frames.append(im)
    i=1
    while True:
        ret, im = vid.read() # .convert('RGB')
        if ret == False:
            break
        im = im[crop_top:im.shape[0]-crop_bottom,crop_left:im.shape[1]-crop_right,:]
        im = cv2.resize(im.mean(-1), (W,H), None, None)
        diff = np.linalg.norm((small_frames[-1]-im).flatten())
        if diff > temp_thresh:
            i+=1
            small_frames.append(im)

    b_img = small_frames[0]
    small_frames = small_frames[cut_ind:-5]
    frames = len(small_frames)
    data_mat = np.stack(list(map(lambda x: np.reshape(x,(H*W,1)),small_frames)),axis=1)


addon_b_img = np.repeat(b_img[:,:,np.newaxis], 25, axis=2)
addon_b_img = np.reshape(addon_b_img, (b_img.shape[0]*b_img.shape[1],25))

data_mat = np.concatenate((addon_b_img, data_mat), axis=1)

data_mat += np.random.normal(0.0,SIGMA,data_mat.shape)

# f_im_list = glob.glob('../../data/'+vid_name+'_mask/*.png')
# f_im_list = sorted(f_im_list)

# f_mask = np.empty((H,W,len(f_im_list)))
# for i, ims in enumerate(f_im_list):
#     f_im = cv2.imread(ims)
#     im = cv2.resize(f_im.mean(-1), (W,H), None, None)
#     f_mask[:,:,i] = im




# original one person walking
# best: lam1: 5.0, lam2: 5e-2
# [0.05, 2.0, 0.0005, 1e-05]

# two person walking, 24 iters
# rho = 1e-1
# gam = 1e-5
# lam1 = 0.5
# lam2 = 1e-3

# one person walking, 100 iters
# rho: 1e-2
# lam2: 5e-4

# robot reach
rho = 1e-2
gam = 5e-6
lam1 = 5.0
lam2 = 5e-2

# human robot interaction
# rho = 0.05
# gam = 1e-5
# lam1 = 0.1
# lam2 = 1e-4

mu = 0
alp = 0

L, S, scores = admm(np.reshape(data_mat,(H,W,-1)),
                    laplace.s_laplace(np.reshape(data_mat,(H,W,-1))),
                    rho,gam,lam1,lam2,mu,alp,b_img=b_img,thresh=0.0001,iters=80)
                    # rho,gam,lam1,lam2,f_imgs=f_mask,thresh=0.0001,iters=100)

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
    axs[0, i].imshow(small_frames[ind], cmap='gray', vmin=Dmin, vmax=Dmax)

    background = L[:,:, ind]#.reshape(small_frames[ind].shape)
    foreground = S[:,:, ind]#.reshape(small_frames[ind].shape)
    axs[1, i].imshow(background, cmap='gray', vmin=Lmin, vmax=Lmax)
    axs[1, i].set_title("ADMM, L")
    axs[2, i].imshow(np.abs(foreground), cmap='gray', vmin=0.0, vmax=Smax)
    axs[2, i].set_title("ADMM, S")

fig.tight_layout()
plt.savefig("./admm_" + vid_name + ".pdf")


with open('./L/admm_'+vid_name+'_L.npy', 'wb') as f:
    np.save(f, L)

with open('./S/admm_'+vid_name+'_S.npy', 'wb') as f:
    np.save(f, S)
