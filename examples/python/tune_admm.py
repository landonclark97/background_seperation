
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


f_im_list = glob.glob('../../data/'+vid_name+'_mask/*.png')
f_im_list = sorted(f_im_list)

f_mask = np.empty((H,W,len(f_im_list)))
for i, ims in enumerate(f_im_list):
    f_im = cv2.imread(ims)
    im = cv2.resize(f_im.mean(-1), (W,H), None, None)
    f_mask[:,:,i] = im


# b_img = small_frames[50]
small_frames = small_frames[cut_ind:-5]
frames = len(small_frames)
data_mat = np.stack(list(map(lambda x: np.reshape(x,(H*W,1)),small_frames)),axis=1)


T = np.empty((4**4,6))


rho = 1e-2
gam = 1e-5
lam1 = 5.0
lam2 = 1e-3

s_lap = laplace.s_laplace(np.reshape(data_mat,(H,W,-1)))

i = 0
for rho in [0.1, 0.05, 0.01, 0.005]:
    for gam in [1e-4, 5e-5, 1e-5, 1e-6]:
        for lam1 in [100.0, 20.0, 5.0, 2.0]:
            for lam2 in [1e-2, 5e-3, 1e-3, 5e-4]:
                L, S, scores = admm(np.reshape(data_mat,(H,W,-1)),s_lap,
                                    rho,gam,lam1,lam2,f_imgs=f_mask,thresh=0.0001,iters=40)

                print(min(scores))
                T[i,:] = np.array([rho,gam,lam1,lam2,min(scores),np.argmin(scores)])
                i += 1


with open('results.npy', 'wb') as f:
    np.save(f, T)
