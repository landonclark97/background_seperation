import numpy as np

import torch
from torchvision import transforms
from torch import nn
from torch.optim import AdamW


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
cpu = torch.device("cpu")
print("using device:", device)


import matplotlib.pyplot as plt

import glob
from PIL import Image

import random

from memory_profiler import profile

root_dir = "./data/"

d = None

print('loading images')
curr_dir = root_dir
files = [f for f in glob.glob(curr_dir+"*.png")]
for f in files:
    im = Image.open(f).convert('RGB')
    d = transforms.functional.pil_to_tensor(im).float()

d = torch.transpose(d,0,2)
d = torch.transpose(d,0,1)
print(d.shape)

IMG_WIDTH = d.shape[0]
IMG_HEIGHT = d.shape[1]
IMG_PIX = d.shape[2]

STRIDE = 3
PAD_LEN = 2
WIN_LEN = 6

THETA = 1.0

def pix_dist(X,Y,theta):
    cnt = X.shape[1]*X.shape[2] # X and Y should be same shape
    vec_norm = torch.norm(X-Y,2,dim=3)
    return torch.exp(-(torch.mean(torch.mean(vec_norm,dim=2),dim=1))/2*theta**2)


def x_index(i):
    return (i % ((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE))+PAD_LEN

def y_index(i):
    return (i//((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE))+PAD_LEN

def j_index(x,y):
    return (x-WIN_LEN)+((y-PAD_LEN)*(IMG_WIDTH-PAD_LEN-PAD_LEN))

@torch.no_grad()
def W(data):
    width = ((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE)
    height = ((IMG_HEIGHT-PAD_LEN-PAD_LEN)//STRIDE)
    N = width*height
    diff = torch.sparse_coo_tensor(size=(N, N))
    # diff = torch.zeros((N,N),dtype=torch.float32).to(cpu)

    for p in range(N):
        print('\rrun:',p,end='')
        x = ((p%(width))*STRIDE)+PAD_LEN  #x_index(p_i)
        y = ((p//(width))*STRIDE)+PAD_LEN  #y_index(p_i)

        x_len = (min(x+(WIN_LEN*STRIDE),((width*STRIDE)+PAD_LEN))-x)//STRIDE
        y_len = (min(y+(WIN_LEN*STRIDE),((height*STRIDE)+PAD_LEN))-y)//STRIDE

        Bi = torch.empty((x_len*y_len,(2*PAD_LEN)+1,(2*PAD_LEN)+1,IMG_PIX),dtype=torch.float32).to(device)
        Bj = torch.empty((x_len*y_len,(2*PAD_LEN)+1,(2*PAD_LEN)+1,IMG_PIX),dtype=torch.float32).to(device)

        for i in range(x_len):
            for j in range(y_len):
                i_s = i*STRIDE
                j_s = j*STRIDE
                Bi[i+(x_len*j),:,:,:] = data[x-PAD_LEN:x+PAD_LEN+1,y-PAD_LEN:y+PAD_LEN+1,:]
                Bj[i+(x_len*j),:,:,:] = data[x+i_s-PAD_LEN:x+i_s+PAD_LEN+1,y+j_s-PAD_LEN:y+j_s+PAD_LEN+1,:]

        B_diff = pix_dist(Bi,Bj,THETA).to(cpu)

        for b in range(B_diff.shape[0]):
            off = (b%x_len)+((b//x_len)*(width))
            if p+off != p:
                diff[p,p+off] = B_diff[b]
                diff[p+off,p] = B_diff[b]

        x_len = min(((x-PAD_LEN)//STRIDE)+1,WIN_LEN)
        y_len = (min(y+(WIN_LEN*STRIDE),((height*STRIDE)+PAD_LEN))-y)//STRIDE

        Bi = torch.empty((x_len*y_len,(2*PAD_LEN)+1,(2*PAD_LEN)+1,IMG_PIX),dtype=torch.float32).to(device)
        Bj = torch.empty((x_len*y_len,(2*PAD_LEN)+1,(2*PAD_LEN)+1,IMG_PIX),dtype=torch.float32).to(device)

        for i in range(x_len):
            for j in range(y_len):
                i_s = i*STRIDE
                j_s = j*STRIDE
                Bi[i+(x_len*j),:,:,:] = data[x-PAD_LEN:x+PAD_LEN+1,y-PAD_LEN:y+PAD_LEN+1,:]
                Bj[i+(x_len*j),:,:,:] = data[x-i_s-PAD_LEN:x-i_s+PAD_LEN+1,y+j_s-PAD_LEN:y+j_s+PAD_LEN+1,:]

        B_diff = pix_dist(Bi,Bj,THETA).to(cpu)

        for b in range(B_diff.shape[0]):
            off = -(b%x_len)+((b//x_len)*(width))
            if p+off != p:
                diff[p,p+off] = B_diff[b]
                diff[p+off,p] = B_diff[b]

    print()
    return diff

@torch.no_grad()
def L(w):
    k = -1*w
    for i in range(w.shape[0]):
        k[i,i] = torch.sum(w[i,:])
    return k


with torch.no_grad():

    # filt_img = Image.fromarray(d.cpu().detach().numpy(),'RGB')
    # filt_img.show()

    print('finding weight matrix')
    w = W(d)
    l = L(w).to(device)
    del w

    print('is symmetric:', (l.transpose(0,1)==l).all().item())
    print(l)

    width = ((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE)
    height = ((IMG_HEIGHT-PAD_LEN-PAD_LEN)//STRIDE)

    print('finding eigenvalues')
    E,l = torch.linalg.eigh(l.to(torch.float32).to(device))

    done = False
    while not done:
        text = str(input('input index or done:\n'))
        if text == 'done':
            done = True
        else:
            i = eval(text)
            print(E[i].item())
            filt = torch.reshape(l[i,:],(width,height))
            filt_img = Image.fromarray(((filt+1.0)*127.0).byte().cpu().detach().numpy(),'L')
            filt_img.show()




    '''

    degree = np.sum(W, axis=1)
    L = np.zeros(W.shape)
    if(lap_type == 'unn'):
        D = np.diag(degree)
        L = D-W

    elif(lap_type == 'sym'):
        degree_sq_inv = 1/np.sqrt(degree)
        D = np.diag(degree_sq_inv)
        L = np.eye(W.shape[0]) - D.dot(W.dot(D))
    '''
