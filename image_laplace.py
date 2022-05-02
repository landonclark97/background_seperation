import numpy as np

import torch
from torchvision import transforms
from torch import nn
from torch.optim import AdamW

import torch_sparse

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
cpu = torch.device("cpu")
print("using device:", device)


import matplotlib.pyplot as plt

import glob
from PIL import Image

import random

root_dir = "./data/BootStrapping_ds/"

# d = torch.empty((480,640,0))
d = torch.empty((224,224,0))

print('loading images')
'''
i_files = [f for f in glob.glob(root_dir+"input/"+"*.png")]
i_files.sort()
d_files = [f for f in glob.glob(root_dir+"depth/"+"*.png")]
d_files.sort()
for i_f, d_f in zip(i_files, d_files):
    im = Image.open(i_f) # .convert('RGB')
    i_d = transforms.functional.pil_to_tensor(im).float()
    im = Image.open(d_f) #.convert('RGB')
    d_d = transforms.functional.pil_to_tensor(im).float()/24.0
    i_d = torch.transpose(i_d,0,2)
    i_d = torch.transpose(i_d,0,1)
    d_d = torch.transpose(d_d,0,2)
    d_d = torch.transpose(d_d,0,1)
    d = torch.cat((d,i_d,d_d),2)
    break
'''

i_files = [f for f in glob.glob("./data/*.png")]
i_files.sort()
for i_f in i_files:
    im = Image.open(i_f) # .convert('RGB')
    i_d = transforms.functional.pil_to_tensor(im).float()
    i_d = torch.transpose(i_d,0,2)
    i_d = torch.transpose(i_d,0,1)
    d = torch.cat((d,i_d),2)
    break


IMG_WIDTH = d.shape[0]
IMG_HEIGHT = d.shape[1]
IMG_PIX = d.shape[2]

STRIDE = 1
PAD_LEN = 2
WIN_LEN = 2

PAD_OFF = (STRIDE*WIN_LEN)+PAD_LEN

THETA = 1.0

d = transforms.Pad(PAD_OFF,padding_mode='symmetric')(d.T)
d = d.T
d = d.to(device)


def pix_dist(X,Y,theta):
    cnt = X.shape[1]*X.shape[2] # X and Y should be same shape
    vec_norm = torch.norm(X-Y,2,dim=3)
    e = torch.exp(-(torch.mean(torch.mean(vec_norm,dim=2),dim=1))/2*theta**2)
    return e


def x_index(i):
    return (i % ((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE))+PAD_LEN

def y_index(i):
    return (i//((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE))+PAD_LEN

def j_index(x,y):
    return (x-WIN_LEN)+((y-PAD_LEN)*(IMG_WIDTH-PAD_LEN-PAD_LEN))

@torch.no_grad()
def spat_laplace(data):
    width = ((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE)
    height = ((IMG_HEIGHT-PAD_LEN-PAD_LEN)//STRIDE)
    N = width*height
    vals = dict()
    # diff = torch.zeros((N,N),dtype=torch.float32).to(cpu)

    for p in range(N):
        print('\rrun:',p,len(vals.keys()),len(vals.values()),sys.getsizeof(vals),end='')
        x = ((p%(width))*STRIDE)+PAD_LEN+PAD_OFF  #x_index(p_i)
        y = ((p//(width))*STRIDE)+PAD_LEN+PAD_OFF  #y_index(p_i)

        # x_len = (min(x+(WIN_LEN*STRIDE),((width*STRIDE)+PAD_LEN))-x)//STRIDE
        x_len = WIN_LEN
        # y_len = (min(y+(WIN_LEN*STRIDE),((height*STRIDE)+PAD_LEN))-y)//STRIDE
        y_len = WIN_LEN

        Bi = torch.empty((((2*x_len)+1)*((2*y_len)+1),(2*PAD_LEN)+1,(2*PAD_LEN)+1,IMG_PIX),dtype=torch.float32).to(device)
        Bj = torch.empty((((2*x_len)+1)*((2*y_len)+1),(2*PAD_LEN)+1,(2*PAD_LEN)+1,IMG_PIX),dtype=torch.float32).to(device)

        for i in range(-x_len,x_len+1):
            for j in range(-y_len,y_len+1):
                i_s = i*STRIDE
                j_s = j*STRIDE
                Bi[i+(x_len*j),:,:,:] = data[x-PAD_LEN:x+PAD_LEN+1,y-PAD_LEN:y+PAD_LEN+1,:]
                Bj[i+(x_len*j),:,:,:] = data[x+i_s-PAD_LEN:x+i_s+PAD_LEN+1,y+j_s-PAD_LEN:y+j_s+PAD_LEN+1,:]

        B_diff = pix_dist(Bi,Bj,THETA).to(cpu)

        for b in range(B_diff.shape[0]):
            b -= WIN_LEN*WIN_LEN
            off = (b%x_len)+((b//x_len)*(width))
            if off != 0 and B_diff[b+(WIN_LEN*WIN_LEN)] != 0:
                vals[(p,p+off)] = -B_diff[b+(WIN_LEN*WIN_LEN)]
                vals[(p+off,p)] = -B_diff[b+(WIN_LEN*WIN_LEN)]
                # diff[p,p+off] = B_diff[b]
                # diff[p+off,p] = B_diff[b]

    k = torch.transpose(torch.tensor(list(vals.keys())),0,1).tolist()
    v = torch.tensor(list(vals.values())).tolist()

    diff = torch.sparse_coo_tensor(k,v,(N, N))
    diff = diff.coalesce()

    s = torch.sparse.sum(diff,0)
    s = s.coalesce()
    sum_val = s.values().tolist()

    vals = dict()

    for j, ind in enumerate(s.indices()[0]):
        vals[(ind.item(),ind.item())] = -sum_val[j]

    k = torch.transpose(torch.tensor(list(vals.keys())),0,1)
    k = torch.cat((diff.indices(),k),1).tolist()

    v = torch.tensor(list(vals.values()))
    v = torch.cat((diff.values(),v)).tolist()

    diff = torch.sparse_coo_tensor(k,v,(N, N))
    diff = diff.coalesce()

    print(sys.getsizeof(diff))

    return diff


with torch.no_grad():


    print('finding weight matrix')
    w = spat_laplace(d)

    print('is symmetric:', (w.transpose(0,1)==w).all().item())
    print(w)

    width = ((IMG_WIDTH-PAD_LEN-PAD_LEN)//STRIDE)
    height = ((IMG_HEIGHT-PAD_LEN-PAD_LEN)//STRIDE)

    print('finding eigenvalues')
    E,l = torch.linalg.eigh(w.to(torch.float32).to(device))

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
