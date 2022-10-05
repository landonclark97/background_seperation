import numpy as np
import numpy.matlib

import torch
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def im2col(A, BSZ, stepsize=1):
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]



def pdist2(A, B, n=5):
    pd = torch.empty((A.shape[0],n))
    indices = torch.empty((A.shape[0],n))
    for row in range(A.shape[0]):
        diffs = torch.pow(torch.mean(B-A[row,:],1),2)
        diffs, ind = torch.sort(diffs)
        pd[row,:] = diffs[:n]
        indices[row,:] = ind[:n]
    return pd.T, indices.T


@torch.no_grad()
def t_laplace(d):
    if isinstance(d, np.ndarray):
        d = torch.tensor(d).double().to(device)

    IMG_WIDTH = d.shape[0]
    IMG_HEIGHT = d.shape[1]
    IMG_PIX = d.shape[2]

    N = IMG_WIDTH*IMG_HEIGHT

    KT = 8

    STRIDE = 1
    PAD_LEN = 2
    WIN_LEN = 2

    PAD_OFF = (2*PAD_LEN)+1

    T_THETA = 1.0

    d2 = torch.reshape(d,(N,IMG_PIX))

    dt, ind = pdist2(d2.T,d2.T,n=KT+1)
    tmp = torch.exp(-(dt[1:,:]/T_THETA))
    tmp = torch.reshape(tmp,(tmp.shape[0]*tmp.shape[1],))

    idx = np.reshape(np.matlib.repmat(np.arange(IMG_PIX),KT,1),(IMG_PIX*KT))
    idy = np.reshape(ind[1:,:].detach().numpy(),(IMG_PIX*KT))

    ids = torch.cat((torch.tensor(idx).unsqueeze(0),torch.tensor(idy).unsqueeze(0)),0)

    A = torch.sparse_coo_tensor(ids,tmp,(IMG_PIX,IMG_PIX))
    A = A + torch.transpose(A,0,1)
    A.coalesce()

    D = 1.0/torch.sqrt(torch.sparse.sum(A,1).values())

    ids = torch.cat((torch.arange(IMG_PIX).unsqueeze(0),torch.arange(IMG_PIX).unsqueeze(0)),0)
    W = torch.sparse_coo_tensor(ids,D,(IMG_PIX,IMG_PIX))

    Lt = torch.sparse_coo_tensor([torch.arange(IMG_PIX).int().tolist(),torch.arange(IMG_PIX).int().tolist()],torch.ones(IMG_PIX),(IMG_PIX,IMG_PIX)) - torch.sparse.mm(torch.sparse.mm(W,A),W)
    Lt.coalesce()

    return Lt


@torch.no_grad()
def s_laplace(d):
    if isinstance(d, np.ndarray):
        d = torch.tensor(d).double().to(device)
    d /= torch.max(d)

    IMG_WIDTH = d.shape[0]
    IMG_HEIGHT = d.shape[1]
    IMG_PIX = d.shape[2]

    N = IMG_WIDTH*IMG_HEIGHT

    STRIDE = 1
    PAD_LEN = 4
    WIN_LEN = 2

    PAD_OFF = (2*PAD_LEN)+1
    K = (4*(PAD_LEN**2))+(4*PAD_LEN)

    THETA = 1.0

    dT = d.permute(*torch.arange(d.ndim-1,-1,-1))
    d = transforms.Pad(PAD_LEN,padding_mode='symmetric')(dT)
    d = d.permute(*torch.arange(d.ndim-1,-1,-1))
    d = d.cpu().detach().numpy()

    patches = np.empty((N,PAD_OFF**2,IMG_PIX))
    for i in range(IMG_PIX):
        patches[:,:,i] = im2col(d[:,:,i],(PAD_OFF,PAD_OFF)).T
    patches = np.reshape(patches,(N,(PAD_OFF**2)*IMG_PIX))

    ind = np.reshape(np.arange(N,dtype=np.int_),(IMG_WIDTH,IMG_HEIGHT))
    ind = np.pad(ind,(PAD_LEN,PAD_LEN),mode='symmetric')
    ind = im2col(ind,(PAD_OFF,PAD_OFF)).T
    index = np.concatenate((np.expand_dims(np.arange(N,dtype=np.int_),1),ind[:,np.arange(K/2,dtype=np.int_)],ind[:,np.arange((K/2)+1,ind.shape[1],dtype=np.int_)]),axis=1)

    ds = np.empty((K*N))

    i_d = 0

    for i in range(N):
        for j in range(1,K+1):
            di = patches[i,:]
            dj = patches[index[i,j],:]
            ds[i_d] = np.linalg.norm(di-dj)
            i_d += 1

    tmp = np.exp(-(np.power(ds,2)/THETA))

    idx = np.reshape(np.matlib.repmat(np.arange(N),K,1).T,(N*K))
    idy = np.reshape(index[:,1:],(N*K))

    ids = torch.cat((torch.tensor(idx).unsqueeze(0),torch.tensor(idy).unsqueeze(0)),0)

    A = torch.sparse_coo_tensor(ids,tmp,(N,N))
    A.coalesce()
    A = A + torch.transpose(A,0,1)
    A.coalesce()

    D = 1.0/torch.sqrt(torch.sparse.sum(A,1).values())

    ids = torch.cat((torch.arange(N).unsqueeze(0),torch.arange(N).unsqueeze(0)),0)
    W = torch.sparse_coo_tensor(ids,D,(N,N))

    Ls = torch.sparse_coo_tensor(ids,torch.ones(N),(N,N)) - torch.sparse.mm(torch.sparse.mm(W,A),W)
    Ls.coalesce()

    return Ls
