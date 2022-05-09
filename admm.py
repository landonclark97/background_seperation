import torch
import torch.optim as opt
from torchvision import transforms
import numpy as np
import scipy
import scipy.sparse
import scipy.linalg as lin
import laplace

import matplotlib.pyplot as plt

import glob
from PIL import Image

from memory_profiler import profile


def tsp_to_nsp(A):
    return scipy.sparse.csr_matrix((A.coalesce().values().detach().numpy(),A.coalesce().indices().detach().numpy()),shape=(A.shape[0],A.shape[1]))

# @profile
def back_sep_w_admm(D,Phi_s,Phi_t,rho,lam1,lam2,gam1,gam2,thresh=0.001,iters=100):

    D = torch.reshape(D,(D.shape[0]*D.shape[1],D.shape[2])).detach().numpy()

    n = int(D.shape[0])
    t = int(D.shape[1])


    L = np.copy(D) # None
    S = np.zeros((n,t)) # None
    U = np.copy(L) # None
    L_tilda = np.copy(S)

    def shrink(G,mu):
        if scipy.sparse.issparse(G):
            G = G.todense()
        g_max = np.maximum(np.abs(G)-mu,0)
        g_s = np.sign(G)
        return np.multiply(g_s,g_max)

    for i in range(iters):
        print('run:',i)

        S_prev = np.copy(S)
        L_prev = np.copy(L)

        # L subproblem
        # A = (1.0+gam2+rho)*scipy.sparse.identity(n,format='csr') + gam1*Phi_s
        # B = gam2*Phi_t
        # Q = D - S + rho*(U+L_tilda)

        s = 0.05

        for g in range(1000):

            F = (L+S-D) + lam1*Phi_s@L + lam2*L@Phi_t + rho*(L-U-L_tilda)
            f_norm = np.linalg.norm(F,ord='fro')
            print(f'\r{f_norm}',end='')
            if f_norm < 10.0:
                break
            L = L - s*F

        print()
        # S subproblem
        A = D-L
        S = shrink(A,lam2)

        # U subproblem
        A, Sig, B = np.linalg.svd(L-L_tilda,full_matrices=False)
        Sig = scipy.sparse.diags(Sig,format='csr')
        Sig_tilda = shrink(Sig,lam1/rho)
        U = A@Sig_tilda@B.T

        L_tilda = L_tilda + (U-L)

        a = np.linalg.norm(L-L_prev,ord='fro')
        print(a)
        b = np.linalg.norm(S-S_prev,ord='fro')
        print(b)
        if (a < thresh) and (b < thresh):
            break

    return L, S


resize = transforms.Resize((150,200))
gray = transforms.Grayscale()

root_dir = "./data/human_motion/"

d = torch.empty((150,200,0))

print('loading images')
i_files = [f for f in glob.glob(root_dir+"*.png")]
i_files.sort()
# d_files = [f for f in glob.glob(root_dir+"depth/"+"*.png")]
# d_files.sort()
for i_f in i_files:
    im = Image.open(i_f) # .convert('RGB')
    i_d = transforms.functional.pil_to_tensor(im).float()
    # im = Image.open(d_f) #.convert('RGB')
    # d_d = transforms.functional.pil_to_tensor(im).float()/24.0
    i_d = resize(i_d)
    i_d = gray(i_d)
    # d_d = resize(d_d)
    i_d = torch.transpose(i_d,0,2)
    i_d = torch.transpose(i_d,0,1)
    # d_d = torch.transpose(d_d,0,2)
    # d_d = torch.transpose(d_d,0,1)
    # d = torch.cat((d,i_d,d_d),2)
    d = torch.cat((d,i_d),2)
    # break


# Lt = laplace.t_laplace(d).to_sparse_csr()
# Ls = laplace.s_laplace(d).to_sparse_csr()
Lt = laplace.t_laplace(d/255.0)
Ls = laplace.s_laplace(d/255.0)

Lt = tsp_to_nsp(Lt)
Ls = tsp_to_nsp(Ls)

print(Lt.shape)
print(Ls.shape)

L, S = back_sep_w_admm(d,Ls,Lt,0.9,0.1,0.1,0.1,0.1,thresh=0.001,iters=100)

d = d.detach().numpy()
L = np.lib.stride_tricks.as_strided(L,d.shape,d.strides)
print(L.shape)
L = L[:,:,0]
L_img = Image.fromarray(L,'L')
L_img.show()


S = np.lib.stride_tricks.as_strided(S,d.shape,d.strides)

S = S[:,:,0]
S_img = Image.fromarray(S,'L')
S_img.show()
