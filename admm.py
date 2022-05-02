import torch
import torch.optim as opt
from torchvision import transforms
import numpy as np
import scipy
import scipy.linalg as lin
import laplace

import matplotlib.pyplot as plt

import glob
from PIL import Image

from memory_profiler import profile


def tsp_to_nsp(A):
    return scipy.sparse.csr_matrix((A.coalesce().values().detach().numpy(),A.coalesce().indices().detach().numpy()),shape=(A.shape[0],A.shape[1]))

def sparse_identity(n):
    return torch.sparse_coo_tensor([torch.arange(n).int().tolist(),torch.arange(n).int().tolist()],torch.ones(n),(n,n))

def sparse_diag(a):
    return torch.sparse_coo_tensor([torch.arange(a.shape[0]).int().tolist(),torch.arange(a.shape[0]).int().tolist()],a,(a.shape[0],a.shape[0]))

@profile
def back_sep_w_admm(D,Phi_s,Phi_t,rho,lam1,lam2,gam1,gam2,thresh=0.001,iters=100):

    D = torch.reshape(D,(D.shape[0]*D.shape[1],D.shape[2])) #.detach().numpy()

    n = int(D.shape[0])
    t = int(D.shape[1])


    L = torch.clone(D) # None
    S = torch.zeros((n,t)) # None
    U = torch.clone(L) # None
    L_tilda = torch.clone(S)

    def shrink(A,mu):
        a = A.to_dense()
        a = torch.add(-torch.ones((a.shape[0],a.shape[1]))*mu,torch.abs(a))
        a_max = torch.maximum(a,torch.zeros((a.shape[0],a.shape[1])))
        a_s = torch.sign(a)
        return torch.multiply(a_s,a_max)
        # return torch.multiply(torch.sign(A),torch.max(-mu+torch.abs(A),0).values)

    for i in range(iters):
        print('run:',i)

        S_prev = torch.clone(S)
        L_prev = torch.clone(L)

        # L subproblem
        # A = (1.0+gam2+rho)*scipy.sparse.identity(n,format='csr') + gam1*Phi_s
        A = (1.0+gam2+rho)*sparse_identity(n) + gam1*Phi_s
        B = gam2*Phi_t
        Q = D - S + rho*(U+L_tilda)
        # L = lin.solve_sylvester(A,B,Q)
        # F = (L+S-D) + lam1*Psi_s*L + lam2*L*Phi_t + rho*(L-U-L_tilda)
        # L = L - s*F
        x = torch.rand((A.shape[1],B.shape[0]),requires_grad=True)
        o = opt.RMSprop([x],lr=2.0)
        for g in range(70):

            error = (torch.sparse.mm(A,x)+torch.sparse.mm(B.t(),x.t()).t())-Q
            loss = torch.mean(torch.mean(error,0),0)**2
            print('\r'+str(i)+', '+str(g)+', '+str(loss.item()),end ='')
            o.zero_grad()
            loss.backward()
            o.step()
        print()

        L = x
        L = L.requires_grad_(False)
        # print(L)

        # plt.imshow(L.detach().numpy(),interpolation='nearest')
        # plt.show()

        # S subproblem
        A = D-L
        print(A.shape)
        S = shrink(A,lam2)

        # U subproblem
        # A, Sig, B = np.linalg.svd(L-L_tilda,full_matrices=False)
        A, Sig, B = torch.svd_lowrank(L-L_tilda,niter=4)
        # Sig = Sig*scipy.sparse.identity(Sig.shape[0],format='csr')
        print(Sig.shape)
        Sig = sparse_diag(Sig)
        print(Sig.shape)
        Sig_tilda = shrink(Sig,lam1/rho)
        U = torch.sparse.mm(torch.sparse.mm(A,Sig_tilda),B.T)

        L_tilda = L_tilda + (U-L)

        if (torch.norm(L-L_prev,p='fro') < thresh) and (torch.norm(S-S_prev,p='fro') < thresh):
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

print(Lt.shape)
print(Ls.shape)

L, S = back_sep_w_admm(d,Ls,Lt,0.9,0.1,0.1,0.1,0.1,thresh=0.001,iters=100)

plt.imshow(L,interpolation='nearest')
plt.show()

x = input('?')

plt.imshow(S,interpolation='nearest')
plt.show()

x = input('?')
