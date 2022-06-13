import torch
import torch.optim as opt
from torchvision import transforms
import numpy as np
import scipy
import scipy.sparse
import scipy.linalg as lin
import __laplace

import matplotlib.pyplot as plt

import glob
from PIL import Image

from memory_profiler import profile


def tsp_to_nsp(A):
    return scipy.sparse.csr_matrix((A.coalesce().values().detach().numpy(),A.coalesce().indices().detach().numpy()),shape=(A.shape[0],A.shape[1]))

# @profile
def back_sep_w_admm(D,Phi_s,Phi_t,rho,gam1,gam2,lam1,lam2,step=0.1,thresh=0.001,iters=100):

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
        # print('run:',i)

        # print(D)
        # print(L)
        # print(S)
        print(np.linalg.norm(L+S-D,ord='fro'))

        # '''
        L_print = np.array(L).reshape((150,200,-1))
        S_print = np.array(S).reshape((150,200,-1))

        p = L_print[:,:,40]
        plt.title('L: ' + str(i))
        # plt.imshow(p, cmap='gray', interpolation='nearest', vmin=0.0, vmax=255.0)
        # plt.show()
        plt.imsave('./data/rpca/L_frame'+str(i)+'.png', p, cmap='gray', vmin=0.0, vmax=255.0)


        q = S_print[:,:,40]
        plt.title('S: ' + str(i))
        # plt.imshow(q, cmap='gray', interpolation='nearest', vmin=0.0, vmax=255.0)
        # plt.show()
        plt.imsave('./data/rpca/S_frame'+str(i)+'.png', q, cmap='gray', vmin=0.0, vmax=255.0)
        # '''

        S_prev = np.copy(S)
        L_prev = np.copy(L)

        # L subproblem
        # A = (1.0+gam2+rho)*scipy.sparse.identity(n,format='csr') + gam1*Phi_s
        # B = gam2*Phi_t
        # Q = D - S + rho*(U+L_tilda)

        s = step

        for g in range(60):
            L_grad_prev = np.copy(L)
            F = (L+S-D) + gam1*Phi_s@L + gam2*L@Phi_t + rho*(L-U-L_tilda)
            L = L - s*F
            if np.linalg.norm(L-L_grad_prev,ord='fro')/np.linalg.norm(L_grad_prev,ord='fro') < 1e-4:
                break

        # S subproblem
        S = D-L
        if i == 2:
            print(np.mean(S))
        S = shrink(S,lam2)

        # U subproblem
        A, Sig, B = np.linalg.svd(L-L_tilda,full_matrices=False)
        Sig = scipy.sparse.diags(Sig,format='csr')
        # print(np.sort(Sig.todense())[-10:-1])
        Sig_tilda = shrink(Sig,lam1/rho)
        U = A @ Sig_tilda @ B.T

        L_tilda = L_tilda + (U-L)

        a = np.linalg.norm(L-L_prev,ord='fro')/np.linalg.norm(L_prev,ord='fro')
        # print('relative L error:',a)
        b = np.linalg.norm(S-S_prev,ord='fro')/np.linalg.norm(S_prev,ord='fro')
        # print('relative S error:',b)
        if (a < thresh) and (b < thresh):
            break

    return L, S


resize = transforms.Resize((150,200))
gray = transforms.Grayscale()

root_dir = "./data/walking_data/"

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


bckgnd_img = Image.open('./data/walking_bgnd.png')
b_img = transforms.functional.pil_to_tensor(bckgnd_img).float()
b_img = resize(b_img)
b_img = gray(b_img)
b_img = torch.transpose(b_img,0,2)
b_img = torch.transpose(b_img,0,1)
b_img = b_img.squeeze(2)
b_img = b_img.detach().numpy()


# Lt = laplace.t_laplace(d).to_sparse_csr()
# Ls = laplace.s_laplace(d).to_sparse_csr()
Lt = __laplace.t_laplace(d/255.0)
Ls = __laplace.s_laplace(d/255.0)

# Ls, Lt = _laplace.laplacians(d/255.0)

Lt = tsp_to_nsp(Lt)
Ls = tsp_to_nsp(Ls)


print(Lt.shape)
print(Ls.shape)

low_score = np.inf
low_comb = None

'''
for rho in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
    for lam1 in [1000.0, 100.0, 10.0, 1.0, 0.1]:
        for lam2 in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            for gam1 in [0.1, 0.01, 0.01, 0.0001, 0.00001]:
                for gam2 in [0.1, 0.01, 0.001, 0.0001, 0.00001]:

                    L, S = back_sep_w_admm(d,Ls,Lt,rho,gam1,gam2,lam1,lam2,thresh=0.0001,iters=35)

                    L = np.array(L)
                    L = L.reshape((150,200,-1))
                    L_mean = np.mean(L,axis=2)
                    # S = np.array(S)

                    diff = b_img-L_mean
                    score = np.linalg.norm(diff,ord='fro')/np.linalg.norm(b_img,ord='fro')

                    print([rho, lam1, lam2, gam1, gam2])
                    print(score)

                    if score < low_score:
                        low_score = score
                        low_comb = [rho, lam1, lam2, gam1, gam2]


print('lowest score:',low_score)
print('lowest combination:',low_comb)


'''

rho = 1e-4
gam1 = 1e-6
gam2 = 1e-5
lam1 = 2.0
lam2 = 0.1

L, S = back_sep_w_admm(d,Ls,Lt,rho,gam1,gam2,lam1,lam2,thresh=0.0001,iters=35)

L = np.array(L)
L = L.reshape((150,200,-1))
S = np.array(S)
S = S.reshape((150,200,-1))

for i in range(10):
    l = L[:,:,i]
    plt.title('L: ' + str(i))
    plt.imshow(l, cmap='gray', interpolation='nearest', vmin=0.0, vmax=255.0)
    plt.show()


    s = S[:,:,i]
    plt.title('S: ' + str(i))
    plt.imshow(s, cmap='gray', interpolation='nearest', vmin=0.0, vmax=255.0)
    plt.show()


LS = L+S
plt.imshow(LS, cmap='gray', interpolation='nearest')
plt.show()
