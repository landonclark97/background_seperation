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
def back_sep_w_admm(D,Phi_s,rho,gam,lam1,lam2,back_img,step=0.1,thresh=0.001,iters=100):

    D = torch.reshape(D,(D.shape[0]*D.shape[1],D.shape[2])).detach().numpy()

    n = int(D.shape[0])
    t = int(D.shape[1])

    L = np.copy(D) # None
    S = np.zeros((n,t)) # None
    U = np.copy(L) # None
    L_tilda = np.copy(S)

    Dt = -np.identity(t) + np.eye(t,k=-1)
    Dt[0,t-1] = 1.0

    scores = []

    def shrink(G,mu):
        if scipy.sparse.issparse(G):
            G = G.todense()
        g_max = np.maximum(np.abs(G)-mu,0)
        g_s = np.sign(G)
        return np.multiply(g_s,g_max)

    for i in range(iters):

        print(f'\rrun: {i}', end='')
        #print(np.linalg.norm(L+S-D,ord='fro'))

        #'''
        L_print = np.array(L).reshape((150,200,-1))
        S_print = np.array(S).reshape((150,200,-1))

        p = L_print[:,:,40]
        plt.title('L: ' + str(i))
        # plt.imshow(p, cmap='gray', interpolation='nearest', vmin=0.0, vmax=255.0)
        # plt.show()
        plt.imsave('./data/rpca/L_frame'+str(i)+'.png', p,cmap='gray')
                  #vmin=0.0, vmax=1.0)


        q = S_print[:,:,40]
        plt.title('S: ' + str(i))
        # plt.imshow(q, cmap='gray', interpolation='nearest', vmin=0.0, vmax=255.0)
        # plt.show()
        plt.imsave('./data/rpca/S_frame'+str(i)+'.png', np.abs(q), cmap='gray')
                  #vmin=0.0, vmax=1.0)
        #'''

        S_prev = np.copy(S)
        L_prev = np.copy(L)


        # L subproblem
        # A = (1.0+gam2+rho)*scipy.sparse.identity(n,format='csr') + gam1*Phi_s
        # B = gam2*Phi_t
        # Q = D - S + rho*(U+L_tilda)

        s = step

        for g in range(50):
            L_grad_prev = np.copy(L)
            # F = (L+S-D) + (gam1*(Phi_s@L)) + (gam2*(L@Phi_t)) + (rho*(L-U-L_tilda))
            F = (L+S-D) + (gam*(Phi_s@L@Dt@Dt.T)) + (rho*(L-U-L_tilda))
            L = L - (s*F)
            if np.linalg.norm(L-L_grad_prev,ord='fro')/np.linalg.norm(L_grad_prev,ord='fro') < 1e-4:
                break

        # S subproblem
        S = shrink(D-L,lam2)

        # U subproblem
        A, Sig, B = np.linalg.svd(L-L_tilda,full_matrices=False)
        Sig_tilda = shrink(Sig,lam1/rho)
        Sig_tilda = scipy.sparse.diags(Sig_tilda,format='csr')
        U = A @ Sig_tilda @ B

        L_tilda = L_tilda + (U-L)

        L_this = np.array(L)
        L_this = L_this.reshape((150,200,-1))

        L_mean = np.mean(L_this,axis=2)

        diff = back_img-L_mean
        score = np.linalg.norm(diff,ord='fro')/np.linalg.norm(back_img,ord='fro')

        scores.append(score)

        a = np.linalg.norm(L-L_prev,ord='fro')/np.linalg.norm(L_prev,ord='fro')
        b = np.linalg.norm(S-S_prev,ord='fro')/np.linalg.norm(S_prev,ord='fro')
        if (a < thresh) and (b < thresh):
            break

    return L, S, scores


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


#bckgnd_img = Image.open('./data/walking_bgnd.png')
#b_img = transforms.functional.pil_to_tensor(bckgnd_img).float()
#b_img = resize(b_img)
#b_img = gray(b_img)
#b_img = torch.transpose(b_img,0,2)
#b_img = torch.transpose(b_img,0,1)
#b_img = b_img.squeeze(2)
#b_img = b_img.detach().numpy()
#b_img = b_img/255.0



b_img = np.zeros((150,200))
dnp = d[:,:,[0,-1]].detach().numpy()
b_img[:,0:100] = dnp[:,0:100,0]
b_img[:,100:200] = dnp[:,100:200,1]
b_img = b_img/255.0


Ls = __laplace.s_laplace(d/255.0)
Ls = tsp_to_nsp(Ls)


print(Ls.shape)

low_score = np.inf
low_comb = None



'''
for rho in [0.05, 0.01, 0.005]:
    for lam1 in [10.0, 5.0, 2.0]:
        for lam2 in [0.005, 0.001, 0.0005]:
            for gam in [5e-5, 1e-5, 5e-6]:

                L, S, scores = back_sep_w_admm(d/255.0,Ls,rho,gam,lam1,lam2,b_img,thresh=0.0001,iters=35)

                L = np.array(L)
                L = L.reshape((150,200,-1))
                L_mean = np.mean(L,axis=2)
                # S = np.array(S)

                diff = b_img-L_mean
                score = np.linalg.norm(diff,ord='fro')/np.linalg.norm(b_img,ord='fro')

                print([rho, lam1, lam2, gam])
                print(score)

                if score < low_score:
                    low_score = score
                    low_comb = [rho, lam1, lam2, gam]


print('lowest score:',low_score)
print('lowest combination:',low_comb)

'''



rho = 5e-2
gam = 5e-5
lam1 = 5.0
lam2 = 1e-2

L, S, scores = back_sep_w_admm(d/255.0,Ls,rho,gam,lam1,lam2,b_img,thresh=0.0001,iters=75)

L = np.array(L)
L = L.reshape((150,200,-1))

L_mean = np.mean(L,axis=2)

diff = b_img-L_mean
score = np.linalg.norm(diff,ord='fro')/np.linalg.norm(b_img,ord='fro')

print([rho, lam1, lam2, gam])
print(score)

plt.subplot(2,2,1)
plt.title('L mean')
plt.imshow(L_mean, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)
#plt.show()

plt.subplot(2,2,2)
plt.title('Back img')
plt.imshow(b_img, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)
#plt.show()

plt.subplot(2,2,3)
plt.title('Diff img')
plt.imshow(diff, cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)
#plt.show()

plt.subplot(2,2,4)
plt.title('Scores')
plt.plot([i for i in range(len(scores))],scores)
plt.show()



#'''
