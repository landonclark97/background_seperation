import torch
import numpy as np
import scipy
import scipy.sparse
import scipy.linalg as lin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def admm(D,Phi_s,rho,gam,lam1,lam2,b_img=None,f_imgs=None,step=0.1,thresh=0.001,iters=100):
    assert not ((b_img is not None) and (f_imgs is not None))
    if isinstance(D, np.ndarray):
        D = torch.tensor(D).double().to(device)
    scale = torch.max(D).item()
    D /= scale
    if b_img is not None:
        b_img /= np.amax(b_img)
    if f_imgs is not None:
        f_imgs /= np.amax(f_imgs)

    D = torch.reshape(D,(D.shape[0]*D.shape[1],D.shape[2])).cpu().detach().numpy()

    n = int(D.shape[0])
    t = int(D.shape[1])

    L = np.copy(D) # None
    S = np.zeros((n,t)) # None
    U = np.copy(L) # None
    L_tilda = np.copy(S)

    Dt = -np.identity(t) + np.eye(t,k=-1)
    Dt[0,t-1] = 1.0
    DtDtT = Dt@Dt.T

    scores = []

    def shrink(G,mu):
        if scipy.sparse.issparse(G):
            G = G.todense()
        g_max = np.maximum(np.abs(G)-mu,0)
        g_s = np.sign(G)
        return np.multiply(g_s,g_max)

    for i in range(iters):
        print(f'\riter: {i}', end='')

        L_prev = np.copy(L)
        S_prev = np.copy(S)

        U_L_tilda = U+L_tilda
        S_D = S-D
        
        
        for g in range(50):
            Phi_s_L = torch.sparse.mm(Phi_s.to(device), torch.tensor(L).to(device)).cpu().detach().numpy()
            F = (L + S_D) + (gam*(Phi_s_L @ DtDtT)) + (rho*(L - U_L_tilda))
            L = L - (step*F)

        # S subproblem
        D_L = D-L
        DL_n = np.mean(D_L)
        if i % 3 == 0:
            if DL_n < lam2/2.0:
                lam2 /= 2.0
            # elif DL_n > lam2*2.0:
                # lam2 *= 2.0
        S = shrink(D_L,lam2)

        # U subproblem
        A, Sig, B = np.linalg.svd(L-L_tilda,full_matrices=False)
        Sig_n = np.mean(Sig)
        if i % 3 == 0:
            if Sig_n < (lam1/rho)/2.0:
                lam1 /= 2.0
            # elif Sig_n > (lam1/rho)*2.0:
                # lam1 *= 2.0
        Sig_tilda = shrink(Sig,lam1/rho)
        Sig_tilda = scipy.sparse.diags(Sig_tilda,format='csr')
        U = A @ Sig_tilda @ B

        L_tilda = L_tilda + (U-L)

        if b_img is not None:
            L_this = np.array(L)
            L_this = L_this.reshape((150,200,-1))

            L_mean = L_this[:,:,L_this.shape[2]//2]

            diff = b_img-L_mean
            score = np.linalg.norm(diff,ord='fro')/np.linalg.norm(b_img,ord='fro')

            scores.append(score)

        if f_imgs is not None:
            S_this = np.array(S)
            S_this = S_this.reshape((150,200,-1))

            diff = f_imgs-np.abs(S_this)
            # score = np.linalg.norm(diff,ord='fro')/np.linalg.norm(b_img,ord='fro')
            score = np.mean(diff)

            scores.append(score)


        a = np.linalg.norm(L-L_prev,ord='fro')/np.linalg.norm(L_prev,ord='fro')
        b = np.linalg.norm(S-S_prev,ord='fro')/np.linalg.norm(S_prev,ord='fro')
        if (a < thresh) and (b < thresh):
            print(f'leaving on iteration: {i}')
            break

    if b_img is not None:
        return L*scale, S*scale, scores
    return L*scale, S*scale
