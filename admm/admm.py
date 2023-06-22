import torch
import numpy as np
import scipy
import scipy.sparse
import scipy.linalg as lin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def admm(D,
         Phi_s,
         rho=0.1,
         gam=0.1,
         lam1=50.0,
         lam2=50.0,
         beta=1.5,
         K=10,
         mu=0.0,
         alp=1.0,
         normalized=True,
         b_img=None,
         f_imgs=None,
         step=0.5,
         grad_thresh=1e-3,
         thresh=1e-4,
         iters=100):

    assert not ((b_img is not None) and (f_imgs is not None))
    if isinstance(D, np.ndarray):
        D = torch.tensor(D).double().to(device)
    if not normalized:
        scale = 255.0
    else:
        scale = torch.max(D).item()
    D /= scale
    if (b_img is not None) and (not normalized):
        b_img /= 255.0
    elif b_img is not None:
        b_img /= scale
    if f_imgs is not None:
        f_imgs /= np.amax(f_imgs)

    ids = torch.cat((torch.arange(Phi_s.shape[0]).unsqueeze(0),torch.arange(Phi_s.shape[0]).unsqueeze(0)),0)

    Phi_s = Phi_s + (mu*torch.sparse_coo_tensor(ids,torch.ones(Phi_s.shape[0]),Phi_s.shape))
    Phi_s.coalesce()

    Phi_s = Phi_s**alp
    Phi_s.coalesce()

    D = torch.reshape(D,(D.shape[0]*D.shape[1],D.shape[2])).cpu().detach().numpy()

    n = int(D.shape[0])
    t = int(D.shape[1])

    S = np.zeros((n,t))
    L = D-S
    U = np.copy(L)
    L_tilda = np.copy(S)

    def w(k):
        return ((-1.0)**k)*scipy.special.binom(beta,k)

    Dt = np.identity(t)*w(0)
    for k in range(1,K):
        Dt += np.eye(t,k=-k)*w(k)
        Dt += np.eye(t,k=t-k)*w(k)

    DtDtT = Dt@Dt.T

    scores = []

    def shrink(G,mu):
        if scipy.sparse.issparse(G):
            G = G.todense()
        g_max = np.maximum(np.abs(G)-mu,0)
        g_s = np.sign(G)
        return np.multiply(g_s,g_max)

    for i in range(iters):
        print(f'iter: {i}')

        L_prev = np.copy(L)
        S_prev = np.copy(S)

        U_L_tilda = U+L_tilda
        S_D = S-D
        
        
        for g in range(50):
            with torch.no_grad():
                Phi_s_L = torch.sparse.mm(Phi_s.to(device), torch.tensor(L).to(device)).cpu().detach().numpy()
            F = (L + S_D) + (gam*(Phi_s_L @ DtDtT)) + (rho*(L - U_L_tilda))
            L = L - (step*F)
            if np.linalg.norm(step*F,ord='fro') < grad_thresh:
                break

        # S subproblem
        D_L = D-L
        lam2_shrink = np.percentile(np.abs(D_L), lam2)
        S = shrink(D_L,lam2_shrink)

        # U subproblem
        A, Sig, B = np.linalg.svd(L-L_tilda,full_matrices=False)
        lam1_shrink = np.percentile(Sig, lam1)
        Sig_tilda = shrink(Sig,lam1_shrink)
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
            S_this = np.where(np.abs(S_this) > 0.05, 1.0, 0.0)

            diff = np.square(np.abs(f_imgs)-S_this)
            score = np.mean(diff)
            scores.append(score)

        print('____________________________________________________')
        print(f'|| {score:.6f} :||: {lam1_shrink:.6f}, {lam2_shrink:.6f}, {gam:.3f}, {rho:.3f} ||')
        print('____________________________________________________')


        a = np.linalg.norm(L-L_prev,ord='fro')/np.linalg.norm(L_prev,ord='fro')
        b = np.linalg.norm(S-S_prev,ord='fro')/np.linalg.norm(S_prev,ord='fro')
        if (a < thresh) and (b < thresh):
            print(f'leaving on iteration: {i}')
            break

    if (b_img is not None) or (f_imgs is not None):
        return L*scale, S*scale, scores
    return L*scale, S*scale
