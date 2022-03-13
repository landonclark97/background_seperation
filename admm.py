import torch
import numpy as np
import scipy
import scipy.linalg as lin


def back_sep_w_admm(D,Phi_s,Phi_t,rho,lam1,lam2,gam1,gam2,iters=100):

    n = D.shape[0]
    t = D.shape[1]

    '''
    Phi_t = np.random.rand(t,t)
    for i in range(t):
        Phi_t[i,i] = 0.0
    Phi_t = (Phi_t+Phi_t.T)/2.0
    for i in range(t):
        Phi_t[i,i] = np.sum(Phi_t[i,:])

    Phi_s = np.random.rand(n,n)
    for i in range(n):
        Phi_s[i,i] = 0.0
    Phi_s = (Phi_s+Phi_s.T)/2.0
    for i in range(n):
        Phi_s[i,i] = np.sum(Phi_s[i,:])

    D = np.random.rand(n,t) # None # Data ==> (n,t) matrix
    '''

    L = D # None
    S = np.zeros(n,t) # None
    U = L # None
    L_tilda = S

    def shrink(A,mu):
        return np.multiply(np.sign(A),np.max(np.abs(A)-mu,0))

    for i in range(iters):

        S_prev = S
        L_prev = L

        # L subproblem
        A = (1.0+gam2+rho)*np.identity(n) + gam1*Phi_s
        B = gam2*Phi_t
        Q = D - S + rho*(U+L_tilda)
        L = lin.solve_sylvester(A,B,Q)

        # S subproblem
        A = D-L
        S = shrink(A,lam2)

        # U subproblem
        A, Sig, B = np.linalg.svd(L-L_tilda,full_matrices=False)
        Sig = Sig*np.identity(Sig.shape[0])
        Sig_tilda = shrink(Sig,lam1/rho)
        U = np.matmul(np.matmul(A,Sig_tilda),B.T)

        L_tilda = L_tilda + (U-L)

        if (np.norm(L-L_prev,ord='fro') < thresh) and (np.norm(S-S_prev,ord='fro') < thresh):
            break

    return L, S
