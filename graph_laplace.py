import numpy as np


import matplotlib.pyplot as plt





def adjac(V,sigma,k=0,E=None):
    l = V.shape[0]
    A = np.empty((l,l))
    if not E:
        if k == 0:
            mn = np.inf
            for i in range(l):
                for j in range(l):
                    d = np.abs(V[i]-V[j])
                    if d < mn and np.abs(d) > 1e-6:
                        mn = d
            for i in range(l):
               for j in range(l):
                   A[i,j] = np.exp(-(np.abs(V[i]-V[j])-d)/sigma**2.0)
        else:
           for i in range(l):
               for j in range(l):
                   dist = np.abs(V[i]-V[j])
                   if dist < k:
                       A[i,j] = np.abs(V[i]-V[j])
                   else:
                       A[i,j] = 0.0

        return A

    else:
        print('W only defined for fully connected graphs')

def deg(W):
    l = W.shape[0]
    D = np.zeros((W.shape))
    for i in range(l):
        D[i,i] = np.sum(W[:,i])
    return D

# s = np.array([2.0,1.2,2.4,3.2,1.2,5.4,6.5,4.3,2.1,2.4,1.2])
s = np.linspace(0.0,1.0,num=10)



A = adjac(s,1.0)
D = deg(A)


L = D-A

print(L)

e, v = np.linalg.eig(L)

print(np.linalg.eig(L))


plt.plot(s,e)

plt.show()
