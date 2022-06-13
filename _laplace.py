import numpy as np

import torch

import matlab
import matlab.engine

eng = matlab.engine.start_matlab()


def laplacians(D):
    d = matlab.double(D.tolist())
    Ls, Lt = eng.compute_laplacian(d, 3.0, 4.0, 1.0, 1.0, nargout=2)
    print(Ls)
    print(Lt)
    quit()
    return Ls, Lt
