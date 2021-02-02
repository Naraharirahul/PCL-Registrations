import numpy as np
from scipy.spatial.distance import cdist

def transform(A,B):

    assert len(A) == len(B)

    centroid_A = np.mean(A,axis=0)
    centroid_B = np.mean(B,axis=0)

    #Translating the points to its centeroid
    AA = A - centroid_A
    BB = B - centroid_B

    #rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    
    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t