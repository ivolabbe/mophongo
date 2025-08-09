from scipy.sparse import lil_matrix, rand
from time import time as timer
from numpy import array, concatenate, empty

### sparse appending ####
def sparse_append(A):
    dim = A.shape[1]
    mat = lil_matrix(A.shape, dtype = A.dtype)

    sparse_addtime = 0
    i = 0
    for vector in A:
        st = timer()

        mat[i] = vector
        i += 1
        et = timer()
        sparse_addtime += et-st

    return sparse_addtime



#### dense append ####
def dense_append(A):
    dim = A.shape[1]
    mat = empty([0,dim])

    dense_addtime = 0

    for vector in A:
        st = timer()
        mat = concatenate((mat,vector))
        et = timer()
        dense_addtime += et-st

    return dense_addtime



### main ####
if __name__ == '__main__':
    dim = 400
    n = 200

    A = rand(n, dim, density = 0.1, format='lil')
    B = A.todense() #numpy.ndarray

    t1 = sparse_append(A)
    t2 = dense_append(B)

    print t1, t2


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

; alternative
mat=lil_matrix((len(arr),len(arr)))
markov(arr)
f = open('spmatrix.pkl','wb')
cPickle.dump(mat,f,-1)
f.close()


def symmetrize(a):
    return a + a.T - numpy.diag(a.diagonal())

; ok!
def sparse_symmetrize(a):
   return a + a.transpose() - a.diagonal()

def sparseSym(rank, density=0.01, format='coo', dtype=None, random_state=None):
  density = density / (2.0 - 1.0/rank)
  A = scipy.sparse.rand(rank, rank, density=density, format=format, dtype=dtype, random_state=random_state)
  return (A + A.transpose())/2


def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, linalg, spdiags, issparse

A = rand(10000L, 10000L, density = 0.0001, format='csr')
A = 0.5*(A*A.transpose())
A.setdiag(rand(10000L)+0.2)
A.count_nonzero()/1./np.prod(A.shape)

plt.spy(A)
plt.show()

b = np.ones(10000L)
x = linalg.spsolve(A,b)

Ainv_approx = linalg.spilu(A)
Jpre =
x = linalg.cg(A, b, x0=None, tol=1e-05, maxiter=None, xtype=None, M=None, callback=None)





A = rand(10000L, 10000L, density = 0.01, format='lil')
B = 0.5*(A*A.transpose())
A.setdiag(np.ones(10000L))

B=sparseSym(1e4,density=0.01,format='csr')
B.setdiag(np.ones(10000L))
perm = scipy.sparse.csgraph.reverse_cuthill_mckee(B,symmetric_mode=True)
Blil = B.tolil()
C = Blil[perm, perm]
D = C.tocoo()

ax = plot_coo_matrix(D)
ax.figure.show()


; sparse covariance and correlation

import numpy as np
from scipy import sparse

def sparse_corrcoef(A, B=None):

    if B is not None:
        A = sparse.vstack((A, B), format='csr')

    A = A.astype(np.float64)

    # compute the covariance matrix
    # (see http://stackoverflow.com/questions/16062804/)
    A = A - A.mean(1)
    norm = A.shape[1] - 1.
    C = A.dot(A.T.conjugate()) / norm

    # the correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))

    return coeffs


import numpy as np
from scipy import sparse

def dropcols_fancy(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(np.arange(M.shape[1]), idx_to_drop, assume_unique=True)
    return M[:, np.where(keep)[0]]

def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


# Ok, so I'm pretty sure the "right" way to do this is: if you are slicing columns,
# use tocsc() and slice using a list/array of integers. Boolean vectors does not
# seem to do the trick with sparse matrices -- the way it does with ndarrays in
# numpy. Which means the answer is.
indices = np.where(bool_vect)[0]
out1 = M.tocsc()[:,indices]
out2 = M.tocsr()[indices,:]
#A = B.tocsr()[np.array(list1),:].tocsc()[:,np.array(list2)]
#You can see that row'S and col's get cut separately, but each one converted to the fastest sparse format, to get index this time.

from scipy import sparse
from bisect import bisect_left

class lil2(sparse.lil_matrix):
    def removecol(self,j):
        if j < 0:
            j += self.shape[1]

        if j < 0 or j >= self.shape[1]:
            raise IndexError('column index out of bounds')

        rows = self.rows
        data = self.data
        for i in xrange(self.shape[0]):
            pos = bisect_left(rows[i], j)
            if pos == len(rows[i]):
                continue
            elif rows[i][pos] == j:
                rows[i].pop(pos)
                data[i].pop(pos)
                if pos == len(rows[i]):
                    continue
            for pos2 in xrange(pos,len(rows[i])):
                rows[i][pos2] -= 1

        self._shape = (self._shape[0],self._shape[1]-1)

    def removerow(self,i):
        if i < 0:
            i += self.shape[0]

        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index out of bounds')

        self.rows = numpy.delete(self.rows,i,0)
        self.data = numpy.delete(self.data,i,0)
        self._shape = (self._shape[0]-1,self.shape[1])
