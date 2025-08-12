import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg

from mophongo.fit import make_sparse_chol_prec


def test_sparse_cholesky_preconditioner_solve():
    n = 5
    A = diags([2 * np.ones(n), -1 * np.ones(n - 1), -1 * np.ones(n - 1)], [0, -1, 1], format="csc")
    b = np.ones(n)
    M = make_sparse_chol_prec(A)
    x, info = cg(A, b, M=M, atol=0.0, rtol=1e-12)
    x_true = np.linalg.solve(A.toarray(), b)
    assert info == 0
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-8)
