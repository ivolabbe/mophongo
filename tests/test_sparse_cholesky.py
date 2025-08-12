import time
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import csc_matrix, diags, isspmatrix

from mophongo.fit import make_sparse_chol_prec, sparse_cholesky


def test_sparse_cholesky_preconditioner_solve():
    n = 5
    A = diags([2 * np.ones(n), -1 * np.ones(n - 1), -1 * np.ones(n - 1)], [0, -1, 1], format="csc")
    b = np.ones(n)
    M = make_sparse_chol_prec(A)
    x, info = cg(A, b, M=M, atol=0.0, rtol=1e-12)
    x_true = np.linalg.solve(A.toarray(), b)
    assert info == 0
    np.testing.assert_allclose(x, x_true, rtol=1e-8, atol=1e-8)


def test_sparse_cholesky_preconditioner_properties():
    # ---------------------------
    # Helpers: 1D Laplacian, verification, benchmark
    # ---------------------------
    def laplacian_1d(n: int) -> csc_matrix:
        """Dirichlet 1D Poisson (tridiagonal SPD)."""
        diagonals = [2 * np.ones(n), -1 * np.ones(n - 1), -1 * np.ones(n - 1)]
        return diags(diagonals, [0, -1, 1], format="csc")

    def frob_norm(M):
        if isspmatrix(M):
            return np.linalg.norm(M.toarray(), "fro")
        return np.linalg.norm(M, "fro")

    def verify_against_dense(A: csc_matrix, use_rcm: bool = False):
        Ad = A.toarray()
        t0 = time.perf_counter()
        Ld = np.linalg.cholesky(Ad)
        t_dense = time.perf_counter() - t0
        res_dense = frob_norm(Ld @ Ld.T - Ad)

        t1 = time.perf_counter()
        Ls, p = sparse_cholesky(A, use_rcm=use_rcm)
        t_sparse = time.perf_counter() - t1
        Aperm = A[p, :][:, p]
        res_sparse = frob_norm((Ls @ Ls.T - Aperm).toarray())
        return t_dense, res_dense, t_sparse, res_sparse, Ls.nnz, A.nnz

    def run_bench(sizes, use_rcm: bool = False):
        print(f"Benchmark: 1D Laplacian, sizes={sizes}, use_rcm={use_rcm}")
        header = f"{'n':>6}  {'nnz(A)':>8}  {'nnz(L)':>8}  {'dense_t(s)':>10}  {'sparse_t(s)':>11}  {'||LLᵀ-A||_F dense':>16}  {'||LLᵀ-A||_F sparse':>18}"
        print(header)
        print("-" * len(header))
        for n in sizes:
            A = laplacian_1d(n)
            td, rd, ts, rs, nnzL, nnzA = verify_against_dense(A, use_rcm=use_rcm)
            print(f"{n:6d}  {nnzA:8d}  {nnzL:8d}  {td:10.6f}  {ts:11.6f}  {rd:16.3e}  {rs:18.3e}")

    run_bench([100, 200, 400, 800, 1600])
