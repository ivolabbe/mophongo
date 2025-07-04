; @@@ mask out in phot any area not covered by detection map -> make normal
;
;
;
; Finally: what if, as John suggests, you have to solve Ax=b for many
; different b’s? How do you encode this in R without inverting A?
; ok, so do LU, then update righ hand (background regisration
; and use LU to quickly solve
;
; Imagine rewriting our problem as Ax = LUx = b, then defining y=Ux, so that we can solve it
; in two stages: Ly = b, Ux = y. We can collapse this in R into a single line, in the form
; x = backsolve(U, forwardsolve(L, b))
;
;


function mophongo::solve, basis, ata, atb, at, b
   logger, 'Solving normal equations '

; solve first
   la = python.import('scipy.sparse.linalg')
  ; sol = la.cg(ata, atb, tol=1e-8)  ; as fast as svd..... but that one has rather nice properties
 ;  x = sol[0]
;   status = sol[1]
   lu = la.splu(ata,permc_spec='MMD_AT_PLUS_A')  ; LU decompose, takes 0.07s  2k sources
   x = lu.solve(atb, trans='N')            ;  solve for RHS 0.006s 2k sources

stop
;   uwv = la.svds(ata, k=atb.length-1, tol=1e-8)   ;  4 seconds or so
;   ilu = la.spilu(ata,permc_spec='MMD_AT_PLUS_A')  ; 0.11
; not sure which one is better....
;   x_ilu = ilu.solve(atb)

; add PSF template around brightest sources with strong residuals
; cross correlate to get exact center
;
   ii = where(basis.id eq 6554)
   b_6554 = basis[ii]
   (b_6554.det) = ptr_new(fconvolve(self.getkernel(b_6554.x,b_6554.y),self.getpsf(b_6554.x,b_6554.y)))

; compute shift on selected objects with x-cor + interpolate -> map

; calculate background and subtract

; ??? already analyse covariance map here ?

; recompoute atb, refit

; ??? only here dedice which to remove


stop

   return, sol
end


pro test_sparse

 a= Python.run('from sklearn.linear_model import Ridge')
 a= Python.run('from sklearn.linear_model import Lasso')
  Python.ata = ata
  Python.atb = atb
  >>>clf = Ridge(alpha=3e6, fit_intercept=False, max_iter=None, normalize=False, random_state=None, solver='sparse_cg', tol=1e-8)
  >>>clf = Lasso(alpha=5e2, positive=True, fit_intercept=False,  normalize=False, random_state=None, tol=1e-6)
  >>>clf.fit(ata, atb)
  >>>clf.get_params()
  >>>x=clf.coef_
  >>>clf.intercept_
  x=python.x
print, x_cg[0:5]
print, x[0:5]
   model0 = self.make_model(basis, x_cg, id=id, residual=residual0)
   model0 = self.make_model(basis, x, id=id, residual=residual0)

end

pro test
;
;goal: initialize a sparse matrix with shape = (1000,1000)
;in Python achieved by lil_matrix(shape) where shape = tuple(M,N)_
;
;IDL>
;sparse = python.import('scipy.sparse')
;ata = sparse.lil_matrix(Python.tuple([1000,1000]))
;
;% STRLOWCASE: Object reference expression not allowed in this context: METHODIN.
;
;alternative: bring over to python first
;
;Python.shape = Python.tuple([1000,1000])
;>>>ata = sparse.lil_matrix(shape)
;>>>ata
;<1x2 sparse matrix of type '<type 'numpy.int16'>'
;	with 2 stored elements in LInked List format>
;
;!!!wrong dimensions!!!
;
;because Python.tuple([1000,1000]) is over send as list and these are not the same in python
;and will call the constructor differently
;
;work around:
;
;Python.n = 1000
;>>>ata = sparse.lil_matrix((n,n))
;ata = Python.ata
;
; running without PYTHONHOME set
end
