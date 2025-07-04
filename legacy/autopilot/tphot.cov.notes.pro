
; also seee mpfit for error propagation
; https://hesperia.gsfc.nasa.gov/ssw/gen/idl/fitting/mpfit/mpproperr.pro


	    /* dof->data now contains all the best-fit fluxes
	       while A has been destroyed.
	       To find the inverse of the original A follow NumRec:
	       the matrix is already decomposed so now just
	       find inverse by columns by back sostitution.
	       calculate the inverse of A to get Fisher matrix, by back-substitution
	       for (j=1;j<=dof->size;j++)
	       cout << dof->data[j] << " " << endl; //exit(0);
	       use the columns of identity matrix for back-substitution*/
	    for (j=1; j<=dof->size; j++) {
	      for (k=1; k<=dof->size; k++) col[k] = 0.0;
	      col[j] = 1.0;
	      lubksb(A, dof->size, indx, col);
	      for (k=1; k<=dof->size; k++) covar[k][j] = col[k];
	    }

	    break;

	    // ------------------------------------------------------------------------------ //
	    // THIS BLOCK FOR CHOLESKY DECOMPOSITION
	    // ------------------------------------------------------------------------------ //

	  case(2):
	    p = dvector(1,dof->size);
	    for (k=1; k<=dof->size; k++) p[k] = 0.0;  // initialize p

	    choldc(A, dof->size, p);  // Cholesky decomp.
	    cholsl(A, dof->size, p, B, dof->data); // Cholesky back-substitution
	    /* dof->data now contains all the best-fit fluxes (and background if fit_to_bg==1)
	       now calculate the inverse of A to get Fisher matrix, by back-substitution
	       cout << "done Cholesky decomposition." << endl;
	       calculate the inverse of L first, stored in the lower triangle of A*/
	    for (k=1;k<=dof->size;k++) {
	      A[k][k] = 1.0/p[k];
	      for (j=i+1;j<=dof->size;j++) {
		sum=0.0;
		for (jj=i;jj<j;jj++) sum -= A[j][jj]*A[jj][k];
		A[j][k]=sum/p[j];
	      }
	      for (j=1;j<i;j++) {
		A[j][k] = 0.0;
	      }
	    }
	    // then use L**-1 to calculate A**-1 and store in covar
	    // A**-1 = (L**-1)^T * L**-1
	    for (k=1;k<=dof->size;k++) {
	      for (j=1;j<=dof->size;j++) {
		sum = 0.0;
		for (jj=1;jj<=dof->size;jj++) {
		  sum += A[jj][k] * A[jj][k];
		}
		covar[k][j] = sum;
	      }
	    }
	    break;

	  case (3):
	    // ------------------------------------------------------------------------------ //
	    // THIS BLOCK FOR IBG - ITERATIVE BICONJUGATE GRADIENT METHOD DECOMPOSITION
	    // ------------------------------------------------------------------------------ //
	    nmax = dof->size + 10;
	    for (k=1;k<=dof->size;k++) {
	      nmax=nmax+2*contaminant[k];
	      //cout << "..." << contaminant[i]<< " " ;
	    }
	    nmax = MIN (nmax, 16777216); //, 2 * dof->size * dof -> size);
	    //cout << ">>> " << nmax <<endl;

	    /* Allocate memory for storing sparse matrix */
	    sa = dvector(1, nmax);
	    /* Allocate memory for storing row indexed array */
	    ija = ulvector(1, nmax);
	    sprsin(A, dof->size, thresh, (unsigned long)nmax, sa, ija);
	    /// converts matrix a into vector sa indexed by vector ija (NumRec)
	    /// NB: still have to solve the system.

	    n = (unsigned long)dof->size;
	    linbcg(sa, ija, n, B, dof->data, itol, tol, itmax, &iter, &erro); /// SOLVE IT.
	    /// itol = 4. tol = 1.e-128. x not initialized?!
	    //printf("\n");
	    free_dvector(sa, 1, nmax);
	    free_ulvector(ija, 1, nmax);
	    //cout << "done IBG decomposition." << endl;

	    // Find covariance matrix:
	    indx = ivector(1,dof->size);
	    for (k=1;k<=dof->size;k++) indx[k] = 1;  // initialize indx
	    col = dvector(1,dof->size);  //  columns of identity matrix

	    ludcmp(A, dof->size, indx, &d);
	    for (j=1; j<=dof->size; j++) {
	      for (k=1; k<=dof->size; k++) col[k] = 0.0;
	      col[j] = 1.0;
	      lubksb(A, dof->size, indx, col);
	      for (k=1; k<=dof->size; k++) covar[k][j] = col[k];}
	    break;

	  }

