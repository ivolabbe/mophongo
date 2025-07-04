//------------------------------------------------------------------------------------------------//

void apertures(int naper, double *f, double *errf, 
	      float Rmax, int binsubpix, float *aper,
	      double *pixels_aper, double *pixels_aper_rms, 
	      int W, int H, float x0, float y0)
{
  double facareasub, dsubpix, R, R2, fbkgd;
  double *facarea;
  float x, y, xp, yp, xm, ym, 
    xa, ya, xb, yb, dx, dy, dsx, dsy;
  int ap, xc, yc, xmin, xmax, ymin, ymax, 
    i, j, iii, jjj, keep_pix;

  dsubpix=1.0/(float)binsubpix;
  facareasub=dsubpix*dsubpix;

  /*for (j=0;j<H;j++){
    for (i=0;i<W;i++){
      cout << pixels_aper[W*j+i]<< " ";
    }
    cout << endl;
    }*/

  x0=(x0-floor(x0))+0.5*(W-1); // Position in thumbnail
  y0=(y0-floor(y0))+0.5*(H-1);

  xmin=0;
  xmax=W-1;
  ymin=0;
  ymax=H-1;

  facarea = (double *) malloc(naper * sizeof(double));

  for(ap=naper-1;ap>-1;ap--) // Loop on apertures, starting from larger
    {
      facarea[ap]=0.0;
      f[ap]=0.0 ; 
      errf[ap]=0.0; 
    }

  xc=floor(x0-0.5e0);
  yc=floor(y0-0.5e0);
  
  for (j=ymin;j<=ymax;j++)
    for(i=xmin;i<=xmax;i++) // Loop on pixels of this object
      {
	x=(float)i;
	y=(float)j;
	xp=x+0.5;
	yp=y+0.5;
	xm=x-0.5; // lower x bound of i pixel
	ym=y-0.5; // lower y bound of i pixel
	
	// "a"=farthest point; "b"=closest point
	if (ym>(float)yc+0.5) // Upper region
	  {
	    if (xm>(float)xc+0.5){xa=xp; ya=yp; xb=xm; yb=ym;} // 1st quad
	    else if (xp<(float)xc-0.5){xa=xm; ya=yp; xb=xp; yb=ym;} // 2nd quad
	    else { // Same column, above
	      if (x0>=xc) {xa=xm; ya=yp; xb=x0; yb=ym;}
	      else {xa=xp; ya=yp; xb=x0; yb=ym;}
	    }
	  }
	else if (ym<(float)yc-0.5) // Lower region
	  {
	    if (xm<(float)xc-0.5){xa=xm; ya=ym; xb=xp; yb=yp;} // 3rd quad
	    else if (xp>(float)xc+0.5){xa=xp; ya=ym; xb=xm; yb=yp;} // 4th quad
	    else{ // Same column, below
	      if (x0>=xc) {xa=xm; ya=ym; xb=x0; yb=yp;}
	      else {xa=xp; ya=ym; xb=x0; yb=yp;}
	    }
	  }
	else // Same row
	  {
	    if (i>xc){ // Right
	      if (y0>=yc) {xa=xp; ya=ym; xb=xm; yb=y0;}
	      else {xa=xp; ya=yp; xb=xm; yb=y0;}
	    }
	    else if (i<xc){ // Left
	      if (y0>=yc) {xa=xm; ya=ym; xb=xp; yb=y0;}
	      else{xa=xm; ya=yp; xb=xp; yb=y0;}
	    }
	  }
	
	keep_pix=1;
	
	for(ap=naper-1;ap>-1;ap--) // Loop on apertures, starting from larger
	  {	
	    
	    if (keep_pix) // If the pixel has been excluded before, no need to check it again
	      {

		R=0.5*aper[ap]; // First loop has R=Rmax
		R2=R*R;
		facarea[ap]=1.0;
		fbkgd=0.0;
		
		// Check if the pixel is totally outside R
		if ((xb-x0)*(xb-x0)+(yb-y0)*(yb-y0)>R2){
		  facarea[ap]=0.0;
		  // Also, don't need to check this pixel anymore
		  keep_pix=0;
		}
		// Now check if it's totally inside R
		else if ((xa-x0)*(xa-x0)+(ya-y0)*(ya-y0)>=R2){
		  // Must do subgrid
		  facarea[ap]=0.0;
		  dx=xm-x0;
		  dy=ym-y0;
		  for (jjj=0;jjj<binsubpix;jjj++)
		    for (iii=0;iii<binsubpix;iii++)
		      {
			dsx=dx+((float)iii+0.5)*dsubpix;
			dsy=dy+((float)jjj+0.5)*dsubpix;
			if (dsx*dsx+dsy*dsy<R2) facarea[ap]+=facareasub;
		      }
		  //printf("---> %d %d (%d) %f\n",i,j,ap,facareasub);
		}
	      }
	    
	    f[ap]+=facarea[ap]*pixels_aper[j*W+i];
	    errf[ap]+=(facarea[ap]*pixels_aper_rms[j*W+i])*(facarea[ap]*pixels_aper_rms[j*W+i]);

	    // Subtract background?
	    f[ap]-=facarea[ap]*fbkgd;
	    
	  } // end loop on aper
	
      } // End loop on pixels; f[ap] is the flux within aperture aper[ap]
  
  for (ap=naper-1;ap>-1;ap--) errf[ap]=sqrt(errf[ap]); //+MAX(0.0,f[ap])/GAIN);    
  free(facarea);   
}
