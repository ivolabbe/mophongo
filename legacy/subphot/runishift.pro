  

  param = "example/mophongo.param"
  readcol, param, key,f='a,a',COMMENT='#',DELIM='#',/SILENT
  for i=0,n_elements(key)-1 do r=execute(key[i])



  ishift, imdir+file_basename(det), imdir+file_basename(det,'.fits')+'_seg.fits', imdir+file_basename(phot), $
        outdir+file_basename(det,'.fits')+'.xy', $
        tsz, sm_shift=4,sm_kern=5, outname=imdir+"out", order=[5,4,2,1], beta=beta, $
        flim=0.7, frlim=0.75, adlim=0.05, klim=0.35, display=1, xysig=2, max_shift=2.
