; expect M+3 columns of length N: x,y, id, [v1, v2, v3... vM]
; remap trailing M data columns to regular grid created from irregular x,y positions
; output gridsize always = sqrt(N)xsqrt(N) to ensure reasonable sampling
; smooth remapped M columns by nsmooth bins as needed
;
; id column is never used, but useful to carry around for debugging
;
; the 2nd smoothing thing this is a bit clunky 
; @@ need a better way to "enlarge" the psfmap to the area of interest in case corners are missing etc
pro remap_xy, fdata, gx, gy, nsmooth=nsmooth, outname=outname
  data = (read_ascii(fdata,comment='#')).(0)
  x = data[0,*]
  y = data[1,*]
  val = data[3:*,*]
  
  if not keyword_set(outname) then outname = fdata+'.map'
  
  ; define regular xy grid that has on average 1 kernel/cell
  sz = size(val)
  nb = round(sqrt(sz[2]))
  if not keyword_set(nsmooth) then nsmooth=2 else nsmooth = (nsmooth > 1) < nb
  xstep = (max(x)-min(x))/nb
  ystep = (max(y)-min(y))/nb
  xout = findgen(nb+1)*xstep+min(x)   ; enlarge regular grid a bit so we are sure 
  yout = findgen(nb+1)*ystep+min(y) 

  rval = fltarr(sz[1],nb+1,nb+1)
  rsval = fltarr(sz[1],nb+1,nb+1)
  rssval = fltarr(sz[1],nb+1,nb+1)

  ; triangulate and interpolate values to regular grid 
  triangulate, x, y, t
  for i=0,sz[1]-1 do rval[i,*,*] = trigrid(x, y, val[i,*], t, xout=xout, yout=yout,missing=!values.f_nan)
  xmat =  xout # replicate(1,nb+1)
  ymat = replicate(1,nb+1) # yout

  ; smooth by nsmooth bins in x and y
  ; @ this is hacky
  for i=0,sz[1]-1 do begin 
    rsval[i,*,*] = smooth(rval[i,*,*],nsmooth,/nan,/edge_mirror) 
  ; smooth again but on 2 x smooth size to catch psfs on the edge
    rssval[i,*,*] = smooth(rval[i,*,*],(1.5*nsmooth) < nb,/nan,/edge_mirror) 
  end

  ; @@@ HOW TO DO THIS BETTER?
  ; substituted for NaNs by double smoothed, to ensure that points on the edge are always surrounded by valid values
  rsmap = total(rsval,1)
  ibad = where(finite(rsmap) eq 0, nbad)
  if nbad gt 0 then begin
     ij = array_indices(rsmap,ibad)
     for i=0,nbad-1 do rsval[*,ij[0,i],ij[1,i]] = rssval[*,ij[0,i],ij[1,i]] ; set bad entries to nearest neigbours
  end

;  tvrscl, reform(rsval[0,*,*]),os=6,pos=3
;  stop
  ; set all remaining NANs to 0: in all likelyhood these are
  ; so far out of field that smoothing didnt help: so it is safe to ignore as they will never be used
  ibad = where(finite(rsval) eq 0,nbad)
  if nbad gt 0 then begin 
     print,'setting  ',nbad,' NANs to zero'
     rsval[ibad] = 0.  
  end
  ; reform arrays in columns and write: x y c1 c2 c3 etc
  j = 1#indgen(n_elements(xmat))
  map = [xmat[j],ymat[j]]  
  for i=0,sz[1]-1 do map = [map,(reform(rsval[i,*,*]))[j]]
  openw, lun, outname, /get_lun
  printf, lun, map, format='(2f12.3,'+strcompress(sz[1],/rem)+'g14.6)'
  close, lun & free_lun, lun

end
