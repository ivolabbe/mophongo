; create rms map by: 
; rebin on scales of (tile size) 
; make global noise equalized map by 
; 1) multiply residual map by sqrt(exp)
; 2) mask 2.5 sigma outliers
; 3) construct running rms map in moving 5x5 window (~tile size)
; 4) smooth by 2*tile size 
; 5) divide by sqrt(exp)
;

function getrmsmap, res2, exp2, tsz, meanmap, medmap, lsz=lsz
  len = n_elements(x)
  lsz = lsz[0]
  
  ; assume tsz is ~5*psf fwhm or typical source size
  ; determine noise on scales of PSF, then rescale to rms/pixel
  if not keyword_set(lsz) then lsz = round(tsz/6.) else lsz = round(lsz) ; linear scale size on which to to calc rms
  div = round(1.*tsz/lsz)   
  print, 'getting rms from residual map on scales of ',lsz,' pixels'  
  sz = size(res2)  
  nx = sz[1]/lsz  
  ny = sz[2]/lsz

  if n_elements(exp2) eq 0 then exp2 = fltarr(sz[1:2])+1. 
  respad = extrac(res2, 0, 0, sz[1],sz[2])
  iblot = where(respad eq 0,nblot)
  if nblot gt 0 then respad[iblot] = !values.f_nan
  bres = rebin(respad[0:nx*lsz-1,0:ny*lsz-1],nx,ny)*lsz*lsz ; conserve flux
  exp = rebin((extrac(exp2,0,0,sz[1],sz[2]))[0:nx*lsz-1,0:ny*lsz-1],nx,ny) 

  ; noise equalize assuming sqrt(exp)
  bresn = bres*sqrt(exp)
  bres[*,0] = !values.f_nan &  bres[*,ny-1] = !values.f_nan
  bres[0,*] = !values.f_nan &  bres[nx-1,*] = !values.f_nan

  r = mad(bresn,res=res)
  bresn[where(growmask(res gt 2.5*r, /inverse))] = !values.f_nan
  r = mad(bresn,res=res)
  bresn[where(growmask(res gt 2.5*r, /inverse))] = !values.f_nan

  ; make local sigma map in moving window of div x div pixels (so ~ tile size)
  div2 = div*div
  mn=smooth(bresn,div,/nan,/edge_mirror)  ; running local mean
  imdev = (bresn - mn)^2   ; map of pixel variance relative to local mean
  imvar = smooth(imdev,div,/nan,/edge_mirror)*div2/(div2-1) ; map of average pixel variance relative to local mean
  srms = smooth(sqrt(imvar), 2*div, /nan, /edge_mirror) ; smooth to ~ 2 x tile size
  rms = srms/sqrt(exp)/lsz    ; rescale with sqrt(t), scale rms back to per pixel instead of per lsz
  rms[where(exp eq 0)] = !values.f_nan
  
  ; simply rebin to original scale
  rmsmap = fltarr(sz[1:2])
  rmsmap[0:nx*lsz-1,0:ny*lsz-1] = rebin(rms,nx*lsz,ny*lsz)


;  mn=smooth(bresn,10*div,/nan,/edge_mirror)  
;  mean = mn/sqrt(exp)/(lsz*lsz)
;  mean[where(exp eq 0)] = !values.f_nan

;  meanmap = fltarr(sz[1:2])  
;  meanmap[0:nx*lsz-1,0:ny*lsz-1] = rebin(mean,nx*lsz,ny*lsz)
 
  med = median(bres*sqrt(exp), div)
  med = med/sqrt(exp)/(lsz*lsz)
  med[where(exp eq 0)] = !values.f_nan

  medmap = fltarr(sz[1:2])
  medmap[0:nx*lsz-1,0:ny*lsz-1] = rebin(med,nx*lsz,ny*lsz)
  
  return, rmsmap
end


  ; mask bad areas, iterate once
;  m = median(bresn)
;  iok = where(finite(bresn))
;  r = robust_sigma(bresn[iok])
;  mask = abs(bresn-m) gt 3.*r
;  ibad = where(mask, nbad)
;  if nbad gt 0 then bresn[iok[ibad]] = !values.f_nan
;  iok = where(finite(bresn))
;  r = robust_sigma(bresn[iok])
;  ibad = where(abs(bresn[iok]-m) gt 3.*r, nbad)
;  if nbad gt 0 then bresn[iok[ibad]] = !values.f_nan

;  rmsfac is the ratio of the rms on scales of size lsz (scaled back to per-pixel)
; to the rms directly measured from the residual map (i.e. the excess noise)
; this may not work for images with very large differences in depth

  ; @@ not sure whether this is very accurate. it should work, but implementation might
  ; not be so robust.
  ; factor is the difference between the rms on scales of size lsz (scaled back to per-pixel)
  ; to the rms directly measured from the residual map
  ; the difference can be used to 'correct' the chi2 

;  res2n = res2*sqrt(exp2)
;  rraw = mad(res2n[where(finite(res2n) and exp gt 0)],median=m,res=res)
;  res2n[where(growmask(abs(res2n-m) gt 2.5*rraw, /inverse))] = !values.f_nan
;  rraw = mad(res2n[where(finite(res2n) and exp gt 0)],median=m,res=res)
;  res2n[where(growmask(abs(res2n-m) gt 2.5*rraw, /inverse))] = !values.f_nan  
;  rmsfac = r/lsz/rraw
; @@@ rmsfac doesnt seem to work on images with large range in depth 
;  print,'rmsfac:', rmsfac

;tvs,rms,mm=[0.15,1.]
;tvs,rms*sqrt(exp/max(exp)),mm=[0.17,0.2]
;tvs,bresn,mm=[-1,1]*50 
;plothist, rms, xrange=[0.1,0.5], bin=0.005,/nan
;print, 'regridding'
