; TODO
; - do least sq error on fitted fluxes -> so we can make 'confusion error map'
; - test even kernels
; - @@@ CHECK AGAIN subpixel alignment of aperture phot

 ; mktmpl @@@@@@@@@ this would be a good place to calc aperture correction as a function
 ; of radius


forward_function interpol_map, getrmsmap, mkmodel, ruopcg, ruobuild, spruo

pro _iphot, img1, seg1, img2, exp2, x, y, id, tsz, gx, gy, buf=buf, basis=basis, $
            kernelmap=kernelmap, shiftmap=shiftmap, raper=raper, outname=outname, $
            rms_sz=rms_sz, fwhm1=fwhm1, soname=soname
COMPILE_OPT DEFINT32
common mpfit_com, tile, objt,fpost,llt,sz,xi,yj,ii,jj,indx,indy

   if not keyword_set(soname) then soname = expand_path('$IFL_DIR')+'/lib/libifl.'+lib_so_ext()
   if not keyword_set(dthr) then dthr = 5d-4
   if not keyword_set(buf) then buf = 30L      ; initialize sparse matrix vectors length to buf x n
   itmax = 100
   sz = size(img2)

   ;initially fine shift is zero
   shxi = 0.
   shyi = 0.


   ; create vector of obj templates, which are conv(det*seg, kern)
   mktmpl, img1, seg1, x, y, id, tsz, obj, ll, rxy, img2, gx, gy, basis=basis, $
           kernelmap=kernelmap, shiftmap=shiftmap, fwhm1=fwhm1
   print, "#### obj ####", size(obj)

   ; set threshold of zero elements to fraction of (diagonal elements), set to dthr of faintest 5%
   ; @@ check if this works ok
   thr = percentiles((total(total(obj*obj,1),1)),value=[0.05])*dthr

 	warning_matrix = where((total(total(obj*obj,1),1)) lt thr)
	 if warning_matrix(0) ne -1 then begin
	 		print,'WARNING:  There may be objects with negative flux screwing up the photometry!'
			print,n_elements(warning_matrix),' objects may have problems'
			print,warning_matrix
	endif


   ; make hessian matrix Aij = tmpli*tmplj and observed vector rj = tmplj*obsj
   mkhess, x, y, obj, ll, img2, a, r, soname=soname, thr=thr, buf = buf


   ; set "zero" diagonal elements to minimum threshold, to satisfy biconjugate gradient
   (*a.xd)[0:a.r] = (*a.xd)[0:a.r] > thr   ; @@@ why not > 0 ?
   spinfo,a

   ; now solve with biconjugate gradient
   print,'solving for ', a.r, ' objects'
   save, a, r, x, y, ll, obj, file=outname+'_obj1.sav'
;   tvs, obj[*,*,27800:*]

   time1=systime(1)
   f = ruopcg(a,double(r),dblarr(a.r),nit=nit, itmax=itmax,tol=1d-20,soname=soname)
   print, systime(1) - time1 ,' SECONDS, '


   ; @@@ hmm. need to address this. because the negative fits will affect the positive fits
   ; can we set a boundary condition somehow?
   ftmpl = total(total(obj,1),1)  ; flux of template objects (after convolution, but should be the same as in original map)
   fpos = (f > 0.)     ;  fpos is the 'color' from ftmpl to the best fit flux of img2
   fimg = fpos*ftmpl   ; f*ftmpl is the fitted flux in the img2


   print, 'writing output'
   ; create model map and write model and residual
   model = mkmodel(obj, fpos,shxi,shyi, sz[1], sz[2], ll)
   fits_write, outname+'_model.fits', model
   res = img2 - temporary(model)
   fits_write, outname+'_res.fits', res

   ; tvs, img2[10000:14000,10000:14000],os=0.2
   ; tvs, img2[10000:14000,10000:14000]-model[10000:14000,10000:14000],os=0.2
   ;

   ; get rms on scales of rms_sz averaged in moving window of size tsz and scaled back to rms per pixel
   ; note RMS variations between overlapping tiles are probably very small factor in quality of fit

   rms=getrmsmap(res, exp2, tsz, meanmap, medmap, lsz=rms_sz)
   pixerr = rms[round(x),round(y)]
   fits_write, outname+'_rms.fits', rms
   fits_write, outname+'_bgfit.fits', medmap

;### begin second shiftmap with mpfit


;definition of error
 err = raper[0]*sqrt(!pi)*median(rms)
;first pass tile total fluxes
 flux = total(total(obj,1),1)*fpos

;define 3x3 tsz, overlapping  grid for second shiftmap
 ngridx = ceil((size(img2))[1]/(1.5*tsz))
 ngridy = ceil((size(img2))[2]/(1.5*tsz))
 x = (lindgen(ngridx+2))*(1.5*tsz)
 y = (lindgen(ngridy+2))*(1.5*tsz)

 n = 0

 shx = []
 shy = []
 indx = lindgen(ngridx+1)
 indy = lindgen(ngridy+1)

 ; for: iterate through 3x3 tiles do begin
 ; if: enough bright sources then  mpfit(shx, shy, flux)
 for i=0, n_elements(indx)-3 do begin
    for j=0, n_elements(indy)-3 do begin

       ; if more than 3 flux/err >20.0 sources in tile
       if n_elements(where( flux/err gt 20.0 and $
                            ll[1,*]+rxy[1,*] gt y[indy[j]] and $
                            ll[1,*]+rxy[1,*] lt y[indy[j]+2] and $
                            ll[0,*]+rxy[0,*] gt x[indx[i]] and $
                            ll[0,*]+rxy[0,*] lt x[indx[i]+2] )) gt 3 then begin

          ;extract tile from img2 for flux/shift fit
          tile = img2(x[indx[i]]:x[indx[i]+2],y[indy[j]]:y[indy[j]+2])

          ;indices of elements within tile
          inerd= [where(ll[1,*]+rxy[1,*] gt y[indy[j]] and $
                        ll[1,*]+rxy[1,*] lt y[indy[j]+2] and $
                        ll[0,*]+rxy[0,*] gt x[indx[i]] and $
                        ll[0,*]+rxy[0,*] lt x[indx[i]+2])]
          ;objects, flux clolour, lower lefts within tile
          objt =  obj[*,*,inerd]
          fpost= fpos[inerd]
          llt =    [ll[0,inerd],ll[1,inerd]]

          ;initial shift parameter
          shxi = 0.0
          shyi = 0.0
          maxshift = 5.0

;          p = objt,fpost,(size(tile))[1],(size(tile))[2], [llt[0,*]+shxi,llt[1,*]+shyi]
;          p = objt,fpost,shxi, shyi, (size(tile))[1],(size(tile))[2], llt]
;          purple are named keywords referenced below

          parinfo = replicate({value:0.D, fixed:0, step:0.D, mpminstep:0.D, $
                               limited:[0,0], limits:[0.D,0]}, n_elements(fpost)+2)
          parinfo[*].value = [shxi,shyi,fpost] ; initialize with shift 0, and the flux values from second fit
          parinfo[0:1].step = 1                ; make sure step size is large enough to measure LM differential
          parinfo[2:-1].fixed = (total(total(objt,1),1)*fpost)/err lt 20
          parinfo[0:1].limited = [1,1]
          parinfo[0:1].limits = [-maxshift,maxshift]

          ;redefine loop indexes to common blocks -> eval_fcu
          ii = i
          jj = j
          xi = x
          yj = y


          print, "mpfit tile:", n
          p = mpfit('eval_fcn', ftol=1e-5, parinfo=parinfo, STATUS=status, nfev=nfev,$
                    BESTNORM=chi2, covar=covar, perror=perror, niter=niter, nfree=nfree,$
                    npegged=npegged, dof=dof, ERRMSG=errmsg,functargs=fargs)
          ;save derived shift for later use ## currently not used!
          shx = [shx,p[0]]
          shy = [shy,p[1]]

          n+=1
       end &$
    end &$
  end
   ;## need to speed up this up before moving forward
                                ;## maybe move second shift calc to
                                ;separate procedure and add keyword
                                ;option -> mophongo.param


   print, "# obj", size(obj)


   ;###############################################################

   print, "Start second pass"

   img2 = img2 - medmap;meanmap

   fits_write, outname+'_bgsub_phot.fits', img2

   mkhess, x, y, obj, ll, img2, a, r, soname=soname, thr=thr, buf = buf

   (*a.xd)[0:a.r] = (*a.xd)[0:a.r] > thr
   spinfo,a
   print,'solving for ', a.r, ' objects'
   save, a, r, x, y, ll, obj, file=outname+'_bgsub_obj.sav'
   time1=systime(1)
   f = ruopcg(a,double(r),dblarr(a.r),nit=nit, itmax=itmax,tol=1d-20,soname=soname)
   print, systime(1) - time1 ,' SECONDS, '
   ftmpl = total(total(obj,1),1)
   fpos = (f > 0.)
   fimg = fpos*ftmpl
   print, 'writing output'
   model = mkmodel(obj, fpos, shx,shy, sz[1], sz[2], ll)
   fits_write, outname+'_bgsub_model.fits', model
   res = img2 - temporary(model)
   fits_write, outname+'_bgsub_res.fits', res
   ; fits_write, outname+'_bgsub_rms.fits', rms

   print, "### obj", size(obj)
   print, "End second pass"

   ;###############################################################







;   print, 'fpos', size(fpos)
;   print, 'rxy', size(rxy)
;   print, 'forg', size(forg)
;   print, 'flux1', size(flux1)
;   print, 'flux2', size(flux2)
;   print, 'apcor', size(apcor)
;   print, 'err2', size(err2)
;   print, 'chi', size(fres)
;   print, 'absdev', size(absdev)

   save, fpos, rxy, file=outname+'_bgsub_fposrxy.sav'

; note:
; - 3x oversample means err < 0.5% random err for S/N>30 sources in 2 pix radius apertures
; - err, fres, chi, absdev, are not oversampled. for 2,3 pix they are uncertain to 25%,10%
; @@@@ YOU clearly want X5 and also the rms oversampled, but now takes too long
; need faster (e.g. native) implementation
   if n_elements(raper) gt 0 then $
     aperphot, raper, obj, img1, img2, res, seg1, rms, id, fpos, ll, rxy, forg, flux1, flux2, apcor, err2, fres, chi, absdev, $
               rmsfac=1., oversample=3., shiftmap=shiftmap

  ; note: the blotting appears not quite correctly aligned, because the segmentation
  ; map needs to be shifted as well; but it is done correctly in aperphot
   res = float(res*(round(temporary(seg1)) eq 0)) ; dont need seg + res any more
   fits_write, outname+'_blot.fits', res

; write catalog
  print, 'writing catalog'
  naper=n_elements(raper)
  str=strtrim(lindgen(naper)+1,2)+' '
  openw,lun, outname+'_phot.cat',/get_lun
  printf, lun, '# x y id ftmpl ffit ', 'ftmpl'+str, 'forg'+str, 'f'+str, 'e'+str, 'fcor'+str, 'ecor'+str, $
               'apcor'+str, 'chi'+str, 'res'+str, 'adev'+str, format='('+strtrim(10*naper+1)+'a)'
  printf, lun, '# x,y,id = coordinates in template image of source id'
  printf, lun, '# ftmpl, ffit = total flux of the template, and the best fit (within the tile)'
  printf, lun, '# forgX, fX, eX = original flux, flux + error after "cleaning" = subtraction of neigbours (in aperture X)'
  printf, lun, '# fcorX, ecorX = cleaned flux + error after aperture correction)'
  printf, lun, '# apcorX = aperture correction to apply to fX to match aperture flux in template image)'
  printf, lun, '# chiX,resX,adevX = (integrated)  chi2, relative residual, and relative absolute deviation'
  for i=0L,a.r-1 do printf, lun, x[i], y[i], id[i],  ftmpl[i], fimg[i], flux1[*,i], forg[*,i], flux2[*,i], err2[*,i], $
     flux2[*,i]*apcor[*,i],  apcor[*,i]*err2[*,i], apcor[*,i], chi[*,i], fres[*,i],  absdev[*,i], f='(2f10.2,i10,2g12.5,'+strtrim(string(10*naper),2)+'g12.5)'
  close,lun & free_lun,lun

;print, x[i], y[i], id[i], fimg[i], ftmpl[i], forg[*,i], flux[*,i], err[*,i], fres[*,i], chi[*,i], absdev[*,i], f='(2f9.2,i10,2g11.5,'+strtrim(string(6*naper),2)+'g11.5)'

end


pro aperphot, raper, obj, img1, img2, res, seg, rms, id, f, ll, rxy, forg, flux1, flux2, apcor, ferr, fres, fchi2, fabsdev, rmsfac=rmsfac, $
              oversample=oversample, shiftmap=sm
COMPILE_OPT DEFINT32
  if keyword_set(oversample) then os = round(oversample) else os=1  ; oversampling factor for more accurate small apertures
  if not keyword_set(rmsfac) then rmsfac=1.  ; not sure this works for superlarge image
  naper = n_elements(raper)

  print, 'performing aperture photometry in apertures of radii ',raper,' pixels',f='(a,'+strcompress(naper,/rem)+'f6.1,a)'
  time1 = systime(1)

  len = (size(obj))[3]
  tsz = (size(obj))[1]
  naper = n_elements(raper)
  flux1 = fltarr(naper,len)
  flux2 = fltarr(naper,len)
  apcor = fltarr(naper,len)
  forg = fltarr(naper,len)
  ferr = fltarr(naper,len)
  fres = fltarr(naper,len)
  fchi2 = fltarr(naper,len)
  fabsdev = fltarr(naper,len)

  ; note the coordinates come from img1. so find shift at x,y coordinates (in shiftmap) and
  ; add the shift to rxy to find the coordinates in img2.
  ; @@@ CHECK AGAIN THAT THE SHIFT HAS THE CORRECT SIGN
  if keyword_set(sm) then begin
   xy = ll + rxy  ;###1 was here
    shxy =  reform(interpol_map(sm[0,*], sm[1,*], sm[2:3,*], xy[0,*], xy[1,*]))
  end else shxy = fltarr((size(rxy))[1],(size(rxy))[2])
  xy = ll + rxy ; here so tile center reference is correct if not keword_set(sm)
  cxy = rxy + shxy

 ; do aperture measurements on minitiles of size ~ aper
 ; rxy is always < 1 pixel (fractional pixel position)
 ; shxy can be several pixels: so make minitile larger
  maxaper = ceil(max(raper) + max(abs(shxy)))
  tm1 = tsz-1
  td2 = tsz/2
  t1 = td2 - maxaper
  s1 = 4*maxaper
  mrxy = rxy - t1               ; center of sources in minitile in img1
  mcxy = cxy - t1               ; center of sources in minitile in img2

  llx = ll[0,*]+t1
  lly = ll[1,*]+t1

  for i=0L,len-1 do begin
  ; make a tile of maximum size raper*2
     tobj = obj[t1:t1+s1-1,t1:t1+s1-1,i] ; template * best fit flux scaling
     timg1 = extrac(img1, llx[i], lly[i], s1, s1)
     timg2 = extrac(img2, llx[i], lly[i], s1, s1)
     tres = extrac(res, llx[i], lly[i], s1, s1)
     tseg = extrac(seg, llx[i], lly[i], s1, s1)
     trms = extrac(rms, llx[i], lly[i], s1, s1)
     mseg1 = tseg eq 0 or tseg eq id[i]     ; mask seg pixels of nearest neigbours
     mseg2 = shift(mseg1, round(shxy[*,i])) ; shift to img2 coords
     tphot1 = timg1*mseg1                   ; (tseg eq id[i])  ; blot neighbouring sources
     tphot2 = tres*mseg2 + tobj*f[i]        ; actual photometry image: fit + residual map with blotted neighbours

     ; @@@ the chi2 should actually include the error from neighbours fitted error map!
     ; otherwise a high chi-2 is artifical, because it is actually included in rms map
     ; calculate error and chi2 without oversampling
     ; means pi*r^2 + randomn()*sqrt(2*pi*r) pixels used in measuring error
     ; calculate all errors withouth oversampling, so make error sqrt(circum)/area:
     ; - means 25%,10% error in rms estimate for 2,3 pix radius
     trms2 = trms^2.
     chi2 = (tres*rmsfac)^2/trms2

    ; this is used to form apertures
     mkcoo, s1, cntr = mrxy[*,i], d=d1   ; aperture centered on img1
     mkcoo, s1, cntr = mcxy[*,i], d=d2   ; aperture centered on img2

     ; oversample to improve accuracy for small apertures, conserve flux
     tobjos = rebin(tobj, s1*os, s1*os)/os^2.
     tresos = rebin(tres, s1*os, s1*os)/os^2.
     tphot1os = rebin(tphot1, s1*os, s1*os)/os^2.
     tphot2os = rebin(tphot2, s1*os, s1*os)/os^2.
     timg2os = rebin(timg2, s1*os, s1*os)/os^2.
     d1os = rebin(d1, s1*os, s1*os)
     d2os = rebin(d2, s1*os, s1*os)

;     otobj = obj[*,*,i]*f[i]  ; template * best fit flux saling
;     otimg1 = img1[ll[0,i]:ll[0,i]+tm1,ll[1,i]:ll[1,i]+tm1]
;     otimg2 = img2[ll[0,i]:ll[0,i]+tm1,ll[1,i]:ll[1,i]+tm1]
 ;    otres = res[ll[0,i]:ll[0,i]+tm1,ll[1,i]:ll[1,i]+tm1]  ; residual map
 ;    otrms = rms[ll[0,i]:ll[0,i]+tm1,ll[1,i]:ll[1,i]+tm1]
 ;    otseg = seg[ll[0,i]:ll[0,i]+tm1,ll[1,i]:ll[1,i]+tm1]
 ;    omseg = shift(otseg eq 0 or otseg eq id[i],round(shxy[*,i]))       ; mask seg pixels of nearest neigbours
 ;    otres = otres * omseg   ; set these pixels to zero in resmape

     ; center the pixel distance map on current object
     ; @@@ CHECK AGAIN subpixel alignment of aperture phot
     ; Note: the measurement center has been properly shifted (b/c of shifts between img1 and img2)
  ;   mkcoo, tsz, cntr = rxy[*,i], d=od1   ; aperture centered on img1
  ;   mkcoo, tsz, cntr = cxy[*,i], d=od2   ; aperture centered on img2

; check; ok!
; cleaned blotted image looks good. image minitile looks good.
; centroids look ok, but should be able to do better...
     if total(tobj) gt 0.5 and 0 then begin
        mm1 = d1 le 15
        mm2 = d2 le 15
;    timg1[round(mrxy[0,i]),round(mrxy[1,i])] = -100
;    timg2[round(mcos*tobjosxy[0,i]),round(mcxy[1,i])] = -100
;    tres[round(mcxy[0,i]),round(mcxy[1,i])] = -100
        tvrscl, timg1*mm1, pos=0, os=3, mm=[-1,1]*1e0
        tvrscl, timg2*mm2, pos=1, os=3,mm=[-2,2]*1e-3
        tvrscl, tobj*mm2, pos=2, os=3,mm=[-2,2]*1e-3
        tvrscl, tres*mm2, pos=3, os=3,mm=[-2,2]*1e-3
        tvrscl, trms*mm2, pos=4, os=3,mm=[-2,2]*1e-3
        tvrscl, mseg1, pos=5, os=3,mm=[0,1]
        tvrscl, mseg2, pos=6, os=3,mm=[0,1]
        tvrscl, timg1*mm1,pos=7,os=3, mm=[-1,1]*1e0
        tvrscl, (tobj+tres)*mm2,pos=8,os=3,mm=[-2,2]*1e-3
;     otimg1[round(rxy[0,i]),round(rxy[1,i])] = -100
;     otimg2[round(cxy[0,i]),round(cxy[1,i])] = -100
;     tvrscl, otimg1, os=3, pos=2, mm=[-1,1]*1e0
;     tvrscl, otimg2, os=3, pos=3, mm=[-2,2]*1e-3
;  s
     end

     ; correct chisq for oversampling and smoothing:
     ; 1) here  we oversampled by os x os, so the rms noise has gone down by factor os
     ; 2) also multiply tres by the 'smoothing factor', which is the ratio of the
     ; noise on scale lsz (from getrmsmap) to the raw noise measured from the residual map
     ; accounts for the fact that the measurement image could have been smoothed
     ; which suppressed the residuals
     ; @@@@ but now that we scaled up the noise, means that we do not have npix-1
     ; degrees of freedom in chi2 statistic... what to use?


     ; cleaned = original image - neighbours = best-fit template + residual map
     ; note: the neighbours centers were blotted to zero (using the segmap) which
     ; makes it more robust against residuals in the centers of neigbours (e.g. due to
     ; small shifts, color gradients, and PSF mismatches in the centers
     ; do circular aperture measurements on cleaned image
     for j=0,naper-1 do begin
        m1os = (d1os le raper[j])  ; masks defining circular aperture of radius raper[j] in img1
        m2os = (d2os le raper[j])  ; masks defining circular aperture of radius raper[j] in img2
        m1 = (d1 le raper[j])
        m2 = (d2 le raper[j])

        ; @@@ not entirely correct. the tobjos was calculated from segmap + stellar correction
        ; outside segmap, however,
        apcor[j,i] = total(m1os*tphot1os)/total(m2os*tobjos) ; aperture correction to correct to img1
        flux1[j,i] = total(m1os*tphot1os)
        flux2[j,i] = total(m2os*tphot2os)
        forg[j,i] = total(m2os*timg2os)
        ferr[j,i] = sqrt(total(m2*trms2)) ; add variances, forget about the n-1
        fres[j,i] = total(m2*tres)/flux2[j,i]  ; relative residual = residual/flux
        ; @@@ the chi2 should actually include the error from neighbours
        fchi2[j,i] = total(m2*chi2)  ; chi2
        ; ratio of abs(residual)/flux ; not sure how meaningful this is
        fabsdev[j,i] = total(m2*abs(tres))/flux2[j,i]
;        if total(tobj) gt 0.5 then begin
;          print,raper[j],   total(m1*timg1*(tseg eq id[i])), total(m1os*tphot1os),total(m2os*tobjos), apcor[j,i]
; if i gt 1000 then
;        end
     end

;     ndf = !pi*raper^2.-1

;     if chi[0,i]/ndf[0] lt 2 and absdev[0,i]/(flux[0,i] > err[0,i]) lt 0.2 then begin
;        tvrscl, tobj,os=3, mm=[-5,5]*median(trms)
;        tvrscl, tres, os=3, pos=1, mm=[-5,5]*median(trms)
;        tvrscl, (tobj+tres), os=3, pos=2, mm=[-5,5]*median(trms)
;        tvrscl, timg, os=3, pos=3, mm=[-5,5]*median(trms)
;        print, flux[*,i],err[*,i], chi[*,i]/ndf, absdev[*,i]
;
;     end

  end

   help,/mem, out=mem
   m=float((strsplit(mem,/ex))[3])/(1024.)^2.
   print, systime(1) - time1 ,' SECONDS, ', m,' MB MEMORY'

end

; build the kernel for all the grid points, then interpolate over kernel to get
; kernel and shift at locations x,y
pro mktmpl, img1, seg1, x, y, id, tsz, obj, ll, rxy, img2, gx, gy, $
            basis=basis, kernelmap=km, shiftmap=shiftmap, fwhm1=fwhm1
COMPILE_OPT DEFINT32
  if not keyword_set(minsegpix) then minsegpix = 50L
  if keyword_set(shiftmap) then sm = shiftmap


  ngrow = 2
  len = n_elements(x)
  print,'building template vector of ',len,' objects'
  time1=systime(1)

  obj = fltarr(tsz,tsz,len)
  npix = long(tsz*tsz)
  ix = long(x)
  iy = long(y)
  s2 = (tsz-1)/2              ; shift for fft = half tile size

  dxy = [1#(x-ix), 1#(y-iy)]  ; fractional pixel offset
  rxy = s2 + dxy              ; exact coordinates in tile
  ll = long([1#ix,1#iy]-s2)   ; coordinates of lowerleft pixel  (offset of tile in image)

 ;:::klen = (size(km))[2]
 klen = (size(km))[3]         ; number of kernels used
 xg = lindgen(tsz)




;additional shiftmapping is not done because it's assumed that
;the images are registered well enough after shiftmap/reg steps
 if keyword_set(sm) then begin
  ; slen = (size(sm))[2]
  ; sx =  sm[0,*]
  ; sy =  sm[1,*]
   ;@@@ check interpol map
 ;
;   coeff_sh= reform(interpol_map(km[0,*], km[1,*], km[2:*,*], sx, sy))
   sm = fltarr(4,klen)
   sx = gx; + rxy[0,*]; km[0,*]
   sy = gy                      ; + rxy[1,*]; km[1,*]
   slen = klen
 end else begin
   ;coeff_sh= km[2:*,*]
   sm = fltarr(4,klen)
   sx = gx; + rxy[0,*]; km[0,*]
   sy = gy; + rxy[1,*]; km[1,*]
   slen = klen
 end

 ; make kernel in real space, shift by sm[0,i],sm[1,i], then take the fft
 ; direction of shifts is ok!
 ; alternative: use kernel in real space and fft in the loop. take a little speed hit, but
 ; yields another 400 MB or so
 ; find kernel coefficients on the shift map (which is probably higher spatial resolution)

 ; Basically putting getkern from doall.pro in here so
 ; the kernel map from mkkern is used instead.

 if not keyword_set(silent) then print, 'mktmpl: building kernel map'
 ft_kern_map = complexarr(tsz, tsz, slen)
 for i=0L, slen-1 do begin &$
   ;kern = interpolate(mkmodel(basis,coeff_sh[*,i]), xg-sm[2,i], xg-sm[3,i], /grid, missing=0.) &$
    kern = interpolate(km[*,*,i], xg-sm[2,i],xg-sm[3,i], /grid, missing=0.0)
    ft_kern_map[*,*,i] = fft(kern/total(kern)) &$
   ;### km from mkkern is already the kern so it can be used directly
  ;iimage, km[*,*,i]
  ;r = get_kbrd(1)
   ; ft_kern_map[*,*,i] = fft(km[*,*,i]/total(km[*,*,i])) &$
 end
 ;use restore, "/Users/mark/mophongo/example/out_put/psf/kern_CDFS-1_Ks_v0.9.4_sci_CDFS_CH1.sav"
 ;ft_kern_map = complexarr((size(kernel))[1],(size(kernel))[2],(size(kernel))[3])
 ;ft_kern_map = fft(kernel/total(kernel))

 ; prepare interpolation, by expressing x,y in terms of sx,sy bins
 xout = sx[uniq(sx,sort(sx))]
 yout = sy[uniq(sy,sort(sy))]
 nx = n_elements(xout)
 ny = n_elements(yout)
 xstep = (max(sx)-min(sx))/(nx-1.)
 ystep = (max(sy)-min(sy))/(ny-1.)
 gx = reform(sx,nx,ny)
 gy = reform(sy,nx,ny)
; get index of x,y in regular grid
 xs = (x-xout[0])/xstep
 ys = (y-yout[0])/ystep

; this is hacky
; psf = (readfits('psf_k.fits'))[19:83,19:83]
 if keyword_set(fwhm1) then begin
   psf = psf_moffat(NPIXEL=(tsz+0.), FWHM=fwhm1, beta=3.,/norm)
   print, 'adding PSF profile with fwhm',fwhm1,' outside segmap for faintest sources'
 end

 ; @@@@@@@@@ this would be a good place to calc aperture correction as a function
 ; of radius
; showme=1

 xind = lindgen(tsz)
 kmap_reform =  reform(ft_kern_map,tsz*tsz,nx,ny)
  if not keyword_set(silent) then print, 'mktmpl: populating obj vector'
  for i=0L, len-1 do begin
    t1 = extrac(img1,ll[0,i],ll[1,i],tsz,tsz)
    t2 = extrac(img2,ll[0,i],ll[1,i],tsz,tsz)
    s1 = extrac(seg1,ll[0,i],ll[1,i],tsz,tsz)

    ; make template from pixels in segmap only, force positivity
   ;  o =  t1*(s1 eq id[i]) > 0.     ; this leaves negative peaks
   ; try to make template from everything except the other ids
    mask_nn = growmask(s1 eq id[i] or s1 eq 0)  ; neighbours are 0
    mask_obj = s1 eq id[i]
                                ;indexes referring to pixeles in obj
                                ;mask which are nonzero and the
                                ;nsegpix is the number of them.
    isegpix = where(mask_obj, nsegpix)

    ; grow segmap
    for j=0L,ngrow-1 do mask_obj = growmask(mask_obj,/inver)
    mask = mask_obj*mask_nn

    ; add point source profile outside the segmentation map
    ; @@@ need to do this to calculate the aperture correction too!!!
    ; this is for very small masks, smaller than PSF
    if nsegpix lt minsegpix and keyword_set(psf) then begin &$
    ;   tvrscl, t1*mask_obj, mm=[-10,10], pos=0 &$
      psf_i = interpolate(psf, xind-dxy[0,i], xind-dxy[1,i],/grid) &$
      fscale = total(t1*mask)/total(psf*mask) &$
      o = (t1*mask+ (1-mask)*fscale*psf) > 0. &$
    end else o = (t1*mask) > 0.

    ; @@ not just faster to do filter image noconf?
    ; interpolate kern/shift map to x,y of object
    ft_kern_i = reform(interpolate(kmap_reform, xs[i],ys[i]),tsz,tsz)
    ; do all multiplications and FFTs in doubles  @@@ why?
    ft_o = fft(double(o))
;    ft_kern =  dcomplex(ft_kern_i)
    ; convolve with shifted kernel; where shift and kernel are interpolated to x,y
    m = shift( npix * double( FFT( ft_o * dcomplex(ft_kern_i), 1 ) ), s2+1, s2+1)
    obj[0,0,i] = m

    if (id[i] eq 10001L) and keyword_set(stopme) then begin
			delvarx,mm,mm1
			tvrscl, t1, os=3, pos=0,mm=mm1
			tvrscl, t2, os=3, pos=1,mm=mm
			tvrscl, s1, os=3, pos=2,mm=[0,1]
			tvrscl, o, os=3, pos=3,mm=mm1*5
			tvrscl, m, os=3, pos=4,mm=mm1
			writefits, 'object_check_tmpl.fits', t1
			writefits, 'object_check_tmpl_seg.fits', o
			writefits, 'object_check_tmpl_seg_conv.fits', m
			print,mm1
			tvrscl, t2-m*max(t2*mask)/max(m), os=3, pos=5,mm=mm
	;		stop
    end

; chart diffence between kernel0 and kernel_xy as a function of dx dy
;  m0 = shift( npix * float( FFT( ft_o * ft_kern0, 1 ) ), s2+1, s2+1) ; i dont know
;  diff[i] = total(abs(m-m0))/(total(m0) > 1e-6)
;  distx[i] = (kxy[0]-x[i])
;  disty[i] = (kxy[1]-y[i])
 end

  help,/mem, out=mem
  m=float((strsplit(mem,/ex))[3])/(1024.)^2.
  print, systime(1) - time1 ,' SECONDS, ', m,' MB MEMORY'

end

; ft_kern_i = complexarr(tsz, tsz, klen)
; for i=0L, klen-1 do begin
;   kern = interpolate(mkmodel(basis,km[2:*,i]), xg-sm[2,i], xg-sm[3,i], /grid, cubic=-0.3, missing=0.)
;   ft_kern_i[*,*,i] = fft(kern/total(kern))
; end

 ; reorder fft from grid cube(tsz,tsz,klin) to grid column list of tsz*tsz, len so that we can
 ; use interpol_map, to interpolate the (shifted) kernel grid to each x,y point of objects
 ; afterwards reorder back into tsz, tsz, len: now we have a list with shifted kernels at each position

; ft_kern_i = interpol_map(km[0,*], km[1,*], temporary(reform(ft_kern_i,tsz*tsz,klen,/overwrite)), x, y)
; ft_kern_i = reform(ft_kern_i, tsz,tsz,len,/overwrite)  ; and reform

; pick central kernel to compare to
; kxy0 = km[0:1,ik]
; ik=28
; kern0 = mkmodel(basis,km[2:*,ik])
; ft_kern0 = fft(kern0)

; real kernel
; gkern = dblarr(tsz,tsz,klen)
; for i=0L, klen-1 do $
;    gkern[*,*,i] = interpolate(mkmodel(basis,km[2:*,i]), xg-sm[2,i], xg-sm[3,i], /grid, cubic=-0.3, missing=0.)

 ; interpolate data on grid to x,y points
 ; x,y positions that fall out of bounds are set to the nearest entry
; coeff_ip = interpol_map(km[0,*], km[1,*], km[2:*,*], x, y)
; shift_ip = interpol_map(sm[0,*], sm[1,*], sm[2:*,*], x, y)

; ft_kern_ip = interpol_map(km[0,*], km[1,*], reform(ftgkern,tsz*tsz,klen), x, y)
; ft_kern_ip = reform(ft_kern_ip, tsz,tsz,len,/overwrite)  ; and reform

; f2 = interpol_map(km[0,*], km[1,*], reform(fkern,tsz*tsz,klen), x, y)
; f2 = reform(f2, tsz,tsz,len,/overwrite)  ; and reform

; check
; for i=0,len-1 do begin
;   tvrscl, g2[*,*,i], os=3, mm=[-1,1]*1e-4, pos=0
;   tvrscl,  fftinv(f2[*,*,i]), os=3, mm=[-1,1]*1e-4, pos=1
;   tvrscl, mkmodel(basis,coeff_ip[*,i]), os=3, mm=[-1,1]*1e-4, pos=2
;   d= g2[*,*,i] - fftinv(f2[*,*,i])
;   d2= g2[*,*,i] - mkmodel(basis,coeff_ip[*,i])
;   tvrscl, d, os=3, pos=3, mm=[-1,1]*1e-4
;   tvrscl, d2, os=3, pos=4, mm=[-1,1]*1e-4
;   ; ok!
;   print, i, shift_ip[0:1,i], total(abs(d)), total(abs(d2)), f='(i,5g12.5)'
; end
; ok difference between interpolation coeff and interpolating kernel is in the numerical noise



;stop
; @@@@ TODO check: the encircles aperture differences compared to applying the "average"
; or "center" kernel

; dist = sqrt(distx^2 + disty^2)
;plot, dist,diff, psym=3
;iplot, distx,disty,diff, sym_index=3, linestyle=6
; iplot looks good: very gradual change from 0 to ~15% at edges

;print, where(finite(dist) eq 0)
;print, where(finite(diff) eq 0)  ; ok no NANs!
;surface, distx, disty, diff

; IDLs FFT is weird. if do double fft use this
function fftinv, a
  npix = n_elements(a)
  return, npix*shift(rotate(fft(a),2),1,1)
end


; make hessian matrix
; Aij = sum tmpl_i*tmpl_j and observed vector r = tmpli*obs
; @@@ what to do with diagonal elements that are ~ 0.
; will bicon grad work with such a degenerate matrix?
; @@@ ruobuild breaks when diagonal values are 0
; for now, make sure the array has no 0 diagonal values
pro mkhess, x, y, obj, ll, obs, a, r, thr=thr, soname=soname, buf=buf
  COMPILE_OPT DEFINT32
  if not keyword_set(thr) then thr = 1e-6
  print, 'building hessian matrix'
  time1=systime(1)

  sz = size(obj)
  len = sz[3]
  tsz = sz[2]
  tsz2 = tsz*tsz

  ; set threshold to eps * scale of 'typical object'
  ; @@@ eps is critical parameter to get right. want as small as possible, to save matrix space
  ; but do not want to delete real objects: dynamic range in catalog ~10 mag ~ 1e4
  ; so 1e-6 times brightest source, means 1e4*1e-6 ~1% for faintest source: should be ok
  ; because faint sources have much lower S/N than 1%
  row = dblarr(len)
  r = dblarr(len)

  a = 0L
  tm1 = tsz-1L
  for j=0L,len-1L do begin ;@@@ for each object
    row = dblarr(len-j)
    obj_j = double(obj[*,*,j]) ; do multiplications in doubles

    ; find the overlapping tiles with a where -> speed is ~linear with number of objects
    ldx = ll[0,j] - ll[0,j:len-1] ;@@@ calculate distance to other objects
    ldy = ll[1,j] - ll[1,j:len-1]
    iov = where((abs(ldx) > abs(ldy)) le tm1,nov) ; is always > 1 because of diagonal elements
    ;@@@ select objects whose greater distance, in x or y,  is less than tile size

    if nov eq 0 then print,'error: zero diagonal element in object '+j+' in mkhess'

    ; for every row in the ruo matrix there are usually ~10-20 overlapping tiles
    for i=0L,nov-1 do begin
      dx = ldx[iov[i]]
      dy = ldy[iov[i]]
       ; build row of hessian matrix
      ; note: extrac is really slow, we do much faster by juggling indices
  ;     jx1 = -dx > 0
  ;     jx2 = (tm1 - dx) < tm1
  ;     jy1 = -dy > 0
  ;     jy2 = (tm1 - dy) < tm1
  ;     ix1 = dx > 0
  ;     ix2 = (tm1 + dx) < tm1
  ;     iy1 = dy > 0
  ;     iy2 = (tm1 + dy) < tm1
       row[iov[i]] = total(obj_j*extrac(double(obj[*,*,iov[i]+j]),dx,dy,tsz,tsz))

       ;each element of the row is sum obj
    end


    a  = ruobuild(a, double(row), irow, t=double(thr), un=long(buf*len), stopme=stopme)  ; each subsequent row has to have 1 element fewer
 ;   print, j, irow, long(buf*len)
;    print,(*a.ix)[0:6], (*a.jx)[0:6], (*a.xn)[0:6], (*a.xd)[0:6]

    ; observed vector: do not allow negative fluxes
    ; why ????
    llx = ll[0,j]
    lly = ll[1,j]
;    r[j] = total(obj_j*double(obs[llx:llx+tm1,lly:lly+tm1])) > 0.
    r[j] = total(obj_j*extrac(obs,ll[0,j],ll[1,j],tsz,tsz)) > 0.
  end

  help,/mem, out=mem
  m=float((strsplit(mem,/ex))[3])/(1024.)^2.
  print, systime(1) - time1 ,' SECONDS, ', m,' MB MEMORY'

return

;    row0 = dblarr(len-j)
;    for i=j,len-1L do begin
;       dx = ll[0,j]-ll[0,i]
;       dy = ll[1,j]-ll[1,i]
;      if (abs(dx) > abs(dy))  gt tm1 then continue  ; if tilesize overlaps
;       row0[i-j] = total(obj_j*extrac(double(obj[*,*,i]),dx,dy,tsz,tsz))
;       print,i-j,dx,dy
;    end
  spinfo,a
  ; check, ok, but check the length of the usage of the elements
  ; this was just a trick to make the matrix symmetric.
  oo = double(o+transpose(o)-identity(len)*o)    ; orig
  spo = spruo(double(oo),t=double(thr),ur=buf*len,un=buf*len,soname=soname) ; official way to make sparse
  aa = ruoinf(a)       ; mine, reinflated
  oo2 = ruoinf(spo)    ; offical, reinflated
;  print,*spo.ix, *spo.jx, *spo.xd
;  print,*a.ix, *a.jx, *a.xd
  print,total(abs(aa-oo))
  print,total(abs(aa-oo2))
  print,total(abs(oo-oo2))
  tvrscl, oo,mm=[-1,1]*1e-5,pos=0
  tvrscl, aa,mm=[-1,1]*1e-5,pos=1
  tvrscl, oo-aa,mm=[-1,1]*1e-7,pos=2
  tvrscl, oo2-aa,mm=[-1,1]*1e-7,pos=3
  stop

end

; create a model of size(nx,ny) by adding vector of images "obj" offsetted by ll
; if nx is missing assume obj is aligned: e.g. for adding basis function
function mkmodel, obj, f,shx,shy, nx, ny, ll, cid
COMPILE_OPT DEFINT32


 if size(obj, /n_dimensions) eq 3 then begin
    sz = size(obj)
    len = sz[3]
    tsz = sz[1]
 end else begin
    sz = size(obj)
    len = 1
    tsz = sz[1]
 end
                                ;### this will result in a object with
                                ;id cid NOT being build into the
                                ;model and thus not subtracted from
                                ;phot
  if n_elements(cid) ne 0 then begin
    print, f[cid]
    f[cid] = 0.0 * f[cid]
    print, f[cid]
 end
                                ;### this will not yeild desired
                                ;results if you want diagnostic images
                                ;on multiple sources in one tilesize
                                ;for that a loop itrating through one
                                ;cid at a time could be used.


  ; if nx is missing, assume we're adding aligned images (e.g. for adding basis functions)
  if n_elements(nx) eq 0 then begin
    nx = tsz
    ny = tsz
    ll=fltarr(2,len)
  end

  ; if type is (d)complex do complex array e.g. for adding fft's, else double array
  if (sz[sz[0]+1] eq 6) or (sz[sz[0]+1] eq 9) then $
     model = complexarr(nx,ny) else  model = fltarr(nx,ny)

  ; adding aligned images -> fast
  if max(abs(ll)) eq 0 then begin
    for i=0L,len-1 do model = model + double(f[i])*obj[*,*,i]
  end else begin
    ; when adding obj tiles with offsets into a larger frame, we need to
    ; fix for fact that "model" is filled in with array assignment, whereas obj
    ; was created with extrac (hence padded with zero's)

    ; @@ note this crashes if sources are very close to edge
    for i=0L,len-1 do begin
       ;This is where fine shift is added
       x = ll[0,i]+shx
       y = ll[1,i]+shy

       ; step1: determine effective tile size; i.e. the part of the tile that is inside model boundary
       tszx = (nx - x) < tsz + x*(x lt 0)
       tszy = (ny - y) < tsz + y*(y lt 0)

       ; step2: fix offset relative to lower-left pixel; i.e. register the obj + model
       if x lt 0 or y lt 0 then add_model = shift(obj[*,*,i],x*(x lt 0),y*(y lt 0)) else add_model = obj[*,*,i]

       ;print, x>0, x>0+tszx-1, y>0, y>0+tszy-1
       model[x>0:x>0+tszx-1,y>0:y>0+tszy-1] = $
             model[x>0:x>0+tszx-1,y>0:y>0+tszy-1] + double(f[i])*add_model[0:tszx-1,0:tszy-1]
     end ; for
  end ; else

  return, model
end

;@@@ newnew
function eval_fcn, p, dp
  common mpfit_com,tile, objt,fpost,llt,sz,xi,yj,ii,jj,indx,indy

 ;p = [shx,shy,fpost]

 ;make fullsize model to avoid edge effects/pixel location confusion
 yfit =mkmodel(objt,p[2:*],p[0],p[1],sz[1],sz[2],llt)

 ;print, size(yfit(xi[indx[ii]]:xi[indx[ii]+2],yj[indy[jj]]:yj[indy[jj]+2]))

 ; extract same region as tile from yfit
 rest = tile - yfit(xi[indx[ii]]:xi[indx[ii]+2],yj[indy[jj]]:yj[indy[jj]+2])

 return, reform(rest, n_elements(rest),/overwrite)
end



pro iphot, fimg1, fseg1, fimg2, fexp2, fcat, par = fpar, tilesize=tilesize, $
           kernelmap=fkernelmap, shiftmap=fshiftmap, buf=buf, raper=raper, rms_sz=rms_sz, fwhm1=fwhm1, $
          outname=outname, soname=soname
COMPILE_OPT DEFINT32
  resolve_routine, 'soi', /either, /COMPILE_FULL_FILE

  if not keyword_set(outname) then outname=file_basename((strsplit(fimg2,'.',/extract))[0])
  print, "outname = ", outname
  img1 = readfits(fimg1,/silent,nanval=0)
  img2 = readfits(fimg2,/silent,nanval=0)
  if not keyword_set(fexp2) then fexp2 = 1.0*finite(img2) else exp2 = readfits(fexp2,/silent,nanval=0)
  seg1 = long(readfits(fseg1,/silent,nanval=0))    ; force long integers; @@@  note ecdfs detection map contained non-integers
  cat=(read_ascii(fcat)).(0)

;  read basis functions for rebuilding psf: this could be more flexible
  par  = fpar  ;(read_ascii(fpar,comment='#')).(0)
  ; set tilesize to nearest larger odd integer (only if even) and always greater than kernel size
  if keyword_set(tilesize) then tsz = long(round(tilesize)) else tsz = long(par[0])
  ;tsz =  long(tsz + ((tsz+1) mod 2) > par[0])
  print, tsz


  ; build the basis in the correct tile size, but always crop on the maximum kernel radius
  basis = hermite_basis(par[2:2+par[1]-1],par[2+par[1]:*],tsz,rmax=par[0]/2)
  ; hermte_basis(n, beta, s, rmax)


  ;## read kernelmap from mkkern output
  restore, fkernelmap ; gx, gy, kernel

  ;kernelmap =  (read_ascii(fkernelmap,comment='#')).(0)
  if keyword_set(fshiftmap) then shiftmap =   (read_ascii(fshiftmap,comment='#')).(0)
  ;shiftmap = (read_ascii(imgdir+"out.shift.map",comment="#")).(0)

  ; kernels and tile sizes are always odd
  ; for convolutions if size kern is odd, make tile size odd, and vice versa. best practice is to always make odd
  ; note: if not odd, "convolve.pro" doesnt center well
  ; for tilesize: fast_factor finds nearest larger integer that has small prime factors (for fast fft)
  sz = size(img1)

  x = cat[0,*]-1.  & y = cat[1,*]-1.   ; convert from FITS to IDL coord
  id = long(cat[2,*])                 ; force integer ID
  iok = (where(x gt 0 and x lt sz[1]-1 and y gt 1 and y lt sz[2]-1, n0)) ; only do source that lie inside map
  print, '####cat####',size(iok)
  _iphot, img1, seg1, img2, exp2, x[iok], y[iok], id[iok], long(tsz), gx, gy, buf=buf, raper=raper,$
          basis=basis, kernelmap=kernel, shiftmap=shiftmap, outname=outname, $
          rms_sz=rms_sz, fwhm1=fwhm1, soname=soname

end




pro ltest
  img = fltarr(2000,2000)
  ll=fltarr(2,1e5) + 1000

  t0=systime(1)
  tm1 = 100
  for i=0L,1e5-1 do begin
    timg = img[ll[0,i]:ll[0,i]+tm1,ll[1,i]:ll[1,i]+tm1]
  end
  print, systime(1)-t0
stop
  t0=systime(1)
  tm1 = 100
  for i=0L,1e5-1 do begin
    timg = extrac(img,ll[0,i],ll[1,i],tm1+1,tm1+1)
  end
  print, systime(1)-t0

end



; ------------------------------------------------------------------------ from here only tests

pro testsoibuild
  soname = expand_path('$IFL_DIR')+'/lib/libifl.'+lib_so_ext()

  thr=1e-6
  a = identity(5)+0.1
  a[1,1] = 0 & a[1,3] = 0 & a[3,1] = 0
  b = spruo(a,t=double(1e-6),ur=26,un=26,soname=soname)

  c  = ruobuild(0, a[0:*,0], t=1e-6)
  c  = ruobuild(c, a[1:*,1], t=1e-6)
  c  = ruobuild(c, a[2:*,2], t=1e-6)
  c  = ruobuild(c, a[3:*,3], t=1e-6)
  c  = ruobuild(c, a[4:*,4], t=1e-6)

cexam,b
print,(*b.ix)[0:10], (*b.jx)[0:10], (*b.xn)[0:10], (*b.xd)[0:10]
cexam,c
print,(*c.ix)[0:10], (*c.jx)[0:10], (*c.xn)[0:10], (*c.xd)[0:10]

  print, ruoinf(b)-a
  print, ruoinf(c)-a
  ; ok!
end

   ; alternatively use sparse matrices
pro sparsetest, nx, ny, s, t=t, m=m, soname=soname
   if not keyword_set(soname) then soname = expand_path('$IFL_DIR')+'/lib/libifl.'+lib_so_ext()
;   soname = '/export/data1/ivo/PROG/idl/ifl/lib/libifl.'+idlutils_so_ext()
 ;  t1 = systime(/sec)

   fwhm = nx/sqrt(s)/10.
   rlim = long(7.*fwhm)
   print, 'FWHM', fwhm, ' RLIM ', rlim

   id = lindgen(s)+1
   xlis = randomu(1002L,s,1)*nx*0.8+ny*0.1
   ylis = randomu(1003L,s,1)*nx*0.8+ny*0.1
   f = abs(randomn(1004L,s)*10+31.)
   f[3]=100.
   err = 0.001   ; fixed error

   ; generate cube of templates tsz x tsz x len
   ; save offset coordinates of lower left corner pixel ll=[x,y]
   mkcube, xlis, ylis, rlim, fwhm, csp, ll, rxy
   obs = mkmodel(csp, f,shx,shy, nx, ny, ll) + err*randomn(1005L,nx,ny)

   erase
   tvrscl,obs,pos=0

   t1 = systime(/sec)

 ;  print,minmax(obs-obs_sp)
 ;  xsh = randomu(1007L,4)*4-2
 ;  ysh = randomu(1008L,4)*4-2
;FUNCTION xyoff, a, b, fna, box=box, maska=maska, maskb=maskb, $
;         ccf=ccf, shfine=shfine, quiet=quiet
 ;   obs_sh =  interpolate(obs, indgen(nx)+xsh[0],indgen(ny)+ysh[0], missing=0, /GRID, cubic=-0.35)
 ;  sh = xyoff(obs, obs_sh, fna, box=2*fwhm, ccf=ccf, shfine=shfine,/quiet)

   ; make hessian matrix
   mkhess, xlis, ylis, rlim, csp, ll, obs, a, r

  ; clip to 0 pixels with values lt 1e-6 (flux profile normalized to total=1 ~ peak flux 0.0022) or less
  ; then 1/2000 of peak value for gaussian throw away 0.04% of flux
   lothr = 1e-6
   asp = spruo(double(a),t=double(lothr),ur=s+1,un=s+1,soname=soname)
   spinfo,asp
 ;  atest = ruoinf(asp) &  print,minmax(a*(a gt lothr)-atest)
  ; solve
   v = ruopcg(asp,double(r),dblarr(s),nit=nit,tol=1e-10,soname=soname);

   !p.multi=[0,2,1]
   d = 1.086*(f-v)/f
   plothist, d, bin=robust_sigma(d)/4., xr=[-1,1]*max(robust_sigma(d))*20.
   plot, -2.5*alog10(f), d, psym=3, yr=[-1,1]*max(robust_sigma(d))*20.

   t=systime(/sec)-t1
   help,/mem,out=mem
   m=float((strsplit(mem,/ex))[3])/1e6

   print,' TIME MEMORY ', t,m
   print, 'rsigma minmax ', robust_sigma(d), minmax(d)
stop

end

pro fulltest, nx, ny, s, t=t, m=m
   t1 = systime(/sec)

   fwhm = nx/sqrt(s)/5.
   rlim = long(7.*fwhm)
   print, 'FWHM', fwhm, ' RLIM ', rlim

   cube=dblarr(nx,ny,s)
   id = lindgen(s)+1
   xlis = randomu(1002L,s,1)*nx*0.8+ny*0.1
   ylis = randomu(1003L,s,1)*nx*0.8+ny*0.1
   f = abs(randomn(1004L,s)*10+31.)
   f[3]=100.
   err = 0.001   ; fixed error

  ; straightforward linear least squares
  ; build cube + model
   model=0B
   for i=0,s-1 do begin
     cube[0,0,i] = psf_gaussian(npix=nx[0], fwhm=fwhm,centroid=[xlis[i],ylis[i]],/norm)
     j=check_math()
   end
   for i=0,s-1 do model = model + f[i]*cube[*,*,i]
   len=n_elements(model)
   erase
   obs = shift(model,0,0) + err*randomn(1005L,nx,ny)
   tvrscl,obs,pos=0

   ; brute force lls
   y = reform(obs,1,len)
   x = fltarr(s,len)
   for i=0,s-1 do x[i,*] = reform(cube[*,*,i],1,len)
   for i=0,0 do lar = la_least_squares(x,y,residual=res)

   !p.multi=[0,2,1]
   d = 1.086*(f-lar)/f
   plothist, d, bin=robust_sigma(d)/4., xr=[-1,1]*max(abs(d))*3.
   plot, -2.5*alog10(f), d, psym=3, yr=[-1,1]*max(abs(d))*3.

   print, max(1#(f-lar)/f*100.)
   t=systime(/sec)-t1
   help,/mem,out=mem
   m=float((strsplit(mem,/ex))[3])/1e6

   print,' TIME MEMORY ', t,m
   print, 'rsigma minmax ', robust_sigma(d), minmax(d)

end

pro testfft
  a = psf_gaussian(ndim=2,npix=71,fwhm=5)
  psf = psf_gaussian(ndim=2,npix=51,fwhm=8)
  psfsh = psf_gaussian(ndim=2,npix=51,fwhm=8,centroid=[25,33])
  ca = convolve(a,psfsh)

  ; make sure it is even,even or odd, odd (kernel vs tile size)
  ; so that difference is always even
  ksz = (size(psf))[1]
  tsz = (size(a))[1]
  npix = n_elements(a)
  s2 = tsz/2 + (tsz MOD 2)	;shift + correction for odd size images.

  ft_a = FFT(a,-1 )

  ; fft of shifted psf
  tpsh = extrac(psfsh, (ksz-tsz)/2, (ksz-tsz)/2, tsz, tsz)
  ft_psh = fft( tpsh, -1, /OVERWRITE )
  cc = npix * float( FFT( ft_a * ft_psh, 1 ) )
  ca2 = shift( cc, s2, s2)

  ; shifted fft of original psf : DOESNT WORK
  tp = extrac(psf, (ksz-tsz)/2, (ksz-tsz)/2, tsz, tsz)
  ft_psh2 = fft( tp, -1, /OVERWRITE )
  rp = real_part(ft_psh2)
  ip = shift(imaginary(ft_psh2),7,7)
  cc = npix * float( FFT( ft_a * complex(rp,ip), 1 ) )
  ca3 = shift( cc, s2, s2)

  tvrscl, a, mm=[-0.1,0.1], pos=0
  tvrscl, ca, mm=[-0.1,0.1], pos=1
  tvrscl, ca2, mm=[-0.1,0.1], pos=2
  tvrscl, ca3, mm=[-0.1,0.1], pos=2
  tvrscl, ca-ca3, mm=[-0.1,0.1], pos=3

end

pro aperphotostest

; comes after this in _phot
;   if n_elements(raper) gt 0 then $
;     aperphot, raper, obj, img2, res, rms, fpos, ll, rxy, flux, err, fres, chi, absdev, rmsfac=rmsfac

; to check the effect on oversampling on measured fluxes, as fraction of measured error.
; even for bright sources S/N > 30 the error is less than half the rms even for radii of 2 pixel
; which is smallest reasonable pixel radius for aperture

   aperphot, raper, obj, img2, res, rms, fpos, ll, rxy, flux5, err5, fres, chi, absdev, rmsfac=rmsfac, oversample=5.
  ; differences are pretty small; 10% of errors; ah thats because most sources undetected
  ib = where(flux[2,*]/err[2,*] gt 30.)
  plot,[0], xr=[-2,2],yr=[0,40] & for k=0,n_elements(raper)-1 do  plothist, (flux[k,ib]-flux5[k,ib])/err5[k,ib],bin=0.05, xr=[-2,2], yr=[0,50],/overplot,linestyle=k
  for k=0,n_elements(raper)-1 do print,robust_sigma((flux[k,ib]-flux5[k,ib])/err5[k,ib])
 ; differences 1.7,0.9,0.4 * rms in 2,3, 6 radius aperture for not oversampling

  ; now difference between x2 and x5
  aperphot, raper, obj, img2, res, rms, fpos, ll, rxy, flux, err, fres, chi, absdev, rmsfac=rmsfac, oversample=2.
  plot,[0], xr=[-2,2],yr=[0,50] & for k=0,n_elements(raper)-1 do  plothist, (flux[k,ib]-flux5[k,ib])/err5[k,ib],bin=0.05, xr=[-2,2], yr=[0,50],/overplot
  for k=0,n_elements(raper)-1 do print,robust_sigma((flux[k,ib]-flux5[k,ib])/err5[k,ib])
  ; 0.5, 0.4, 0.11, for 2,3,6 raper for x2 oversampling

  ; now difference between x3 and x5
  aperphot, raper, obj, img2, res, rms, fpos, ll, rxy, flux, err, fres, chi, absdev, rmsfac=rmsfac, oversample=3.
  plot,[0], xr=[-2,2],yr=[0,50] & for k=0,n_elements(raper)-1 do  plothist, (flux[k,ib]-flux5[k,ib])/err5[k,ib],bin=0.05, xr=[-2,2], yr=[0,50],/overplot
  for k=0,n_elements(raper)-1 do print,robust_sigma((flux[k,ib]-flux5[k,ib])/err5[k,ib])
  ; 0.3, 0.2, 0.09, for 2,3,6 raper for x2 oversampling, and S/N > 10 sources
  ; for sources of S/N > 30 in 2pix; the error is 1,0.6,0.3 for 2,3,6 pix (where rms err is 0.03%; so for bright)
  ; sources
  ; ok so 3x oversampling is enough

end

pro vmake
   nx = 350
   ny = 400
   s = 300.

   fulltest, nx, ny, s, t=t1, m=m1
   sparsetest, nx, ny, s, t=t2, m=m2

   nx = 850
   ny = 850
   s = 1200.
   sparsetest, nx, ny, s, t=t3, m=m3

   nx = 3000
   ny = 3000
   s = 6000.
   sparsetest, nx, ny, s, t=t4, m=m4


   nx = 550
   ny = 550
   s = 400.
   sparsetest_shift, nx, ny, s, t=t3, m=m3

   print,nx*1.*ny*s*4./1e6
end

