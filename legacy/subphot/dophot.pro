; need to implement something that makes the interpolation of the PSF map smooth
; or at least allows one to fix the scale

;; @@@ there is probably a slight bias, depending on amount of contaminating flux
; -> can be checked with simulations and corrected for

; @@@ allow dophot PSFs to be different sizes
;     do something clever wit cutting of the where the growthcurve converges
; 
; @@@@ weird fat edges.... of PSF

pro dophot_shiftmap, param=param, _extra=extra
  readparam, param, default='phot.param', _extra=extra
  
  if not file_test(imdir) then file_mkdir,  imdir
  
  for i=0,n_elements(phot)-1 do begin
    print, 'detection shifts on ',phot[i]
    fout = imdir+'/'+file_basename(phot[i],'.fits')+'_'+file_basename(det,'.fits')+'_shift'
    if file_test(fout) then file_delete, fout

    print, '--> fixwcs_abs fit distortion map using polynomial of order '+strn(shiftmap_dim)
    img = readfits(phot[i],/silent)*sqrt(readfits(photw[i],/silent))
    detimg = readfits(det,/silent)
    wcslfit, phot[i], det, kx, ky,degree=shiftmap_dim, h2=detwcs

    nx = sxpar(detwcs,'NAXIS1')
    ny = sxpar(detwcs,'NAXIS2')
    pixscl = pixscale(detwcs)
    print, 'PIXEL SCALE ', pixscl

    img_align = poly_2d(img,kx,ky,1,nx,ny,missing=!values.f_nan)
    detobj, img_align, f1, e1, bg1, x1, y1, nobj=nobj, box=shiftmap_box/pixscl, objthresh=shiftmap_snrlim[i], snrlim=shiftmap_snrlim[i]

    gcntrd, detimg, x1, y1, x2, y2, shiftmap_box/pixscl, /silent ; corresponding location in wcs
    fixwcs_map, x1, y1, x2, y2, kx_wcs, ky_wcs, x1t, y1t, shiftmap_dim, plim=shiftmap_plim, $
                    nobj=nobj, nok=nok, pscale=pixscl, power=power, fout=fout+'.ps'
    if nok gt (shiftmap_dim+1)^2 then begin
       save, kx_wcs, ky_wcs, file=fout+'.sav'
       delvarx, img_align, img
    end else print, 'NOT enough sources for deriving shiftmap. Skipping....'
  end ; image

; ok.
; the kx_wcs transformation transforms positions in the wcs grid from the locations of the objects
; to the locations of the reference image. this transf can be used in wcslfit to "distort" the reference
; target grid.
end

pro dophot_reg, _extra=extra
  readparam, param, default='phot.param', _extra=extra

  det=file_search(det,/fully) 
  detw=file_search(detw,/fully) 
  
  file_mkdir, imdir
  if not keyword_set(skysub_scale) then skysub_scale = 5

  if keyword_set(skysub_det) then begin
     if file_test( imdir+'/'+file_basename(det) ) and  not keyword_set(force) then begin
        print, 'found '+ imdir+'/'+file_basename(det) + '... SKIPPING'
     end else begin
       nbin = ceil(skysub_scale*tile_size[0]/pixscale(det))
       print, 'skysubtracting detection image in bin=',nbin
       imdet = readfits(det,hdet, /silent)
       bg = gbg(imdet, bin=nbin, nsig=skysub_nsig, fwidth=3, nsmooth=nsmooth, $
               lsig=lsig,hsig=hsig, stopme=stopme, display=display)
       imdet[where(finite(imdet) eq 0 or finite(bg) eq 0 or imdet eq 0,/null)] = 0.0
       writefits, imdir+'/'+file_basename(det), float(temporary(imdet)-temporary(bg)), hdet
     end
  end else begin 
  print, det, detw
    if not file_test(imdir+'/'+file_basename(det)) then spawn, 'ln -s '+det+' '+imdir+'/'+file_basename(det)
  end
  
 if keyword_set(detw) then if file_test(detw) and not file_test(imdir+'/'+file_basename(detw)) then spawn, 'ln -s '+detw+' '+imdir+'/'+file_basename(detw)

  if not keyword_set(reg_rescale) then reg_rescale = replicate(1.0,n_elements(phot)>1) $
  else print, 'rescaling ',phot,' by ', reg_rescale

  for i=0,n_elements(phot)-1 do begin
    print, 'registering ',phot[i]
    if file_test( imdir+'/'+file_basename(phot[i]) )  and not keyword_set(force) then begin
      print, 'found '+ imdir+'/'+file_basename(phot[i]) + '... SKIPPING'
      continue
    end
    ; use a shift correction map if it exists
    wcscor = imdir+'/'+file_basename(phot[i],'.fits')+'_'+file_basename(det,'.fits')+'_shift.sav'
    wcslfit, phot[i], det, kx, ky, deg=reg_dim, err=err, h2=h2, exten=reg_exten, fluxcor=fluxcor, wcscor=wcscor

    imphot = readfits(phot[i],/silent,exten=reg_exten)

    if keyword_set(skysub_phot) then begin
      nbin = ceil(skysub_scale*tile_size[i+1]/pixscale(det)*sqrt(fluxcor))
      print, 'skysub ', nbin
      bg = gbg(imphot, bin=nbin, nsig=skysub_nsig, fwidth=3, nsmooth=nsmooth, $
               lsig=lsig,hsig=hsig, stopme=stopme, display=display)
      ibad = where(finite(imphot) eq 0 or finite(bg) eq 0 or imphot eq 0)
      imphot -= bg
      imphot[ibad] = !values.f_nan
    end

    preg = poly_2d(temporary(imphot),kx,ky,2, $
         sxpar(h2,'NAXIS1'),sxpar(h2,'NAXIS2'),missing=!values.f_nan,cubic=-0.35)
    writefits, imdir+'/'+file_basename(phot[i]), float(preg*fluxcor*reg_rescale[i]), h2

    if keyword_set(photw) then if file_test(photw[i]) then begin
        print, 'registering ',photw[i]
        imphotw = readfits(photw[i],/silent)
        preg = poly_2d(temporary(imphotw),kx,ky,2, $
           sxpar(h2,'NAXIS1'),sxpar(h2,'NAXIS2'),missing=!values.f_nan,cubic=-0.35)
        writefits, imdir+'/'+file_basename(photw[i]), float(preg), h2
    end
    delvarx, preg
  end
end



pro dophot_sex, _extra=extra
   readparam, param, default='phot.param', _extra=extra

  if not file_test('sex/default.sex') then begin
    file_mkdir, 'sex/'
    file_copy, '../../setup/sex/default*', 'sex/', /overwrite
    file_copy, '../../setup/sex/gauss*', 'sex/', /overwrite
  end

  ndet = '../'+imdir+'/'+file_basename(det)
  if keyword_set(detw) then ndetw = '../'+imdir+'/'+file_basename(detw)

  pushd, 'sex'
  if not keyword_set(detect_deblend_mincont) then detect_deblend_mincont = 1e-5
  if not keyword_set(detw) then ndetw = 'NONE -WEIGHT_TYPE NONE '
  print, 'detecting on ',ndet,' ', ndetw
  sexcmd =  'sex '+ndet+' -c default.sex -WEIGHT_IMAGE '+ndetw+' -DETECT_THRESH '+$
    strn(detect_thresh)+' -DEBLEND_MINCONT '+str(detect_deblend_mincont,5)+$
    ' -CATALOG_NAME '+file_basename(det,'.fits')+'.cat -FILTER_NAME '+filter_name+$
    ' -CHECKIMAGE_NAME  '+file_basename(det,'.fits')+'_seg.fits'
  print, sexcmd
  spawn, sexcmd
  popd
end

; @@@  change prepcat so that if more than 1 source within radius
;      then merge them.
pro dophot_prepcat, _extra=extra
   readparam, param, default='phot.param', _extra=extra

  imdet = imdir+'/'+file_basename(det)

  readcol, 'sex/'+file_basename(det,'.fits')+'.cat', sx, sy, sid, comment='#',/silent
  sx--
  sy--
  nsex = n_elements(sx)
  seg = readfits('sex/'+file_basename(det,'.fits')+'_seg.fits',h,/silent)
  pscl = pixscale(imdet)
  print, 'pixel scale ',pscl
  rlim = prepcat_rlim / pscl
  tsz = round((min(tile_size) > (rlim*2))/pscl/6) ; we only need a small tile...
  rlim_thresh = 2.0  ; minimum distance threshold always 1.0 pixel

  ss = replicate('-1', nsex)
  for k=0,n_elements(cat)-1 do begin
    print, 'processing ',cat[k]
    rcat, cat[k],/unpack
;    readcol, cat[k], id, ra, dec, f='a,a,a', comment='#' ,/silent
    if strmatch(ra[0],"*:*") then dra = rdten(ra,/hour) else dra = float(ra)
    if strmatch(dec[0],"*:*") then ddec = rdten(dec) else ddec = float(dec)
    adxy, headfits(imdet), dra, ddec, x, y

    im = match_2d(x, y, sx, sy, 50, match_distance=d)
    im[where(d gt (rlim > rlim_thresh),/null)] = -1.
    if keyword_set(prepcat_ignore_existing) then im[*] = -1
    ibad = where(im eq -1, nbad, ncomple=nok,/null)
    
    img =  readfits(imdet,/silent)
    getstamps, img, x, y, sth, tsz=tsz
    sz = size(img,/dim)

    mkcoo, tsz, d=dd
    find_mask = dd lt ceil(rlim)
    
    new_id = long(max(sid))+1
    for i=0L,n_elements(im)-1 do begin
      j = im[i]
      if not(x[i] gt 0 and x[i] lt sz[0]-1 and y[i] gt 0 and y[i] lt sz[1]-1) then begin
        print, id[i], ' outside field of view... skipping '
        if keyword_set(stopme) then stop
        continue
      end
      lx = round(x[i])-tsz/2L
      ly = round(y[i])-tsz/2L
      tx = x[i] - lx
      ty = y[i] - ly
      t = extrac(seg, lx, ly, tsz, tsz)

 ; find id of objects in tile, within find_mask. if tile empty, id_obj  = 0
;  tdet = t*find_mask
; hdet = HISTOGRAM(tdet,min=1,max=max(t>1),rev=r)
 ; j_obj = where(hdet gt 0, complement=inok, ncomplement=nnok, nobj) 

;      if j ne -1 then begin
     if j ne -1 then begin 
        ; select object with most segment pixels within search mask 
        ; match is found at r < rlim in segmap and distance segmap
        ; add r=rlim/2 circular segment to map with same id
        ; @@@ shouldnt we just select nearest?
         mkcoo, tsz, cntr=[sx[j]-lx,sy[j]-ly], d=dd         
         t[where(dd le ((rlim/2.) > rlim_thresh))] = j+1
         print, "found ",long(sid[j]), "  ", id[i], " d=",d[i]*pscl,' arcsec'
         ss[j] = id[i]
;        print,'SEG found ',id_obj,' -> '+str( id_match)
; tim = extrac(img, lx, ly, tsz, tsz)
; tvs, t, os=5,pos=0,/minmax
; tvs, tim, os=5,pos=1 
;print, t[14:18,14:18]
      end else begin
        ; if no match in catalog, but segmentation pixels within view with id > nsex
        ; so it was added previously: add it again with the same id. 
        tid = t[round(tx),round(ty)]  
      ; no match is found, add new segment to segmap and to x,y list

         mkcoo, tsz, cntr=[tx, ty], d=dd

        if tid gt nsex then begin 
           print, 'REFOUND ',tid, ' ',id[i]
           sid = [sid, tid] 
           print, "adding",new_id, "  ", id[i], " d=",d[i]*pscl,' arcsec'
         end else begin 
           sid = [sid, new_id]
           t[where(dd le ((rlim/2.) > rlim_thresh) )] = new_id
           new_id++
        end  
  
         sx = [sx,x[i]]
         sy = [sy,y[i]]         
         ss = [ss, id[i]] 
         print, "adding",new_id, "  ", id[i], " d=",d[i]*pscl,' arcsec' 
   end
;      tvs, t, os=2,mm=minmax(t[where(t ne 0)])+[-1,1]*10
  ;      tvs, dd le rlim, os=2, pos=1, mm=[-1,1]
  ;      tvs, sth[*,*,i], pos=2, os=2, fac=10
;        seg[lx:lx+tsz-1,ly:ly+tsz-1] = t
; this wouldnt be necessary if we have a tsz/2 buffer on all sides
     ; step1: determine effective tile size; i.e. the part of the tile that is inside model boundary
       tszx = (sz[0] - lx) < tsz + lx*(lx lt 0)
       tszy = (sz[1] - ly) < tsz + ly*(ly lt 0)
    ; step2: fix offset relative to lower-left pixel; i.e. register the tile and segmap

   if lx lt 0 or ly lt 0 then add_tile = shift(t,lx*(lx lt 0),ly*(ly lt 0)) else add_tile = t
;       print, lx>0, lx>0+tszx-1, ly>0, ly>0+tszy-1

     seg[lx>0:lx>0+tszx-1,ly>0:ly>0+tszy-1] = add_tile[0:tszx-1,0:tszy-1]
  end
    print, 'total ',n_elements(im),' found ', nok, ' within ',prepcat_rlim, ' added ',nbad
  end
;  stop
  forprint, sx+1, sy+1, long(sid), ss, f='(2f12.3,i10,"     ",a)', textout=file_basename(det,'.fits')+'.xy', comment='# xsex ysex idsex id_orig'
  writefits, imdir+'/'+file_basename(det,'.fits')+'_seg.fits', seg, h
end

pro dophot_star, _extra=extra
   readparam, param, default='phot.param', _extra=extra

  if not file_test(imdir) then file_mkdir, imdir

  print, 'selecting stars in '+det+' from sex/'+file_basename(det,'.fits')+'.cat'
  tsz = min(tile_size) / pixscale(det)

  bin=0.1            ; binwidth for rh historgram
  rfac = 1.2
  raper = rfac^indgen(20)
  tsz = ceil(max(raper)*2)
  naper = n_elements(raper)

  img = readfits(det,/silent,nanval=0)
  sz = size(img)

;  readcol, file_basename(det,'.fits')+'.xy', x, y, id, /silent
  readcol, 'sex/'+file_basename(det,'.fits')+'.cat', x, y, id, /silent, comment='#'
  n = n_elements(x)
  x--
  y--

; get peak flux on x,y locations
  ixy  = round(y) * sz[1] + round(x)
  ixy3 = transpose(rebin(ixy,n,5)) + rebin([-1,0,1,-sz[1],sz[1]],5,n)
  f = total(img[ixy3],1)
  i= where(f gt percentiles(f,val=1-star_prh), nrh)

; aperture fluxes, growthcurces and half light radi
  faper, img, x[i], y[i], raper, ff
  rh = fltarr(nrh)
  fc = ff / rebin( transpose(rebin(max(ff,dim=1),nrh,naper)) ,naper,nrh)
  for j=0,nrh-1 do rh[j] = interpol(raper,fc[*,j],0.5)
  m = mag(ff[4,*])    ; r=2 pixel magnitudes

  ; for these sources calc light weighted axis ratios within segmentation map
  getstamps, imdir+'/'+file_basename(det), x[i], y[i], std, tsz=tsz, nanval=0.
  getstamps, 'sex/'+file_basename(det,'.fits')+'_seg.fits', x[i], y[i], sts, tsz=tsz

  rax = fltarr(nrh)
  for j=0,nrh-1 do begin
    iseg = where(sts[*,*,j] eq id[i[j]])
    epts = fitellipse(iseg, (std[*,*,j])[iseg], xsize=tsz, ysize=tsz, center=center, axes=axes, orientation=orientation)
    rax[j] = min(axes)/max(axes)
;    print, j, rax
  end

; select objects with rhlim to mode and with axes ratio gt star_axratio
  if keyword_set(star_rhlim_abs) then ih = where(rh lt star_rhlim_abs) else ih = where(finite(rh))
  h = histogram(rh[ih],binsize=bin,loc=loc)
  mo = (loc[where(h eq max(h))]+bin/2.)[0]
  ii = where(abs(rh - mo)/mo lt star_rhlim and rax ge star_axratio, nstar)

  forprint, x[i[ii]]+1, y[i[ii]]+1, id[i[ii]], lindgen(nstar), textout=file_basename(det,'.fits')+'_star.xy', /nocomment, format='(2f12.3,2i)'
  print, 'found ',nstar,' stars within ',star_rhlim,' pix of mode, and with axis ratio >',star_axratio

  ps, file_basename(det,'.fits')+'_star.ps',xs=22,ys=10
  cleanplot,/silent
  plotinit
  !p.multi=[0,2,1]
  !p.charsize=1.3
  cgplot, m, rh, psym=16, yr=[0,percentiles(rh,val=0.9)], xr=[percentiles(m,val=0.05)-1, percentiles(m,val=0.95)],xtit='mag', ytit='half light radius [pix]', symsize=0.3
  oplot, [-1e5,1e5],[mo,mo]-star_rhlim*mo, linest=2, col=cgcolor('forestgreen')
  oplot, [-1e5,1e5],[mo,mo]+star_rhlim*mo, linest=2, col=cgcolor('forestgreen')
  plothist, rh,bin=bin,xr=[0,2*mo],  xtit='half light radius [pix]', charsize=1.5
  oplot, [mo,mo]-star_rhlim*mo, [0,1e5],linest=2, col=cgcolor('forestgreen')
  oplot, [mo,mo]+star_rhlim*mo, [0,1e5],linest=2, col=cgcolor('forestgreen')
  print, 'mode half light ratio of ',long(star_prh*100),'% brightests objects ',mo
  ps,/close

end

pro dophot_mkpsf, _extra=extra
  readparam, param, default='phot.param', _extra=extra, stopme=stopme

 file_mkdir, psfdir
 psf =  keyword_set(psf) ? psf : (keyword_set(phot) ? [det,phot]: det) 
 if not keyword_set(psf_cat) then psf_cat =  [file_basename(det,'.fits')+'_star.xy']
 if psf_cat.length lt psf.length then psf_cat = [psf_cat, replicate(psf_cat[-1], psf.length-psf_cat.length)]
 if not keyword_set(psf_snrlim) then psf_snrlim = [400]
 if psf_snrlim.length lt psf.length then psf_snrlim = [psf_snrlim, replicate(psf_snrlim[-1], psf.length-psf_snrlim.length)]
  print, 'creating PSF maps for ', det, n_elements(psf) ? psf : ''

;print, 
  fpsf=psf.replace('_sci','')
  fpsf=fpsf.replace('.fits','_psf.fits') 
  if n_elements(psf_max_basis) ne 0 then if psf_max_basis.length lt psf.length then psf_max_basis = [psf_max_basis, replicate(psf_max_basis[-1], psf.length-psf_max_basis.length)]
  for i=0,n_elements(psf)-1 do if not file_test(fpsf[i]) then print, imdir+'/'+file_basename(psf[i]), psf_cat[i], ' snrlim:', psf_snrlim[i], ' max_basis:', n_elements(psf_max_basis) eq psf.length ? psf_max_basis[i] : !NULL

  for i=0,n_elements(psf)-1 do if not file_test(fpsf[i]) then $
    klpsf, param, image=imdir+'/'+file_basename(psf[i]), psf_snrlim=psf_snrlim[i], display=klpsf_checkplots, ratio_thresh=klpsf_ratio_thresh, starcat=psf_cat[i], $
    max_basis=n_elements(psf_max_basis) eq psf.length ? psf_max_basis[i] : !NULL, tile_size=tile_size[i], stopme=stopme, blender=blender

end

;getklpsf, root+file_basename(det,'.fits')+'.sav', gx, gy, tmplpsf
; if output pixelsize and tile size are given (tsz in pixels), then crop output
pro dophot_getpsf_aor, fpsf, ra, dec, psf, pa=pa, display=display, pixscl=pixscl, tsz=tsz, totcor=totcor
  common getklpsf_aor_com, mpsf, hpsf, map, hmap, current_psf

  if not keyword_set(pa) then pa = 0.0
  if not fpsf.equals(current_psf) then begin 
      print,'getpsf_aor --> restoring ',fpsf
      mpsf = readfits(fpsf, hpsf, exten=1,/silent)
      map = readfits(fpsf, hmap, exten=2, /silent)
      current_psf = fpsf
  end 
  szpsf = size(mpsf,/dim)
  sz = size(map,/dim)
  len = sz[3]
  xy0 = (szpsf-1)/2.
  mkgrid, szpsf, d=d
 
  if keyword_set(pixscl) then begin
     if abs(pixscale(hpsf)/pixscl-1) gt 1e-3 then begin
         rscl = pixscale(hpsf)/pixscl 
         if keyword_set(tsz) then sznew=tsz else  sznew = oddsize(round(rscl*szpsf[0]),div=3)
         gpsf = extrac(mpsf,0,0,sznew,sznew)
     end
   end else begin
     gpsf = mpsf
     rscl=1.00
   end

   adxy, hmap, ra[*], dec[*], x, y          ; use extension WCS header for map
   ix = round(x)
   iy = round(y)

  psf = fltarr([size(gpsf,/dim),x.length])
  tmp = fltarr(size(gpsf,/dim))
  for j=0,x.length-1 do begin
  
    if total(map[ix,iy,0,1:*]) eq 0 then  return
     mkgrid, sznew, d=d
    for i=0,len-1 do begin
       aor_w = map[ix,iy,0,i]  ; could interpolate for finer grid result
       aor_pa = map[ix,iy,1,i]
       if aor_w[0] le 0 then continue  ; skip if no weight
       tmp += rot(gpsf,-aor_pa[0]-pa,rscl[0],xy0[0],xy0[1],cubic=-0.5,missing=0.0)*aor_w[0]/rscl^2
       if keyword_set(display) then begin
          print, i+1, aor_w, aor_pa, format='(i,3f)'
          tvs, rot(gpsf,-aor_pa[0]-pa,rscl,xy0[0],xy0[1],cubic=-0.5,missing=0.0), fac=10, pos=0
          tvs, tmp, fac=10, pos=1
       end
    end
    
    psf[*,*,j] = tmp/total(tmp*(d lt  (sznew-1.0)/2.))
  end

  if keyword_set(display) then print, ra,dec, rscl, total(psf)
end
   ; print, (cra[round(x),round(y)]-ra)*3600, (cdec[round(x),round(y)]-dec)*3600
   ; print, ra, dec, x, y, ix, iy
;   cra = map[*,*,0,0]
;   cdec = map[*,*,1,0]
;   ix = value_locate(cra[*,0], ra)   ; could interpolate for finer grid result
;   iy = value_locate(cdec[0,*], dec)

pro dophot_starcat,  _extra=extra
   readparam, param, default='phot.param', _extra=extra

; add star ids to sextractor catalog
  readcol, file_basename(det,'.fits')+'_star.xy', xstar, ystar, idsex_star, idstar, f='f,f,l,l', /silent
  readcol, file_basename(det,'.fits')+'.xy', x, y, idsex, id_orig, f='f,f,l,a', /silent
  for i=0L,idsex_star.length-1 do begin
    j = where(idsex eq idsex_star[i])
    if j[0] eq -1 then continue
    id_orig[j] = 'star_'+idstar[i].tostring()
    if keyword_set(verbose) then  print, i, j, idsex_star[i], '   '+id_orig[j], f='(3i,a)'
  end
  forprint, x, y, idsex, id_orig, f='(2f12.3,i10,"     ",a)', $
        textout=file_basename(det,'.fits')+'.xy', comment='# x y id id_orig',/silent

; photometry on detection image
  img=readfits(det,/silent, hdet)
  faper, img, xstar-1, ystar-1, subphot_raper/pixscale(hdet), fdet

  ; write separate star catalog
  ; @@@@ XY2AD was WRONG FITS !!!!
  xyad, hdet, xstar-1, ystar-1, ra, dec
  id = 'star_'+idstar.tostring()
  forprint, id,  ra, dec, xstar, ystar, fdet,  format='(a,f,f,f,f,f)', textout=file_basename(det,'.fits')+'_star.cat', $
            comment='# id ra dec x y fdet',/silent
            
 ; call dophot with star catalog as input
  dophot_doall, scat=file_basename(det,'.fits')+'.xy',  $
     cat=file_basename(det,'.fits')+'_star.cat', outdir='star/',  _extra=extra
     
  dophot_gencat, param=param, cat=file_basename(det,'.fits')+'_star.cat', outdir='star/', /verbose, force=force
     
end

pro dophot_mkkern_aor,  _extra=extra
  readparam, param, default='phot.param', _extra=extra, stopme=stopme

  detwimg = readfits(detw,hdet,/silent,nanval=0) 
  szdet = sxpar(hdet,'NAXIS*')
 
; take grid delta from PSF map
  for iphot=0,n_elements(phot)-1 do begin
    fpsf = repstr(repstr(phot[iphot],'.fits'),'_sci','')+'_psf.fits'
    if file_test(fpsf) eq 0 then begin
      print, fpsf, ' not found... skipping...'
      continue
    end
    map = readfits(fpsf,exten=2,/silent,hmap)
    szmap = (size(map,/dim))[0:1] 
    mkcoo, szmap, mapx, mapy,c=[0,0]
    xyad, hmap, mapx, mapy, gra, gdec
    adxy, hdet, gra, gdec, gx, gy
    wxy = detwimg[gx[*],gy[*]]
 
   ;getklpsf, root+file_basename(det,'.fits')+'.sav', gx, gy, tmplpsf
 
   outname=psfdir+'/kern_'+file_basename(det,'.fits')+'_'+file_basename(phot[iphot],'.fits')+'.sav'
   if file_test(outname) and not keyword_set(force) then begin
     print,outname,' found, skipping...'
     continue
   end
    print, 'making kernel map from ', root+file_basename(det,'.fits'), ' and ', $
            root+file_basename(phot[iphot],'.fits')+'.sav'
    fpsf = repstr(repstr(phot[iphot],'.fits'),'_sci','')+'_psf.fits'

    print, '#  x    y    maxdev   rchi '
    ; delete psf maps, otherwise it will keep using the maps from the previous filter
    delvarx, map, mpsf, hpsf, hmap
    rchi = fltarr(gx.length)
    maxdev = fltarr(gx.length)
    method = strarr(gx.length)
    for i=0,n_elements(gx)-1 do begin
      if gx[i] lt 0 or gx[i] gt szdet[0] or gy[i] lt 0 or gy[i] gt szdet[1] or wxy[i] eq 0 then continue
      
     ; further speed up by buffering klpsf ? aor already buffered
      getklpsf, root+file_basename(det,'.fits')+'.sav', gx[i], gy[i], tmplpsf
       getrot, hdet, pa
 
    dophot_getpsf_aor, fpsf, gra[i], gdec[i], photpsf, pa=pa, pixscl=pixscale(hdet), tsz=psz, display=display
     if total(photpsf) eq 0 then continue
 
       decon, tmplpsf, photpsf, kern, maxiter=maxiter, klim=klim, rchi=rr, max_entropy=max_entropy, $
            likelihood=likelihood, hermite=hermite, maxdev=dd, rkernel=rkernel, nbasis=kern_nbasis, $
            basis=basis, verbose=verbose, method=me, display=display, _extra=extra

          method[i] = me
          rchi[i] = rr
          maxdev[i] = dd  
          if finite(dd) eq 0 or n_elements(kern) eq 0 then continue 
          if n_elements(kernel) eq 0 then kernel = fltarr([kern.dim,gx.length])
          kernel[*,*,i] = kern          
          print, gx[i], gy[i], gra[i], gdec[i], wxy[i], dd, rr, ' ', method[i]
        if keyword_set(stopme) then stop
    end
    save, gx, gy, kernel, maxdev, rchi, method, filename=psfdir+'/kern_'+file_basename(det,'.fits')+'_'+file_basename(phot[iphot],'.fits')+'.sav'
  end
end

; @@@@ this one and aor can merge 
pro dophot_mkkern, _extra=extra
   readparam, param, default='phot.param', _extra=extra

;stop
;  if file_test(root+file_basename(det,'.fits')+'.sav') eq 0 then begin
;     print, root+file_basename(det,'.fits')+'.sav  not found, returning...'
;     return
;  end
  delvar,x,y
  getklpsf, root+file_basename(det,'.fits')+'.sav', gx, gy, tmplpsf 

  for iphot=0,n_elements(phot)-1 do begin
    if file_test(repstr(repstr(phot[iphot],'.fits'),'_sci','')+'_psf.fits') then begin
        print, 'skipping ',phot[iphot], ' psf file exists'
        continue
    end else print, 'making kernel map from ', root+file_basename(det,'.fits'), ' and ', $
            root+file_basename(phot[iphot],'.fits')+'.sav'
    getklpsf, root+file_basename(phot[iphot],'.fits')+'.sav', gx, gy, photpsf,  _extra=extra

    rchi = fltarr(gx.length)
    maxdev = fltarr(gx.length)
    method = strarr(gx.length)
    print, '#  x    y    maxdev   rchi '
    for i=0,n_elements(gx)-1 do begin
       decon, tmplpsf[*,*,i], photpsf[*,*,i], kern, maxiter=maxiter, klim=klim, rchi=rr, max_entropy=max_entropy, $
            likelihood=likelihood, hermite=hermite, maxdev=dd, rkernel=rkernel, nbasis=kern_nbasis, $
             basis=basis, verbose=verbose, method=me, display=display, _extra=extra
          method[i] = me
         rchi[i] = rr
       maxdev[i] = dd
       if n_elements(dd) ne 0 then if finite(dd) eq 0 then begin 
            print, 'WARNING Kernel is bogus!!!! '
             kern = 0.0
        end 
         if n_elements(kernel) eq 0 then kernel = fltarr([kern.dim,gx.length])
          kernel[*,*,i] = kern          
          print, gx[i], gy[i], dd, rr, ' ', method[i]
        if keyword_set(stopme) then stop
    end
    save, gx, gy, kernel, maxdev, rchi, filename=psfdir+'/kern_'+file_basename(det,'.fits')+'_'+file_basename(phot[iphot],'.fits')+'.sav'
  end
end

pro dophot_getkern, fkern, x, y, kern, display=display, gx=gx, gy=gy, kernel=kernel
  if not keyword_set(kernel) then  begin
    restore, fkern
    kernel = reform(kernel, [(size(kernel))[1], (size(kernel))[1], size(gx,/dim)])
    help,kernel, gx, gy
  end
  tsz = (size(kernel))[1]   ; @@@ check should take existing grid!!!!!!! 
  len =  n_elements(x)
  kern = fltarr([tsz,tsz,len])
  sz=size(gx,/dim)

 ; kk = reform(kernel, [tsz,tsz,size(gx,/dim)])
  for i=0,len-1 do begin
    ckern = interpolate(kernel, 1.*(sz[0]-1)*(x[i]-min(gx))/(max(gx)-min(gx)), 1.*(sz[1]-1)*(y[i]-min(gy))/(max(gy)-min(gy)), missing=0.,cubic=-0.4)
    kern[*,*,i] = ckern/total(ckern)
  end
  
end

pro dophot_doall, param=param, display=display, force=force, single=single, linear=linear, $
  tweakbg=tweakbg, tweakkern=tweakkern, reverse=reverse, $ ; photreverse=photreverse, clashed name with phot -> changge $
  oldkernel=oldkernel, scat=scat, redokernel=redokernel, index=index, _extra=extra

   readparam, param, default='phot.param', _extra=extra
  if not keyword_set(outdir) then fdecomp, cat, disk, dir, outdir, junk
  ; if single object quick check if output already exists so we can skip 
  if keyword_set(single) and not keyword_set(force) then begin
       if product(file_test( outdir+'/'+file_basename(phot,'.fits')+'/'+single+'/*phot.cat')) eq 1 then begin 
        statusline, 'all outputs exist... skipping  '+single
        return
      end  
  end     

  if not keyword_set(subphot_nsigma) then subphot_nsigma=3.0
  file_mkdir, outdir

  himg = headfits(imdir+'/'+file_basename(det))
  pixscl = pixscale(himg)
  if keyword_set(photreverse) then phot=reverse(phot)
  if not keyword_set(subphot_photbin) then subphot_photbin=pixscale(himg)
  
  if not keyword_set(scat) then scat = file_basename(det,'.fits')+'.xy'
;  readcol, scat, x, y, idsex, id_orig, f='f,f,l,a', /silent
rcat, scat,/unpack
  imseg = readfits(imdir+'/'+file_basename(det,'.fits')+'_seg.fits',/silent,nanval=0.0)
  imtmpl = readfits(imdir+'/'+file_basename(det),/silent,nanval=0.)

  missing = !null
  for iphot=0,n_elements(phot)-1 do begin
    imphot = readfits(imdir+'/'+file_basename(phot[iphot]),/silent,nanval=0.0)
    iok = where(imphot ne 0,/null)
    if keyword_set(tweakbg) then imphot[iok] -= tweakbg
 ;   if keyword_set(photw) then imphotw = readfits(imdir+'/'+file_basename(photw[iphot]),/silent,nanval=0.0)
    file_mkdir, outdir+'/'+file_basename(phot[iphot],'.fits')
    delvarx, subphot_rms, kernel

    for i=0,n_elements(cat)-1 do begin
      print, 'processing ',cat[i]      
     rcat,  cat[i], /unpack
      if keyword_set(reverse) then id = reverse(id)

      nmax = n_elements(id)
      if keyword_set(subphot_trial) then if subphot_trial gt 0 then begin
        print, 'only performing photometry on the first ',subphot_trial, ' sources'
        nmax = subphot_trial
      end
     
      delvarx, basis, rkernel
      for j=0,nmax-1 do begin
        k = where(id_orig eq id[j], nk)
        if keyword_set(single) then if id[j] ne single then continue
        if nk eq 0 then missing = [missing,id[j]]   
        ; !! this should not happen, unless prepcat wasnt run, or duplicate source.
        ; in that case the first source is silently ignored
        if nk eq 0 then print, 'Source not found!!!!  Did you run prepcat?  Otherwise it is a duplicate source'    ; !!!! this should never happen.
        if nk eq 0 then if keyword_set(stopme) then stop else continue 
        print, "found ", id[j], " in ",cat, " at index ",k," in the .xy list"
        ; skip if phot weight image is 0 
        faper, imtmpl, xsex[k], ysex[k], subphot_raper/pixscl, wobj
        if wobj eq 0 then continue
        
        ; @@ create different outputdir
        subphot_outdir = outdir+'/'+file_basename(phot[iphot],'.fits')+'/'+id_orig[k].tostring()
        print, 'starting '+subphot_outdir
        if file_test(subphot_outdir+'/'+str(long(idsex[k]))+'_phot.cat') and not keyword_set(force) then begin
          print, 'exists... skipping '
          continue
        end

        fkern = psfdir+'/kern_'+file_basename(det,'.fits')+'_'+$
                file_basename(phot[iphot],'.fits')+'.sav'
        getklpsf, root+file_basename(det,'.fits')+'.sav', x[k], y[k], tmplpsf

; in single object mode always recompute kernel on the fly 
     if file_test(fkern) eq 0 or keyword_set(redokernel) then begin
         fpsf = repstr(repstr(phot[iphot],'.fits'),'_sci','')+'_psf.fits'
         if keyword_set(oldkernel) or file_test(fpsf) eq 0 then begin
            print, 'klpsf psf'
            getklpsf, root+file_basename(phot[iphot],'.fits')+'.sav', x[k], y[k], photpsf

            decon, tmplpsf, photpsf, kern, maxiter=maxiter, klim=klim,  maxdev=maxdev, basis=basis, $
             rkernel=rkernel, likelihood=likelihood, hermite=hermite, verbose=verbose, display=display, _extra=extra
           end else begin
             print, 'precomputed PSF map - deconvolve on the fly' 
             xyad, himg, xsex[k], ysex[k], kra, kdec
             getrot, himg, pa
             print, fpsf, 'PA ', pa
             dophot_getpsf_aor, fpsf, kra[0], kdec[0], photpsf, pa=pa, pixscl=pixscale(himg), display=display
             decon, tmplpsf, photpsf, kern, maxiter=maxiter, klim=klim,  maxdev=maxdev, basis=basis,   rkernel=rkernel, likelihood=likelihood, hermite=hermite, method=me, _extra=extra
;print, median(kern[*,0:1],dim=1)/max(kern)
           end
        end else begin ; use precomputed kernel map
          print, 'precomputed kernel'
          dophot_getkern, fkern, xsex[k], ysex[k], kern, gx=gx, gy=gy, kernel=kernel
        end

   ; @@@ testing purposes
       if keyword_set(tweakkern) then begin
         print, 'tweaking kernel ',tweakkern
         help,display
           p=psf_gaussian(npixel=size(kern,/dim),fwhm=tweakkern/pixscl,/norm)
             decon, p, kern, nkern,  maxiter=maxiter, klim=klim,  maxdev=maxdev, basis=basis, $
                     rkernel=rkernel, likelihood=likelihood, hermite=hermite, display=display, _extra=extra
           kern = float(nkern)
       end

       if not keyword_set(subphot_rlim) then subphot_rlim=tile_size/2.0 
       if keyword_set(stopme) then stop
tic 
       subphot, imphot, imseg, imtmpl, tmplpsf, kern, idsex[k], [1#xsex,1#ysex,1#idsex], imtmpl2,tmpl_snrlo=tmpl_snrlo,  tmpl2_snrlo=tmpl2_snrlo, $ ;  imphotw=imphotw,
            rlim=subphot_rlim[iphot+1]/pixscl, raper=subphot_raper/pixscl, masksig=masksig, outdir=subphot_outdir, rms=subphot_rms, nomask=subphot_nomask, $
            maxshift=subphot_maxshift/pixscl,ftol=1e-8,   $
            maxiter=20L, display=subphot_display, snrlo=subphot_snrlo, snrhi=subphot_snrhi, $
            snrshift=subphot_snrshift, libnative=native, noaper=noaper, linear=linear, $
            silent=silent, bg=subphot_bg, savefits=subphot_savefits, $
             maskhi=subphot_maskhi, nsigma=subphot_nsigma,  $
            photbin=floor(subphot_photbin/pixscl), himg=himg, sys_err = subphot_sys_err ;, stopme=stopme 
            ; @@@  extra doesnt work here: complains about Ambiguous keyword abbreviation: PHOT.toc
toc
      end ; end object
    end  ; end catalog
   end ; end filter

end

pro dophot_gencat, param=param, total=total, latest=latest, _extra=extra
   readparam, param, default='phot.param', _extra=extra

  nphot=n_elements(phot)
  readcol, file_basename(det,'.fits')+'.xy', x, y, idsex, id_orig, f='f,f,l,a', /silent
  if not keyword_set(gencat_totcor) then gencat_totcor = 1.0    ; additional fixed aperture correction to apply to fluxes
  if not keyword_set(gencat_errcor) then gencat_errcor = 1.0
  if not keyword_set(gencat_header) then gencat_header = ['det',str(indgen(nphot)+1)]

  if keyword_set(latest) then begin
    file_delete, 'latest',/allow
    file_link, outdir, 'latest'
  end

  zpscl = 10.0^( (zpcat - zpab)/2.5 )*gencat_totcor
  iok = where(not (id_orig.matches('-1') or id_orig.matches('star*')),nok)  
  print, 'photometry on ',det
  print, 'zpcat:',zpcat
  print, 'zpab: ',zpab
  print, 'scaling:', zpscl

  nfield = 15
  fxy = fltarr(nphot*nfield+3,n_elements(x))-99.0
  if file_test(det) then begin
    detimg = readfits(det,/silent, hdet)    ; flux measured on detection image. perhaps not all that useful.
    faper, detimg, x[iok]-1, y[iok]-1, subphot_raper/pixscale(hdet), ffd
    delvarx, detimg
  end else ffd = fltarr(1,nok)
  fxy[0:1,iok] = [1#x[iok],1#y[iok]]
  fxy[2,iok] = ffd*zpscl[0]

  ocat = cat
  for i=0,n_elements(ocat)-1 do begin
    print, 'processing ',ocat[i]
    delvarx, aper_corr
    rcat, ocat[i], /unpack
    cid=id
    ff =  fltarr(nphot*nfield+3,n_elements(cid))-99.0
      
    ; @@@ for now, to match Bouwens catalog, scale back fluxes by the aper_corr
    ; for bright sources RB flux_F125W * aper_corr is a good match to F125W_aper (2" diam)
    ; but fainter than h=26.5 scatter becomes very large. need to investigate and see if this is ok.
    if n_elements(aper_corr) eq 0 then aper_corr=fltarr(1,n_elements(ra))
    invtotcorcat = 10.0^(aper_corr*0.4)  ; inverse ap cor < 1 to go from "total" mag, to phot fluxes from RB

    nmax = n_elements(cid)
   for q=0,nphot-1 do begin ; for each photometry image
     
      wht = fltarr(x.length) 
      if file_test(photw[q]) then begin  ; add median normalized weight from weightmap
          print, 'adding weights -> ',imdir+'/'+file_basename(photw[q])
          wimg =  readfits(imdir+'/'+file_basename(photw[q]),/silent,nanval=0.0)
          faper, wimg, x[iok]-1, y[iok]-1, subphot_raper/pixscale(hdet), fw, /mean,/nan
          wht[iok] = fw 
      end else wht[iok] = 1.0

      for j=0,nmax-1 do begin   ; for each object in catalog
         k = where(id_orig eq cid[j], nk,/null)
         statusline, 'processing '+str(cid[j])
         if nk eq 0 then begin
           print, cid[j], ' not found (likely multiple instances of object in catalog)'
           continue
         end
        ; add detection flux and xy pos
        if q eq 0 then ff[0:2,j] = fxy[0:2,k]
   
         objdir =  outdir+'/'+file_basename(phot[q],'.fits')+'/'+id_orig[k]
         if not file_test(objdir) then begin
           print, objdir[0], '   not found!!!'
           continue
         end
        
        subphot_cat = (outdir+'/'+file_basename(phot[q],'.fits')+'/'+id_orig[k]+'/'+str(idsex[k]))[0]
        if file_test(subphot_cat+'_phot.cat') then rcat, subphot_cat+'_phot.cat', /unpack
        if file_test(subphot_cat+'_model.cat') then rcat, subphot_cat+'_model.cat', /unpack

        if ~finite(totcor1) then totcor1=1.0
        if ~finite(apcor1) then apcor1=1.0

        contam = 0 > fnn1/float(forg1) < 1
        fnn_snr = fnn1/e1        
        if finite(f1) eq 0 then begin 
          use = 0
          ff[3+(q*nfield)-1] = 0
          continue 
        end else begin 
;          bad = ( (contam gt 0.3 and chi_ann gt 0.1 and abs(bg_ann) gt 0.1) or  $  
;                (contam gt 0.8 and fnn_snr gt 60) or chi_ann gt 0.3 or chi_red_half gt 0.2 or chi_red gt 1000.0) $
;                and not (chi_red lt 2 and contam lt 0.35) and not (contam gt 0.5 and fnn_snr lt 5)
          bad = ( (contam gt 0.3 and chi_ann gt 0.2) or  $
                (contam gt 0.8 and fnn_snr gt 60) or chi_ann gt 0.4 or chi_red_half gt 0.2 or chi_red gt 1000.0) $
                and 1-(chi_red lt 2 and contam lt 0.35) and 1-(contam gt 0.5 and fnn_snr lt 5) ; or contam eq 0
          use = 1 - bad 
          if keyword_set(showme) and bad and strmatch(phot[q],'*CH1*') then begin   ;  if showme eq idsex[k] or showme eq 1 then 
             print, contam, chi_ann, fnn_snr,  strmatch(phot[q],'*CH1*'), use,  f='(5f8.3)'
             print, subphot_cat, ' use '
             print, 'f1,e1', f1,e1,f1/e1, fcor1/ecor1, '  neighbor:', fnn1/e1, fnn1/ecor1, f='(a,2f8.3)'
             print, 'contam, fnn1, fnn_snr, bg_ann', contam, fnn1, fnn_snr, bg_ann, f='(a,4f8.3)'
             print, 'chi_red, chi_red_half, chi_ann', chi_red, chi_red_half, chi_ann, f='(a,4f8.3)'
             print
             print, 'bad = (contam gt 0.3 and chi_ann gt 0.2) or ', (contam gt 0.3 and chi_ann gt 0.2)
             print, ' (contam gt 0.8 and fnn_snr gt 60)', (contam gt 0.8 and fnn_snr gt 60)
             print, ' or chi_ann gt 0.4 ', chi_ann gt 0.4
             print, 'or chi_red_half gt 0.2 ',chi_red_half gt 0.2
             print, 'or chi_red gt 1000.0) ', chi_red gt 1000.0
             print, ' and 1-(chi_red lt 2 and contam lt 0.35) and 1-(contam gt 0.5 and fnn_snr lt 5) ',  1 - (chi_red lt 2 and contam lt 0.35) and 1 - (contam gt 0.5 and fnn_snr lt 5)
             print, 'use = ',use
             spawn, 'open '+subphot_cat+'.png'
             stop
          end
        end
; Note again: subphot provides PSF corrected fluxes (named incorrectly as "apcor1") in a fixed aperture.
; One way to match the Bouwens catalog fluxes is to 1) correct to total * (totcor1/apcor1) and
; then 2) inverse correct to catalog aperture fluxes using the aper_corr column
; so invapcor[j]*(totcor1/apcor1)
        ff[3+q*nfield:3+(q+1)*nfield-1,j] = [[fcor1,ecor1*gencat_errcor,fnn1]*zpscl[q+1]*invtotcorcat[j]*(totcor1/apcor1)[0], shx, shy, chi_red, chi_red_half, chi_ann, bg_ann, contam, fnn_snr, apcor1, totcor1, wht[k], use]        
      end ; filter
    end ; objects

; @@@ note explicit using apcor1, totcor1, will break multiple aperture measurements
    ss=!null &  for z=1, nphot do ss = [ss,['flux_F','eflux_F','flux_contam','shx','shy', 'chi','chi_half','chi_ann','rbg_ann','contam', 'snr_nn', 'psfcor','totcor', 'wht','use']+gencat_header[z].compress()]
    ofile = outdir+'/'+file_basename(ocat[i],'.cat')+'_mophongo_full_output_v'+str(version)+'.cat'
    print, 'writing to '+ofile
    openw, lun, ofile, /get_lun
    hdr = ' id_irac xdet ydet flux_F'+gencat_header[0].compress()+' '+strjoin(ss,' ')
    printf, lun, '# '+hdr
    for j=0,nmax-1 do printf, lun, cid[j], ff[*,j], format='(a,2f,g11.4,'+str(3+nphot*nfield)+'g12.4)'
    close, lun & free_lun, lun

    rcat, ocat[i], c
    ss=!null & for z=1,nphot do ss = [ss,['flux_F','eflux_F','use','wht']+gencat_header[z].compress()]
    rcat, ofile, n, col=['id_irac','xdet','ydet','flux_F'+gencat_header[0].compress(),ss], /crop
    unpack, n 

; apply aperture correction to all fluxes and errors, to simplify later use of catalog
    iuse = where(n.hdr.contains('use'))
    n.data[iuse,*] = (string(long(n.data[iuse,*]) )).compress()
    iwht = where(n.hdr.contains('wht'))
    ww =  n.data[iwht,*]
    ww[where(ww eq 0,/null)] = !values.f_nan
    if ww.ndim eq 2 then ww /= cmreplicate(median(ww,dim=2),(ww.dim)[1])
    ww[where(finite(ww) eq 0,/null)] = 0
    n.data[iwht,*] = (string(ww)).compress()

; multiply all fluxes to total
    pcat, c, n, cn
    
    iff = where(strmatch(cn.hdr,'flux_*') or strmatch(cn.hdr,'eflux_*'),/null)
    cn.data[iff,*] /=  ( replicate(1,iff.length) # invtotcorcat) 
    
    prcat, cn, outdir+'/'+file_basename(ocat[i],'.cat')+'_v'+str(version)+'.cat'  
    file_copy,  param, outdir, /overwrite
 
    print, '# percentage sources use=1'
    print, '# ',gencat_header[1:*].join(' ')
    ok = float(cn.data[where(cn.hdr.matches('use')),*])
    ok[where(ok eq -99 or finite(ok) eq 0)] = 0.0
    if ok.ndim eq 2 then print, total(ok,2)/nmax
    
    end
end

pro dophot, param=param
  dophot_shiftmap, param=param
  dophot_reg, param=param
  dophot_sex, param=param
  dophot_prepcat, param=param
  dophot_star, param=param
  dophot_mkpsf, param=param

;  dophot_mkkern, param=param
  dophot_mkkern_aor, param=param

  dophot_doall, param=param
  dophot_gencat, param=param
   dophot_doall, param=param, /force, single='GSWB-2496755030',/display
end



