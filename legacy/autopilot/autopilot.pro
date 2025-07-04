
; @@@ expand SEXTRACTOR style segmentation through dilation!!!!
; until the average SNR in co-added pixels < XXX
;
; @@@ reorganize: make into separate objects:
;  - PSF maker (per band)
;  - Object detector, deblender, detection kernel
;  - Background substraction
;  - shiftmapper + registration
;  - Aper + fitting Photometry: filters, kernels, convolutions, book keeping, mophongo
;
;  check @@ http://www.astro.yale.edu/whitaker/contents/Reduction.html
;

compile_opt idl2, logical_predicate

; expects a noise equalized detection image
; check dao-find
; also check profile k-nn fitter
;(1) The sharpness statistic compares the central pixel to the mean of
;       the surrounding pixels.   If this difference is greater than the
;       originally estimated height of the Gaussian or less than 0.2 the height of the
;	Gaussian (for the default values of SHARPLIM) then the star will be
;	rejected.
pro autopilot_star, detimg, param, fwhm=gfwhm, foutname=foutname, stars=stars, stopme=stopme, _extra=extra
  if not keyword_set(foutname) then fout='' else fout = param.dir_work+'/'+file_basename(foutname,'.fits')+'_'

   ; get brightest sources and half light radii
   flux_frac = 0.5
   obj = detect_objects(detimg, minarea=5, detect_thresh=5, flux_frac=flux_frac, /kron, kron_factor = 1.5, _extra=extra)

   ; limit to highest peak SNR highest concentration sources
   lp = alog10(obj.peak)
   mlim = percentiles(lp,value=1-param.star_percentile)
   plim = percentiles(obj.iso_area/obj.peak, val=param.star_percentile)
   gausscor = gauss_cvf(0.16)*2.35/gauss_cvf((1-flux_frac)/2.0)    ; from rhalf to fwhm
   is = where(lp gt mlim and obj.iso_area/obj.peak lt plim and $
               obj.ellipticity lt param.star_ellipticity $
               and (param.star_maxfwhm ? param.star_maxfwhm : 1e30)/gausscor)

; when narrowed down candidate is is available, should derive detailed growth curves
; also, 'recenter' when making curves -> check inner profile before /after
; does that also matter for PSF matching ?

   ; stars are all sources within some fraction of the mode fwhm
   fwhm = obj[is].rhalf*gausscor
   s=mad(fwhm,m=m)
   plothist, fwhm, xh,yh, bin=s/3, /noplot,/nan
   !null=max(yh,imax)
   fwhm_mode = xh[imax]
   istar = is[where(abs(fwhm - fwhm_mode)/fwhm_mode lt param.star_deltafwhm)]

   logger, 'found '+istar.length.s()+' stars, writing to '+fout+'stars.cat', /info
   xyad, headfits(fout+'sci.fits'), obj[istar].x, obj[istar].y, aa, dd
   ; always write fits coordinates to file
   forprint, obj[istar].x+1, obj[istar].y, obj[istar].id+1, aa, dd, textout=fout+'stars.cat', /silent, comment='# x  y  id ra dec '

   ds9 = replicate({ds9reg},istar.length)
   ds9.x = obj[istar].x
   ds9.y = obj[istar].y
   ds9.text = obj[istar].id
   write_ds9reg, fout+'stars.reg', ds9, 'IMAGE'

   ; fit gaussian psf to stack of stamps
   getstamps, detimg, obj[istar].x, obj[istar].y, stars, tsz=5/pixscale(fout+'sci.fits'), /recenter, /normalize, nanval=0. ; , /check
   star_psf = median(stars,dim=3)
   star_psf[where(finite(star_psf) eq 0,/null)] = 0
   stars[where(finite(stars) eq 0 ,/null)]=0
   star_gauss_fit = gauss2dfit(star_psf, gg)
   gfwhm  = sqrt(gg[2:3].product())*2.35

   ; write gaussian kernel to file
   logger, 'fitting gaussian to stack: fwhm = ',gfwhm,', writing kernel to '+fout+'detect_kern.conv', /info
   p = psf_gaussian(fwhm=gfwhm,npix=oddsize(2*gfwhm),/norm)
   writefits, fout+'detect_kern.fits', p
   openw, lun, fout+'detect_kern.conv', /get_lun
   printf, lun, 'CONV NORM'+string(10B)+'# gaussian FWHM = '+nsig(3.68,3)
   for i=0,(p.dim)[1]-1 do printf, lun, p[*,i], format='('+(p.dim)[0].s()+'f8.4)'
   close, lun & free_lun, lun
   !null = check_math(32)

   cgps_open, fout+'stars.pdf', xsize=10,ysize=9, yoff=-1,/quiet
   plotinit, charsize=1.2
   !p.multi=[0,2,2]
   cgplot, lp, obj.rhalf,psym=16,xr=[0,6],yr=[0,6],syms=0.3,col='black',xtit='log(peak_snr)',ytit='r_half'
   cgplot, lp[is], (obj.rhalf)[is], psym=16, col='red',/overplot,symsize=0.5
   cgplot, lp[istar], (obj.rhalf)[istar], psym=16, col='green',/overplot,symsize=0.5

   cgplot, obj.iso_area, obj.iso_area/obj.peak,psym=16,/xlog,/ylog,symsize=0.3,xtit='iso area',ytit='peak_snr/iso_area'
   cgplot, obj[is].iso_area, obj[is].iso_area/obj[is].peak,psym=16, /overplot, col='red',symsize=0.5
   cgplot, obj[istar].iso_area,  obj[istar].iso_area/obj[istar].peak, psym=16, /overplot, col='green',symsize=0.5

   cgplot, obj.rhalf, obj.ellipticity, psym=16,xr=[0,6],yr=[0,1],syms=0.3,xtit='rhalf',ytit='ellipticity'
   cgplot, obj[is].rhalf, obj[is].ellipticity, psym=16, /overplot, col='red',symsize=0.5
   cgplot, obj[istar].rhalf, obj[istar].ellipticity, psym=16, col='green',/overplot,symsize=0.5
   erase
   !p.multi=[0,8,6]
   ii = reverse(sort(lp[istar]))
   s = 10*mad(stars)
   loadct,0
   for i=0,ii.length-1 do cgimage, bytscl(stars[*,*,ii[i]],min=-s,max=s),/keep
   cgps_close

   stars = obj[istar]
   save, stars, filename=fout+'stars.sav'
end

; sanitize image
; smooth weightmap to fill holes
; normalize to median weight
; set image to zero where weight < minweight, and to maxweight if weight > maxweight
; scan for weird values, set weight and value to 0
pro autopilot_sanitize, image, weight, bin, par, showme=showme
   if keyword_set(showme) then begin
      !p.multi=[0,2,1]
      plothist, alog10(image[where(image gt 0.1 and finite(image) ne 0)]), xh,yh,/nan,/auto,col='red',xrange=[0.1,10],/ylog
      fit = exponential_fit(xh,yh,guess=[max(yh),-2.0],sigma=efit)
   end

  ; since we are dividing by weightmap. smooth it (and place limits on dynamic range ?)
   iw0 = where(weight eq 0 or image eq 0 or finite(image) eq 0,/null,complement=iw)
   weight = smooth(temporary(weight), keyword_set(bin) ? (bin>3) : 3 )   ; important... smooth weightmap to fill holes
   if n_elements(weight) gt 1 then wnorm = median(weight[iw]) else wnorm=1.0
   weight /= wnorm

   logger, 'capping minimum maximum normalized weight to', par.minweight, par.maxweight

   iwlo = where(weight lt par.minweight,/null)
   weight[iwlo] = 0.0
   image[iwlo] = 0.0
   weight[where(weight gt par.maxweight,/null)] = par.maxweight
   weight[iw0] = 0.0
   image[iw0] = 0.0

   ; normalize image, get median sigma
   img_norm = image*sqrt(weight)
   sig = mad(img_norm[where(img_norm ne 0)],med=med,/sample)

   ; fit exponential to object pixel distribution, extrapolate to where you would expect < 0.1 pixels
   ; that will be maximum allowed value
   plothist, alog10(img_norm[where(img_norm gt med+10*sig)]), xh, yh, /nan,/auto,col='blue', xrange=[alog10(med+10*sig),10],/ylog, /noplot
   fithi = linfit(xh++, alog10(yh>1))    ; add 1 to log x to extrapolate fit
   limhi = interpol(xh,10^(fithi[0]+xh*fithi[1]), 0.1)
   logger, 'minimum, maximum allowed value in normalized image', [-4*sig,10^limhi]

   ihi = where(img_norm gt 10^limhi,/null,nhi)
   ilo = where(img_norm lt -5*sig,/null,nlo)
   logger, 'flagging ',nlo,' pixels too low and ',nhi,'pixels too high'
   image[ihi] = 0
   weight[ihi] = 0
   image[ilo] = 0
   weight[ilo] = 0

   if keyword_set(showme) then begin
      plothist, alog10(img_norm[where(img_norm gt med+10*sig)]), xh, yh, /nan,/auto,col='blue', xrange=[alog10(med+10*sig),10],/ylog
      cgplot, xh+1,10^(fithi[0]+(xh+1)*fithi[1]),linest=0,col='black',thick=3,/overplot
      vline, limhi, minmax(yh)>0.1, thick=3, linest=2
   end
end

; add SED weights
pro autopilot_prep_detection, par, _extra=extra
    ; check if detection image exist, if so read it, if not regenerate
   report_memory

   fdecomp, par.detect_image, !null, dir, fname
   fname = fname.replace('_sci','')
   wdir = par.dir_work+'/'

    for i=0, n_elements(par.detect_image)-1 do if file_test(wdir+fname[i]+'_norm.fits') then $
       logger, 'already analyzed ', par.detect_image[i], /info else begin

          logger,'reading '+par.detect_image[i]+' ',/info
          img  = readfits(par.detect_image[i], h,/silent)
          wht  = readfits(par.detect_weight[i], /silent)

          bin = ceil(par.phot_minradius/pixscale(h)) > 3
          logger,'detection image size '+(img.length*4/1e6).s()+' MB',/debug

          autopilot_sanitize, img, wht, bin, par

          img = skysub(temporary(img), wht, rms=rmsimg, sky=bgimg,  bin=bin, _extra=extra,/verbose)
          logger, 'writing temporary files to '+par.dir_work, /info
          writefits, wdir+fname[i]+'_wht.fits', temporary(wht), h
          writefits, wdir+fname[i]+'_bg.fits', temporary(bgimg), h
          writefits, wdir+fname[i]+'_sci.fits', img, h
          writefits, wdir+fname[i]+'_rms.fits', rmsimg, h
          nimg =  img/temporary(rmsimg)
          nimg[where(finite(nimg) eq 0 or temporary(img) eq 0,/null)] = 0.0

          autopilot_star, nimg, par, fwhm=fwhm, fout=fname[i], stars=stars, _extra=extra
          writefits, wdir+fname[i]+'_norm.fits', temporary(nimg), h
          report_memory
      end
end

function autopilot_filter, fimg, filter
; @@@  fix naming weirdness, by providing output name
   froot = fimg.replace('_sci','')
   fkern = froot.replace('.fits','_detect_kern.fits')
   fconv = froot.replace('.fits','_detect_conv.fits')

   if file_test(fconv) then begin
     logger, 'found existing '+fconv, /info
     return, readfits(fconv, /silent, nanv=0.0)
   end

   img = readfits(fimg, h, /silent, nanv=0.0)
   if filter eq 0 then return, img

   if isa(filter, /number) then kern = psf_gaussian(fwhm=filter+0.0,npix=oddsize(2*fwhm),/norm,/silent) $
   else if file_test(fkern) then kern=readfits(fkern,/silent) $
   else message, string(filter)+' is not a valid FWHM and '+fkern+' not an existing file'

   logger, 'convolving with ', (isa(filter) ? 'gaussian FWHM = '+string(filter) : fkern), /info
   iw0 = where(img eq 0, /null)
   img = fconvolve(temporary(img), kern)
   img[iw0] = 0

   logger, 'writing to '+fconv, /info
   sxaddpar, h, 'kernel', isa(filter) ? filter : fkern
   writefits, fconv, img, h

   report_memory
   return, img
 end

; @@@ whenever a detection kernel is changed all fitting photometry
; @@@ bands need their kernels updated
pro autopilot_make_detection, par, _extra=extra

   fdecomp, par.detect_image, !null, dir, fname
   fname = fname.replace('_sci','')
   fdet = par.dir_work+fname[0]
   foreach fd, par.detect_band do fdet = fdet.replace('_'+fd,'')

   fdet += '_'+par.detect_band.join('+')+'_detect_sci.fits'
   par['detect_image_weighted'] = fdet

   ; check if detection image exist, if so read it, if not regenerate
   if file_test(fdet) then logger, 'found existing '+fdet, /info else begin
      logger, 'generating ',fdet,' using method ', par.detect_method
     ;  header for output image from first image
      !null = readfits(par.detect_image[0],h,startrow=0,numrow=1,/silent)

      detimg = (detwht = 0.0)
      ; construct detection image, possibly from multiple images
      for i=0, n_elements(par.detect_image)-1 do begin
          ; img is skysub, convolved
          img = autopilot_filter(par.dir_work+fname[i]+'_sci.fits', par.detect_filter)
          wht = readfits(par.dir_work+fname[i]+'_wht.fits', /silent)
          logger,'detection image size '+(img.length*4/1e6).s()+' MB', /debug

         ; @@@  TODO add weights
         if par.detect_method eq 'variance' then begin
            simg = skysub(temporary(img),temporary(wht),rms=rmsimg,bin=3)
            ivar = 1.0/temporary(rmsimg)^2
            ivar[where(finite(ivar) eq 0)] = 0.0
            detimg += temporary(simg)*ivar
            detwht += temporary(ivar)
            report_memory
         end else if par.detect_method eq 'chi2' then begin
            img = skysub(temporary(img),temporary(wht),rms=rmsimg,bin=3)
            detimg += (img/rmsimg)^2       ; build chi2
            detimg_neg += (-img/temporary(rmsimg))^2  ; also build negative chi2 here
            detwht = 1.0
            tvs, detimg, pos=2*i+0
            tvs, detimg_inv, pos=2*i+1
            report_memory
            stop
         end else message, 'unknown detection method '
      end

      ; final detection image
      if par.detect_method eq 'variance' then begin
         detimg = detimg/detwht*sqrt(detwht)         ; detimg*sqrt(temporary(detwht))
         detimg[where(finite(detimg) eq 0)] = 0.0
         rmstot = sqrt(1.0/detwht)
         rmstot[where(finite(rmstot) eq 0)] = 0.0
      end

      report_memory
      writefits, fdet, detimg, h
      writefits, fdet.replace('sci','wht'), temporary(detwht), h
      if par.detect_method eq 'variance' then writefits,  fdet.replace('sci','rms'), temporary(rmstot), h
      if par.detect_method eq 'chi2' then writefits,  fdet.replace('sci','inv'), temporary(detimg_inv), h
   end

; do we need to do this here ? why not in make_phot ?

  ; if file_test(fdet.replace('_sci.fits','_psf')+'.sav') then $
  ;    logger, 'found existing ', fdet.replace('_sci.fits','_psf') else begin
  ;    if ~isa(par.psf_cat) then par.psf_cat = par.dir_work+fname[0]+'_stars.cat'

      ; determine PSF map on combined detection image, for aperture corrections
;      klpsf, !null, image=fdet, psf_snrlim = par.psf_snrlim[0], minstar=par.psf_minstar, $
;         display=par.psf_check, starcat=par.psf_cat, max_basis=par.psf_max_basis, $
;         tile_size = par.psf_tile_size, peakthresh=par.psf_peakthresh, blender=par.psf_blender, $
;         maxshift=par.psf_maxshift, magbuf=par.psf_magbuf, outname=fdet.replace('_sci.fits','_psf'), $
;         average = par.psf_combine.matches('average',/fold), _extra=extra
;   end

end


pro autopilot_detect_objects, par, _extra=extra

   fdet = par.detect_image_weighted
   if file_test((fobj=fdet.replace('detect_sci.fits','objects.sav'))) then begin
     logger, 'found existing ',fobj
     return
   end

   detimg = readfits(fdet,h,/silent)
   thresh = float(sxpar(h,'d_thresh'))

   if file_test((fdetinv = fdet.replace('sci','inv'))) then detneg = readfits(fdetinv,h,/silent)

   if ~ isa(par.detect_thresh) then begin
      if float(thresh) eq 0 then begin
        logger, 'Auto detection, setting threshold so spurious fraction is', par.detect_spurious, /info
        par.detect_thresh = autopilot_threshold(detimg, minarea=par.detect_minarea, $
                                        spurious=par.detect_spurious, $
                                        detneg=detimg_neg, showme=showme)
      end else param.detect_thresh = thresh
   end

   logger, 'Using detection threshold', par.detect_thresh
   modfits, fdet, 0, h ; only update hdr
   sxaddpar, h, 'd_thresh', par.detect_thresh

   obj = detect_objects(detimg, detect_thresh=par.detect_thresh, detect_minarea=par.detect_minarea, $
         deblend_mincont=par.deblend_mincont, deblend_nthresh=par.deblend_nthresh, /deblend, seg=seg,$
         reverse_indices=reverse_indices)

   spurious = detect_objects(-detimg, detect_thresh=par.detect_thresh, detect_minarea=par.detect_minarea, $
         deblend_mincont=par.deblend_mincont, deblend_nthresh=par.deblend_nthresh, /deblend)

   hu = h
   sxaddpar, hu, 'BITPIX', 32
   writefits, fdet.replace('sci','seg'), seg, hu
   writefits, fdet.replace('sci','no-objects'), detimg*(seg eq 0), h
   logger, 'number of detected sources ',obj.length, ', number of detections on inverse image ', spurious.length, /info

   save, obj, reverse_indices, filename=fobj
   save, spurious, filename=fdet.replace('detect_sci.fits','spurious.sav')
end


; @@@ note interpolation edge effects
pro autopilot_register, fimage, fref, foutput, fweight=fweight, snrlim=snrlim, wthresh=wthresh, _extra=extra

   logger, 'registering '+fimage, /info

  ;  output image undersamples inputimage, first rebin
  ; @@@ use frebin
   if (scl = pixscale(fimage)) lt 0.8*(refscl = pixscale(headfits(fref))) then begin
      bin = round(1.2 * refscl / scl)
      logger, 'detection pixel scale ', refscl, ' photometry pixel scale ', scl, ', binning by x ', bin
      img  = readfits(fimage, h,/silent)
      wht  = readfits(fweight, hw,  /silent)
      hrebin, img, h, outsize=img.dim/2
      hrebin, wht, hw, outsize=wht.dim/2
      writefits, fimage, temporary(img), h
      writefits, fweight, temporary(wht), hw
   end

; @@@ change from spline to gridddata and open up interface
   shiftmap, fimage, fref, fimage.replace('.fits','_shift'), fweight=fweight, fmap=fmap, snrlim=snrlim
; @@@ rename reg_spline
   reg_spline, fimage, fref, fimage, fweight=fweight, fmap=fimage.replace('.fits','_shift.sav')

   report_memory
end


; register
pro autopilot_register_phot, par, _extra=extra

   fdecomp, par.detect_image[0], !null, dir, fdet
   fdecomp, par.phot_image, !null, dir, fname
   froot = par.dir_work+fname.replace('_sci','')

   for i=0, n_elements(par.phot_image)-1 do begin

      ; skip if exist and registered to curent detection image
      ; @@@@ just check for the rms image... as that one is the last written
      if file_test((fout = froot[i]+'_sci.fits')) eq 0 then skipreg = 0 else $
        skipreg = par.detect_image_weighted.equals(string(sxpar(headfits(fout),'register')))

     if keyword_set(skipreg) then $
         logger, 'found existing '+fout+' registered to '+par.detect_image_weighted $
      else begin
           ; sanitize images -> necessary before register ? @@@ register crashes if not
         img  = readfits(par.phot_image[i], horig,/silent)
         wht  = readfits(par.phot_weight[i], hw,  /silent)
         autopilot_sanitize, img, wht, bin, par
         writefits, froot[i]+'_sci.fits', temporary(img), horig
         writefits, froot[i]+'_wht.fits', temporary(wht), hw
 ;        pixscale_orig = pixscale(horig)

; @@@ cut the superfluous read/write
; @@@ we should have choice of fit. in most cases a 1-dim transformation suffices
; @@@ don't read images in register....
; fix interpolation edge effects in register
      autopilot_register, froot[i]+'_sci.fits', par.detect_image_weighted, $
                          froot[i]+'_sci.fits', fweight = froot[i]+'_wht.fits', $
                          wthresh=par.minweight, snrlim = par.register_snrlim

         ; @@@@ first fix interpolation edge crap due to register
         ; @@@ fix the anoyying read-writes
         wht = readfits(froot[i]+'_wht.fits',/sil,nan=0.0)
         img = readfits(froot[i]+'_sci.fits',h,/sil,nan=0.0)
         autopilot_sanitize, img, wht, bin, par
         writefits, froot[i]+'_wht.fits', temporary(wht), h
         writefits, froot[i]+'_sci.fits', temporary(img), h

     end ; register
   end ; for each image
end

; measure noise, PSF
pro autopilot_map_psf, par, _extra=extra

   fdecomp, par.detect_image[0], !null, dir, fdet
   fdecomp, par.phot_image, !null, dir, fname
 ;  froot = par.dir_work+fname.replace('_sci','')
   photname = [par.phot_image,  par.detect_image_weighted.replace('_sci.fits','')]
   froot = [par.dir_work + fname.replace('_sci',''), par.detect_image_weighted.replace('_sci.fits','')]

   for i=0, n_elements(photname)-1 do begin
      logger,' processing  '+photname[i]+' ',/info

      ; skip if already processed
      h = headfits(froot[i]+'_sci.fits')
      if isa(sxpar(h,'MP_PTYPE'),/string) and file_test(froot[i]+'_psf.sav') then begin
        logger, 'found existing PSFs for '+froot[i]
        continue
      end

      ; if psf.fits given, copy over and use it
      ;  @@@@ need to finalize psf.fits format.
      if file_test(photname[i].replace('sci','psf')) then begin

         file_copy, photname[i].replace('sci','psf'), froot[i]+'_psf.fits', /overwrite, /verbose
         avgpsf =  readfits(froot[i]+'_psf.fits', hpsf, exten=1,/silent)
         gc = growthcurve(avgpsf, rh=rhalf)

         ; add this to header of current working copy
         sxaddpar, h, 'MP_RHALF', rhalf*pixscale(hpsf), 'half light radius in arcseconds'

      ; derive empirical PSF from image... klpsf does the deblending and kl component stuff
      end else if ~file_test(froot[i]+'_psf.sav') then begin

         ; if no starlist provided, use one derived from detection image
         if ~isa(par.psf_cat) then par.psf_cat = par.dir_work+fdet.replace('_sci','_stars.cat')

         ; klpsf will store MPH_RHMIN keyword in _sci.fits
         klpsf, !null, image=froot[i]+'_sci.fits', psf_snrlim = par.psf_snrlim[0], $
             minstar=par.psf_minstar, $
             display=par.psf_check, starcat=par.psf_cat, max_basis=par.psf_max_basis, $
             tile_size = par.psf_tile_size, peakthresh=par.psf_peakthresh, blender=par.psf_blender,$
             maxshift=par.psf_maxshift, magbuf=par.psf_magbuf, outname=froot[i]+'_psf', $
             average = par.psf_combine.matches('average',/fold), $
             half_light_radius = half_light_radius, _extra=extra

         ; add this to header of current working copy
         sxaddpar, h, 'MP_RHALF', half_light_radius*pixscale(froot[i]+'_sci.fits'), $
            'half light radius in arcseconds'
      end

      ; @@@ use par.phot_rhalf_fit to set the minimum background size
      ; @@@ logic is that PSF matched bands will be smoothed to this size
      ; decide if we are going to do fitting or psf matched aperture
      if sxpar(h,'MP_RHALF') gt par.phot_rhalf_fit then $
          sxaddpar, h, 'MP_PTYPE', 'FITTING' $
      else sxaddpar, h, 'MP_PTYPE',  'APERTURE'

      modfits, froot[i]+'_sci.fits', 0, h
    end
end

; global background subtract, measure noise:
; uses half light radius to set background aperture if available
pro autopilot_subtract_background, par, _extra=extra

   fdecomp, par.detect_image[0], !null, dir, fdet
   fdecomp, par.phot_image, !null, dir, fname
   froot = par.dir_work+fname.replace('_sci','')

   for i=0, n_elements(par.phot_image)-1 do $
      if sxpar(headfits(froot[i]+'_sci.fits'),'MP_WBG') le 0 then begin
         logger, 'subtracting background of ', froot[i]+'_sci.fits'
         img = readfits(froot[i]+'_sci.fits',h,/sil,nan=0.0)
         wht = readfits(froot[i]+'_wht.fits',/sil,nan=0.0)

         ; @@@ should skip this for IRAC, or at least have very different binning?
         ; or bin in the original pixel scale?
         ; something should inform the background scale... tie it it half light radius PSF ?
         ; sky subtract and measure rms
         pixscale_orig = pixscale(par.phot_image[i])
         pixscale_new = pixscale(h)

         ; make sure binning is always the photometric radius or 3 x original pixels
         ; whichever is biggest
         bin = ceil(par.phot_minradius/pixscale_new) > (3.0*pixscale_orig/pixscale_new)

         ; @@@@ now do background to XX * halflight radius
         rhalf_min =  sxpar(h,'MP_RHALF')  >  par.phot_rhalf_fit

         bgwidth = par.phot_background_factor * rhalf_min
         logger, 'setting background subtraction aperture diameter to: ',  $
                 par.phot_background_factor  , '*', rhalf_min, '=', bgwidth
         sxaddpar, h, 'MP_WBG',  bgwidth, 'background subtraction diameter in arsec'

; @@@ need to write background diagnostic image
         img = skysub(temporary(img), wht, bin=bin, rmsimage=rmsimage, bgwidth=bgwidth/pixscale_new, _extra=extra)
         writefits, froot[i]+'_wht.fits', temporary(wht), h
         writefits, froot[i]+'_sci.fits', temporary(img), h
         writefits, froot[i]+'_rms.fits', temporary(rmsimage), h

      report_memory
    end
end

; compute theoretical target PSF for PSF-matched photometry
; target PSF is a moffat, fit to the joint slowed growthcurve at each radius
; guarantees easy to derive convolution kernels to smallest infinite SNR reference PSF
pro autopilot_target_psf, par, _extra=extra

   ftarget = par.detect_image_weighted.replace('detect_sci','target_psf')
   if file_test(ftarget) then begin
      logger, 'found existing target PSF:', ftarget, /info
      return
   end

   ; include detection image in list to perform PSF matching
   fdecomp, par.phot_image, !null, dir, fname
;   froot = [par.dir_work + fname.replace('_sci',''), par.detect_image_weighted.replace('_sci.fits','')]
   froot = par.dir_work + fname.replace('_sci','')

   ; form target PSF from the combined PSFs with MPH_TYPE = APER
   ; construct composite curve of slowest growth, fit moffat to that
   logger, 'generating target psf ',ftarget, /info
   ; @@@ should be able to deal with tile sizes and binning / sampling
   ; @@@ fix it in mophongo... so that just to request the right sampling
   ;  we are updating the target_psf, only consider 'APERTURE' types and delete their kernels
   for i=0, n_elements(fname)-1 do begin
       if (sxpar(headfits(froot[i]+'_sci.fits'),'MP_PTYPE')).matches('APERTURE',/fold) then $
        file_delete, froot[i]+'_kern.sav', /allow else continue
        psfi = Mophongo.getpsf(filename=froot[i]+'_psf.sav')
        max_psfi = max(psfi,dim=3)
        max_psfi /= total(max_psfi)
       if isa(psf) then psf = [[[psf]],[[max_psfi]]] else psf = max_psfi
   end

   ; fit a slowest curve of growth, by penalizing negative residual x 10
   logger, 'fitting slowest curve of growth',/info
   gc = growthcurve(psf,raper=raper)
   gcmin = min(gc,dim=2)
   par_moffat = mpfit('autopilot_moffat', [2.,2], functargs={r:raper, gc:gcmin, w:10.0},/quiet)
   ; add 2.0 pixel FWHM in quadrature so that kernel is always well sampled
   target_psf = psf_moffat(npix=(psf.dim)[0],fwhm=sqrt(par_moffat[0]^2+2.0^2), beta=par_moffat[1], /norm)
   gctarget = growthcurve(target_psf,raper=raper,rhalf=rhalft)

   ; this is essentially an optimization:
   ; broaden target psf until windowed FFT well behaved for all bands
   pu = python.import('photutils')
   for i=0, n_elements(fname)-1 do begin
       h=headfits(froot[i]+'_sci.fits')
      if ~(sxpar(h,'MP_PTYPE')).matches('APERTURE',/fold) then continue
      psfi = Mophongo.getpsf(sxpar(h,'NAXIS1'), sxpar(h,'NAXIS2'), filename=froot[i]+'_psf.sav')
      win=pu.TukeyWindow(alpha=0.3)
      kern = pu.create_matching_kernel(psfi, target_psf, window=win)
      logger, min(kern)/stddev(kern), rhalft

      badness = min(kern)/stddev(kern)
      while badness lt -0.15 do begin
          target_psf = convolve(target_psf,psf_gaussian(fwhm=1.0,npix=(psfi.dim)[0],/norm))
         ;         target_psf_bin = frebin(target_psf,oddsize(dim[0]/nbin),oddsize(dim[1]/nbin),/total)
          gct = growthcurve(target_psf,rhalf=rhalft)
         win=pu.SplitCosineBellWindow(alpha=0.7,beta=0.3)
          kern = pu.create_matching_kernel(psfi, target_psf, window=win)
          logger, 'smoothing (orig):', min(kern)/stddev(kern),rhalft
         ;       kern_bin = pu.create_matching_kernel(psfi_bin, target_psf_bin, window=win)
         ;       logger, 'smoothing (binned):', min(kern_bin)/stddev(kern_bin),rhalft
         ;        badness = (min(kern_bin)/stddev(kern_bin)) > (min(kern)/stddev(kern))
            badness = (min(kern)/stddev(kern))
         ;        kern = (min(kern_bin)/stddev(kern_bin)) gt (min(kern)/stddev(kern)) ?  frebin(kern_bin,dim[0],dim[1],/total) : kern
          logger, 'smoothing (best):', badness,rhalft
      end
      psfconv = convolve(psfi,kern)
      gcc = growthcurve(psfconv,raper=raper)
      if keyword_set(stopme) then begin
            erase
            tvs, target_psf-psfconv,pos=3,mm=[-1,1]*3e-4
            tvs, psfconv,pos=4,mm=[-1,1]*3e-4
            gct = growthcurve(target_psf,raper=raper,rhalf=rhalft)
            gci = growthcurve(psfi,raper=raper,rhalf=rhalfi)
            cgplot, gct/gcc, yr=[0.9,1.1],/noerase, layout=[5,1,5],charsize=2,col='red',thick=3,/xlog,xr=[0.9,max(raper)]
            hline, 1.0, linest=2,xr=[0.1,1000],thick=2
            vline, rhalft, linest=1,thick=2
            vline, rhalfi, linest=1,thick=2
            logger,'kern min/std, rhalf_target_psf: ',min(kern)/stddev(kern),rhalft
         stopkey
       end
   end ; for each band

  writefits, ftarget, target_psf

   ; if min_radius not given, set it to the half light radius of the target PSF
   if ~isa(par.phot_minradius) then par.phot_minradius = rhalf*pixscale(par.detect_image_weighted)

   cgps_open, ftarget.replace('.fits','.pdf'),  xs=11, ys=7, /landscape, /nomatch, xoff=0, yoff=0
   plotinit, [8,5]
   for i=0, (psf.dim)[2]-1 do cgimage, psf[*,*,i] - target_psf, /keep, minv=-5e-5, maxv=5e-5
   plotinit,[1,1]
   plot, raper, yrange=[0,1.1],/yst,/nodata,/xlog, xrange=[0.5,max(raper)],/xst, xtit='radius [pix]', ytit='curve of growth', title='target_psf (green)'
   for i=0,(psf.dim)[2]-1 do oplot, raper, gc[*,i], linest=i mod 4
   cgplot, raper, min(gc,dim=2),col='red',/overplot, thick=5
   cgplot, raper, gctarget, col='green', linest=2, /overplot, thick=8
   cgps_close
end
     ; work in coarsers 1.5x binned pixels... somehow that helps... increased SNR ?
      ; or is it essentially smoothing the images before deconvolution
      ; gains nothing.... looks better, but the kernel will not match the stellar core
   ;    if  badness lt -0.15 then begin
   ;      nbin = 1.5
   ;      target_psf_bin = frebin(target_psf,oddsize(dim[0]/nbin),oddsize(dim[1]/nbin),/total)
   ;      psfi_bin = frebin(psfi,oddsize(dim[0]/nbin),oddsize(dim[1]/nbin),/total)
   ;      win=pu.TukeyWindow(alpha=0.3)
   ;      kern = pu.create_matching_kernel(psfi_bin, target_psf_bin, window=win)
   ;      logger, 'binned:', min(kern)/stddev(kern), rhalft
   ;      tvs, kern ,mm=[-1,1]*5e-4,pos=2
   ;      tvs, kern/stddev(kern) lt -0.05,pos=3
   ;      tvs, convolve(psfi,kern),pos=4,mm=[-1,1]*3e-4
;  par = replicate({fixed:0,tied:''}, 8)
;  par[[4,5,6]].fixed = 1  ; position fixed, angle
;  par[3].tied = 'P[2]'    ; symmetric  width
;   est = [ 3.0e-06,  0.017, 4.0,  4.0, ((target_psf.dim)[0]-1.)/2.,  ((target_psf.dim)[1]-1.)/2., 0.0 ,2.4]
;   mfit = MPFIT2DPEAK(target_psf, p, parinfo=par, estimates=est, /moffat)
;   par = par[0:-2]
;   mfit = MPFIT2DPEAK(target_psf, p, parinfo=par, estimates=est, /gauss)
;   logger, 'Moffat fwhm,beta ',p[2],p[-1], '-> half light radius =',rhalf

; generate kernels
pro autopilot_map_kernel, par, stopme=stopme, _extra=extra
   compile_opt idl2

   ; include detection image in list to perform PSF matching
   fdecomp, par.phot_image, !null, dir, fname
   froot = [par.dir_work + fname.replace('_sci',''), par.detect_image_weighted.replace('_sci.fits','')]
   ftarget = par.detect_image_weighted.replace('detect_sci','target_psf')
   scl = pixscale(froot[0]+'_sci.fits')

   if par.kern_method.matches('max_entropy',/fold) then max_entropy=1 else $
      if par.kern_method.matches('hermite',/fold) then hermite=1 else likelihood=1

   for j=0, n_elements(froot)-1 do begin

      ; skip if kernel found at correct binning
      if file_test(froot[j]+'_kern.sav') then begin
         restore, froot[j]+'_kern.sav'
         if ~isa(bin) then bin = 0
         if par.fit_bin eq bin then logger, 'found existing kernel for '+froot[j],' at binnin ',bin, /info
         if par.fit_bin eq bin or par.fit_bin eq 0 then continue
      end

      hphot = headfits(froot[j]+'_sci.fits')
      delvarx, gx, gy, psf
      logger, 'generating kernel map '+froot[j]+'_kern.sav'

      ; @@@  need a transparent way of reading the PSFs, including irac
      ; @@@ need to sample both grids -> so that PSF variation always properly tracked
      ; @@@ assuming that the grid does... kernel is either:
      ; @@@@ yup... verified to be a problem .....
      ; - psf -> target_psf (aperture photometry )
      ; - det_psf -> psf      (fitting photometry)
      ; set half light radius limit... below that -> match, above that -> fit
      if (sxpar(hphot,'MP_PTYPE')).matches('fitting',/fold) then begin
          hdet = headfits(par.detect_image_weighted)
          getrot, hdet, pa, cdelt

          hphot = headfits(froot[j]+'_sci.fits')
          rhi = sxpar(hdet,'mp_rhalf')/cdelt[1]
          rkern = sqrt(( (sxpar(hphot,'mp_rhalf'))^2 - (sxpar(hdet,'mp_rhalf'))^2 ) > 0.0)/cdelt[1]/3600.

         ; --------------------------------------------------
         ; @@@@@@ THIS NEEDS TO CHANGE
         ; We want PCA models of all PSFs
         ; then use original star lists.
         ; - reconstruct (kriging) interpolate star lists on eachother so that lists complete
         ; - deconvolve starlists.... what kind of regularization -> can we do L1/L0 ?
         ; @@@@@ BINNING
         ; binning / smoothing -> ultimately we are smoothing / modeling at output photometric image resolution
         ; smooth to pixel binning of photometric image, or PSF phot width / 5.0

         ; check if we can downsample
         ; @@@ the kernel defines the pixel scale of the model
         ; @@@ somewhat coarser than template, but higher than phot model ? kinda random

         ; @@@ to avoid undersampling model, we will pre-convolve by smoothing with a rect the size
         ; @@@ of the binnng
         ; if par.fit_bin ge 1 then begin
         ; even binnin will work too?
           bb = [1,3,5,7]
           bin = bb[value_locate(bb,rkern/5)]  ; or kernel, phot pixel scale / 2 ?
           ; do we care about x2 or x3 binning ?
           logger, 'binnin kernel and psf by ',bin

      ;   end else bin = 1
     ; @@@ for now assume rectilinear grid in PA=0 -> will not be the case for EGS
          ; reconstruct IRAC psf on this grid
          ; interpolate (inflate only) to desired scale and pa
          ; fast means that the PSF is assumed to be well sampled: the reconstruction is done
          ; in native pixels, and only then interpolated
          ; need to keep track of 'native' pixel scale -> as it is a convolution
       ;   xyad, hdet, gx, gy, ga, gd
        ; @@@@ do frebin vs bin update crval in same way ??
        ; @@@ it shouldnt... they deal for center pixel different

       if bin gt 1 then hrebin, hdet, outsize = [sxpar(hdet,'NAXIS*')/bin]
          getrot, hdet, pa, cdelt

; need to get rid of this -> change to PCA basis...
          target_psf = Mophongo.getpsf_irac(gra, gdec, filename=froot[j]+'_psf.fits', pixscl = cdelt[1]*3600., pa=pa, /fast, overlap=hdet)

;  @@@@ there is no guarantee that sampling is fine enough here.... this all needs to change
          adxy, hdet, gra, gdec, gx, gy
          psf = Mophongo.getpsf(gx, gy, filename=par.detect_image_weighted.replace('sci.fits','psf.sav'))
          mpsf = mean(psf,dim=3,/nan)
          tsz2 = ((psf.dim)[0]-1)/2

        logger, 'fixing psfs'
        for i=0,gx.length-1 do psf[0,0,i] = autopilot_fixpsf(psf[*,*,i],wsmooth=3 )
        for i=0,gx.length-1 do target_psf[0,0,i] = autopilot_fixpsf(target_psf[*,*,i],wsmooth=3 )

        ; presmooth for binning and interpolation kernel (so effectively the derived kernel
        ; if deconvolved for them. If reconstructing at full resolution, the model need to be
        ; smoothed the same way
        if bin gt 1 then begin
            psfb = fltarr([(psf.dim)[0:1]/bin,gx.length])

            ; deconvolve for cubic or linear interpolation
            if par.fit_interp eq 'cubic' then cubic=-0.5 else interp=1

            for i=0,gx.length-1 do begin
               psm = rbin( rbin(psf[*,*,i],bin) ,bin,/expand,interp=interp,cubic=cubic)  ; @@@  WOA this works! harcoded interp=1, so use that in mophongo
               psfb[0,0,i] = rbin(psm,bin)
            end
            psf = psfb
        end

       end else begin

         psf = Mophongo.getpsf(gx, gy, filename=froot[j]+'_psf.sav')
         target_psf = readfits(ftarget,/silent)  ; can do timestamp checks here if necessary (to see if kern.sav are up to date)

        end

      pu = python.import('photutils')
      maxdev = ( rchi = fltarr(gx.length) )
      method = strarr(gx.length)
      mask = apermask((psf.dim)[0],(psf.dim)[0]/2.0-2)
      delvarx, rhh, rhl  ; keep half light radii rhh, rhl across for same band
      for i=0,n_elements(gx)-1 do begin
         psfd = float(psf[*,*,i] )

         target_psfd = float(target_psf[*,*, target_psf.ndim eq 3 ?  i : 0])  ; single/multiple
         psz = max(psfd.dim)
         tpsz = max(target_psfd.dim)
         ksz = (tpsz > psz)

         npad = 7 + (long(ksz)/2 ne ksz/2.0)

         if psz lt tpsz then psfd = extrac(psfd, (psz-tpsz)/2,  (psz-tpsz)/2, tpsz, tpsz)
         if tpsz lt psz then target_psfd = extrac(target_psfd, (tpsz-psz)/2,  (tpsz-psz)/2, psz, psz)
         ; extra padding for fft edge effects
         psfd = extrac(psfd, -npad/2, -npad/2, ksz+npad, ksz+npad)
         target_psfd = extrac(target_psfd, -npad/2, -npad/2, ksz+npad, ksz+npad)

         if i eq 0 then decon, psfd, target_psfd, kern_iter, $
                    maxiter=par.kern_maxiter, klim=par.kern_klim, rchi=rr, $
                 max_entropy=max_entropy, likelihood=likelihood, hermite=hermite, maxdev=dd, $
                 nbasis=kern_nbasis, basis=basis, nlow=nlow, verbose=verbose, method=me, $
                 display=par.kern_check, step=1.0, regulate=par.kern_regulate, dscale=dscale, $
                 rmin_chi=par.phot_minradius/scl/bin, fout = (i eq 0) ? froot[j]+'_kern.pdf' : !null, $
                 title=froot[j]

        ; win=pu.TukeyWindow(alpha=0.4)
         win=pu.SplitCosineBellWindow(alpha=0.7,beta=0.3)
         kern = pu.create_matching_kernel(psfd, target_psfd, window=win)
         method[i] = 'fft'

         if min(kern)/min(kern_iter) gt 4.0 then begin
            win=pu.SplitCosineBellWindow(alpha=0.7,beta=0.3)
            kern = pu.create_matching_kernel(psfd, target_psfd, window=win)
            logger,'ML ITER softened window: ratio(min,max,std):' ,minmax(kern)/minmax(kern_iter),stddev(kern)/stddev(kern_iter),  (min(kern)/min(kern_iter) gt 4) ? '-> kernel too noisy, doing iterative ML' : ''

            if min(kern)/min(kern_iter) gt 4.0 then begin
               decon, psfd, target_psfd, kern, $
                    maxiter=par.kern_maxiter, klim=par.kern_klim, rchi=rr, $
                 max_entropy=max_entropy, likelihood=likelihood, hermite=hermite, maxdev=dd, $
                 nbasis=kern_nbasis, basis=basis, nlow=nlow, verbose=verbose, method=me, $
                 display=0, step=1.0, regulate=par.kern_regulate, dscale=dscale, $
                 rmin_chi=par.phot_minradius/scl/bin, title=froot[j]

;                stopkey
                method[i] = me
                maxdev[i] = dd
            end
         end else logger, 'FFT  ratio(min,max) ', minmax(kern)/minmax(kern_iter)

         psfcon = fconvolve(psfd,kern)
         gc=growthcurve(target_psfd-psfcon,raper=raper)
         maxdev[i] = max(abs(gc))

          if i eq 0 then kernel = fltarr([ksz,ksz,gx.length])
          kernel[*,*,i] = kern[npad/2:npad/2 + ksz-1,npad/2:npad/2 + ksz-1]
          logger, gx[i], gy[i],  method[i], maxdev[i], /debug
         if  0  then begin
 ;         if keyword_set(stopme)  then begin
               cgerase
               !p.multi=[0,1,1]
               tvs, psfd, mm=[-1,1]*2e-4,pos=0
               tvs, kern, mm=[-1,1]*7e-5,pos=1
               tvs, psfcon, mm=[-1,1]*7e-5,pos=2
               tvs, target_psfd, mm=[-1,1]*7e-5,pos=3
               tvs, target_psfd-psfcon, pos=4, mm=[-1,1]*7e-5
               cgplot, raper, gc,yr=[-0.02,0.02],/xlog,xr=[0.9,85],col='red',/noeras,layout=[5,1,5], charsize=2
                hline,0,linest=2
               logger, i, rhp, rht, froot[j]
         ;  stopkey
          end
        end

      kernel = mophongo.pca(kernel)

; @@@  TODO no checks on regularity: if grid 2D -> regular
; if grid 1D -> irregular
; kernel interpolation is now very fast... so no need for resampling on grid, unless
; desired

      ; @@@ TODO: first resample PSF and kernel on eachothers positions
      ; @@@ then PCA decompose
      ; @@@ resample PCA coefficients on fine grid
      ; @@@ then after kernel has been generated. then interpolate kernel onto fine grid
      ; grid needs to be rectilinear?
      ; rectilinear grid in current mapping
      ; this is not very robust: e.g. if slightest curve/angle and cross pixel boundary
      ix = (round(gx)).uniq()
      iy = (round(gy)).uniq()
      if gra.ndim ne 2 and gdec.ndim ne 2 and ix.length*iy.length eq gx.length then begin
         logger, ' grid is rectilinear grid ', ix.length, iy.length
         gra = reform(gra, ix.length, iy.length)
         gdec = reform(gdec, ix.length, iy.length)
      end
      adxy,hdet,gra,gdec,gx,gy

; @@@ rename hkern earlier
      hkern = hdet
      save, gra, gdec, hkern, kernel, bin, filename=froot[j]+'_kern.sav'

      ;  could write more info       maxdev, rchi, method,
      ; @@@ the reasons to do this are to 1) facilitate easy inspection and analysis
      ; @@@ because at 2) photometry stage interpolation is ~100 x faster
      ; @@@ the rectangular grid will be fine, so can likely be randomly sampled later
      ; @@@ for correlation analyses.
      ; @@@ grid spacing will be factor*fwhm  < size(kernel) < existing avg spacing
      ; deal with fits later

  end ; for each band
end

function autopilot_fixpsf, psf, width=width, radius=radius, rsmooth=rsmooth, wsmooth=wsmooth, showme=showme
   compile_opt idl2

   sm = lambda(x,m,s: 1d / (1d + exp(-(x-m)/s) ) )

   ; width = 3 translates to ~30 pix end-to-end 1% from 0,1
   ; assume > 2.0 pix fwhm -> 6.0 pix end-end
   ; so width of 6/10 = 0.6 dampening width should be enough
   if not keyword_set(width) then width = 0.7

   dim = psf.dim
   dim2 = (dim-1)/2

   d1 = total(psf,2)
   d2 = total(psf,2)

   xmin = max(where(d1 eq 0 and lindgen(dim[0]) lt dim2[0]))
   xmax = min(where(d1 eq 0 and lindgen(dim[0]) gt dim2[0]))
   ymin = max(where(d2 eq 0 and lindgen(dim[1]) lt dim2[1]))
   ymax = min(where(d2 eq 0 and lindgen(dim[1]) gt dim2[1]))

   rmin = min(abs(dim2 - [xmin,ymin]) < abs([xmax,ymax] - dim2)) - 2
   mkgrid, dim, d=dd
   dd = double(dd)
   psf = double(psf)

; smooth outskirts of PSF
   if keyword_set(rsmooth) or keyword_set(wsmooth) then begin
      if not keyword_set(wsmooth) then wsmooth = 3
      if not keyword_set(rsmooth) then rsmooth = rmin/2

      wwinv =  sm(dd, rsmooth, rmin/30.0)  ; rmin/30 is ~ 1/6 sstamp 1% cut on/off
      ww = 1.0 - wwinv
      psfout = ww*psf + wwinv*smooth(psf,wsmooth)

      if keyword_set(showme) then begin
         cgimage, ww*psf, /keep, minv=-1e-6, maxv=1e-6, /str, layout=[3,2,1]
         cgimage, wwinv*psf, /keep, minv=-1e-6, maxv=1e-6, /str, layout=[3,2,2]
         cgimage, smooth(wwinv*psf,3), /keep, minv=-1e-6, maxv=1e-6, /str, layout=[3,2,3]
         cgimage, psfout, /keep, minv=-1e-6, maxv=1e-6, /str, layout=[3,2,4]
         cgimage, fft(psf), /keep, minv=-1e-9, maxv=1e-9, /str, layout=[3,2,5]
         cgimage, fft(psfout), /keep, minv=-1e-9, maxv=1e-9, /str, layout=[3,2,6]
         stopkey
      end
   end else psfout = psf

 ; sigmoid weighting function for edges
   ww = 1.0 - sm(dd, rmin-5.0*width, width)
   psfout *= ww

   if keyword_set(showme) then begin
      cgerase
      cgimage, psfout, /keep, minv=-1e-6, maxv=1e-6, /str, layout=[3,2,1]
      cgplot, alog10(psfout[85,*]/max(psfout))/8.+1, layout=[3,2,2]
      cgplot, ww[dim2[0],*], /overplot, col='red'
      hline, 0, linest=2
      cgimage, psfout*ww, /keep, minv=-1e-6, maxv=1e-6, /str, layout=[3,2,3]
      cgplot,  psfout[dim2[0],*], yr=[-0.5,3]*mad(psfout[85,0:dim2[0]/2]), layout=[3,2,4]
      cgplot, psfout[dim2[0],*]*ww[dim2[0],*], /overplot, col='red'
      cgimage, fft(psf), /keep, minv=-1e-9, maxv=1e-9, /str, layout=[3,2,5]
      cgimage, fft(psfout), /keep, minv=-1e-9, maxv=1e-9, /str, layout=[3,2,6]
      hline, 0, linest=2
     stopkey
   end

  return, psfout/total(psfout)
end

pro testbin
; @@@@ executive summary:
; - if binning (e.g. 3x3) and reconstruction at same pixel scale then no further action necessary
;   + use cubic=-0.5 for all other transformations
; - if binning but recontruction at original resolution presmooth hires by 1) binning 2) interpolation kernel
;   + e.g., smoothing: psf_sm = rbin( rbin(psf1,3), 3,/expand, interp=1)   for linear
;      so psf_pre = rbin(psf_sm, 3)
;

   bin = 1
   hdet = headfits(par.detect_image_weighted)
   if bin gt 1 then hrebin, hdet, outsize = [sxpar(hdet,'NAXIS*')/bin]
   getrot, hdet, pa, cdelt
   target_psf = Mophongo.getpsf_irac(gra, gdec, filename=froot[j]+'_psf.fits', pixscl = cdelt[1]*3600., pa=pa, /fast, overlap=hdet)

   adxy, hdet, gra, gdec, gx, gy
   psf = Mophongo.getpsf(gx, gy, filename=par.detect_image_weighted.replace('sci.fits','psf.sav'))

   psf1 = autopilot_fixpsf(psf[*,*,100],wsmooth=3 )
   psf2 = autopilot_fixpsf(target_psf[*,*,100],wsmooth=5,width=0.8)

   ; @@@ rbin == rebin( shift( smooth(   ,bin), -1, -1), /sample)
 ;  p1sm = smooth(psf1,3,/edge_trunc)
 ;  tvs, rbin(psf1,3)/9.0,mm=[-1,1]*3e-6
 ;  tvs, rebin(shift(p1sm,-1,-1),57,57,/sample),mm=[-1,1]*3e-6
 ;  tvs, rbin(psf1,3)/9.0 - rebin(shift(p1sm,-1,-1),57,57,/sample),mm=[-1,1]*1e-8 ; identical!!

   pu = python.import('photutils')
   wint = pu.TukeyWindow(alpha=0.4)
   winc = pu.SplitCosineBellWindow(alpha=0.7,beta=0.3)

   pp1 = psf1
;   pp2 = smooth(psf2,3)  --> ok! the rebinned is like comparing to a smoothed(3) kernel full
   pp2 = psf2
   sz1 = max(pp1.dim)
   sz2 = max(pp2.dim)
   ksz = (sz2 > sz1)
   npad = 7 + (long(ksz)/2 ne ksz/2.0)
   if sz1 lt sz2 then pp1 = extrac(pp1, (sz1-sz2)/2,  (sz1-sz2)/2, sz2, sz2)
   if sz2 lt sz1 then pp2 = extrac(pp2, (sz2-sz1)/2,  (sz2-sz1)/2, sz1, sz1)
   ; extra padding for fft edge effects
   pp1 = extrac(pp1, -npad/2, -npad/2, ksz+npad, ksz+npad)
   pp2 = extrac(pp2, -npad/2, -npad/2, ksz+npad, ksz+npad)

   tvs,center(pp1,/ch)
   tvs,center(pp2,/ch)

    decon, pp1, pp2, kern_li,  maxiter=par.kern_maxiter, klim=par.kern_klim, rchi=rr, $
                 max_entropy=max_entropy, likelihood=likelihood, hermite=hermite, maxdev=dd, $
                 nbasis=kern_nbasis, basis=basis, nlow=nlow, verbose=verbose, method=me, $
                 display=1, step=1.0, regulate=par.kern_regulate, dscale=dscale, $
                 rmin_chi=par.phot_minradius/scl/bin

   kern_full = kern_li[npad/2:ksz+npad/2-1,npad/2:ksz+npad/2-1]
   pp_full = pp1[npad/2:ksz+npad/2-1,npad/2:ksz+npad/2-1]
   conv_full = fconvolve(kern_full, pp_full,pad=0)

   ; ------ binned psf1 and binned psf2
 pp1 = rbin(psf1,3)
 pp2 = rbin(psf2,3)

   sz1 = max(pp1.dim)
   sz2 = max(pp2.dim)
   ksz = (sz2 > sz1)
   npad = 7 + (long(ksz)/2 ne ksz/2.0)
   if sz1 lt sz2 then pp1 = extrac(pp1, (sz1-sz2)/2,  (sz1-sz2)/2, sz2, sz2)
   if sz2 lt sz1 then pp2 = extrac(pp2, (sz2-sz1)/2,  (sz2-sz1)/2, sz1, sz1)
   ; extra padding for fft edge effects
   pp1 = extrac(pp1, -npad/2, -npad/2, ksz+npad, ksz+npad)
   pp2 = extrac(pp2, -npad/2, -npad/2, ksz+npad, ksz+npad)

   tvs,center(pp1,/ch)
   tvs,center(pp2,/ch)

; works on binned, but not on original!
 ;  kernt = pu.create_matching_kernel(psf1b, psf2b, window=wint)
;   kern_bc = pu.create_matching_kernel(pp1, pp2, window=winc)

    decon, pp1, pp2, kern_lib,  maxiter=par.kern_maxiter, klim=par.kern_klim, rchi=rr, $
                 max_entropy=max_entropy, likelihood=likelihood, hermite=hermite, maxdev=dd, $
                 nbasis=kern_nbasis, basis=basis, nlow=nlow, verbose=verbose, method=me, $
                 display=1, step=1.0, regulate=par.kern_regulate, dscale=dscale, $
                 rmin_chi=par.phot_minradius/scl/bin

   kern_bin3 = kern_lib[npad/2:ksz+npad/2-1,npad/2:ksz+npad/2-1]
   psf_bin3 = pp1[npad/2:ksz+npad/2-1,npad/2:ksz+npad/2-1]
   conv_bin3 = fconvolve(kern_bin3, psf_bin3,pad=0)

; ------ binned pre-smoothed psf1 and binned psf2
; pp1 = rbin(gauss_smooth(psf1,1,/edge_tr),3)
; pp1 = rbin(smooth(psf1,3,/edge_tr),3)                ; @@@@ !!!! yes this works for cubic  !
 pp1 = rbin(rbin(rbin(psf1,3),3,/expand,interp=1),3)   ; @@@@ !!!! yes this works too and corrects for linear interpolant
 pp2 = rbin(psf2,3)

   sz1 = max(pp1.dim)
   sz2 = max(pp2.dim)
   ksz = (sz2 > sz1)
   npad = 7 + (long(ksz)/2 ne ksz/2.0)
   if sz1 lt sz2 then pp1 = extrac(pp1, (sz1-sz2)/2,  (sz1-sz2)/2, sz2, sz2)
   if sz2 lt sz1 then pp2 = extrac(pp2, (sz2-sz1)/2,  (sz2-sz1)/2, sz1, sz1)
   ; extra padding for fft edge effects
   pp1 = extrac(pp1, -npad/2, -npad/2, ksz+npad, ksz+npad)
   pp2 = extrac(pp2, -npad/2, -npad/2, ksz+npad, ksz+npad)

   tvs,center(pp1,/ch)
   tvs,center(pp2,/ch)

 ;  kernt = pu.create_matching_kernel(psf1b, psf2b, window=wint)
 ;  kern_bc = pu.create_matching_kernel(pp1, pp2, window=winc)
   decon, pp1, pp2, kern_lib,  maxiter=par.kern_maxiter, klim=par.kern_klim, rchi=rr, $
                 max_entropy=max_entropy, likelihood=likelihood, hermite=hermite, maxdev=dd, $
                 nbasis=kern_nbasis, basis=basis, nlow=nlow, verbose=verbose, method=me, $
                 display=1, step=1.0, regulate=par.kern_regulate, dscale=dscale, $
                 rmin_chi=par.phot_minradius/scl/bin

   kern_pre3 = kern_lib[npad/2:ksz+npad/2-1,npad/2:ksz+npad/2-1]   ; @@@@ !!!!  kernel is now preconvolved for binning
   psf_pre3 = pp1[npad/2:ksz+npad/2-1,npad/2:ksz+npad/2-1]
   conv_pre3 = fconvolve(kern_pre3, psf_pre3,pad=0)

;---------- compare -> bin3 upsample

  tvs, conv_full - rbin(conv_bin3, 3, /expand, cubic=-0.5), mm=[-1,1]*5e-6, pos=0
  tvs, conv_full - rbin(conv_bin3, 3, /expand, interp=1), mm=[-1,1]*5e-6, pos=1
  tvs, conv_full - fftbin(conv_bin3, 3)/9.0, mm=[-1,1]*5e-6, pos=1
  tvs, smooth(conv_full,3) - rbin(conv_bin3, 3, /expand, cubic=-0.5), mm=[-1,1]*5e-6, pos=2 ; @@@ ok!!!
  tvs, smooth(conv_full,5) - rbin(conv_bin3, 3, /expand,interp=1), mm=[-1,1]*5e-6, pos=3


;---------- compare -> full down sample
; @@@@ so when reconstructing in binned coordinates -> do convolution as usual
; @@@@ and when doing at full resolution, pre-smooth hi-res kernel by binning
  tvs, rbin(conv_full,3) - conv_bin3, mm=[-1,1]*5e-5, pos=0           ; @@@ this is also ok!
  tvs, rbin(smooth(conv_full,3),3) - conv_bin3, mm=[-1,1]*5e-6, pos=1

  tvs, rbin(conv_full,3) - conv_bin3, mm=[-1,1]*1e-5, pos=0
  tvs, rbin(conv_full,3), mm=[-1,1]*1e-5, pos=0

;---------- compare -> pre3 upsample
 tvs, rbin(conv_full,3) - conv_pre3, mm=[-1,1]*5e-6, pos=0
 tvs, rbin(conv_full,3) - conv_pre3, mm=[-1,1]*5e-6, pos=0
  pb1 = rbin(smooth(psf1,3),3)

; so note: @@@ a binning deconvolved kernel is not appropriate to compare to a
; binned template
 tvs, rbin(conv_full,3) - fconvolve(kern_pre3, rbin(psf1,3), pad=0), mm=[-1,1]*5e-6, pos=1
 tvs, rbin(conv_full,3) - fconvolve(kern_pre3, rbin(smooth(psf1,3),3), pad=0), mm=[-1,1]*5e-6, pos=2

 tvs, rbin(smooth(conv_full,3),3) - conv_pre3, mm=[-1,1]*5e-6, pos=1
 tvs, conv_pre3 - conv_bin3, mm=[-1,1]*5e-5, pos=3


;---------- compare -> bin3 upsample

  tvs, conv_full - rbin(conv_bin3, 3, /expand, cubic=-0.5), mm=[-1,1]*5e-6, pos=0
  tvs, conv_full - rbin(conv_bin3, 3, /expand, interp=1), mm=[-1,1]*5e-6, pos=1
  tvs, smooth(conv_full,3) - rbin(conv_bin3, 3, /expand, cubic=-0.5), mm=[-1,1]*5e-6, pos=2
  tvs, smooth(conv_full,5) - rbin(conv_bin3, 3, /expand,interp=1), mm=[-1,1]*5e-6, pos=3

;; @@@@ !!!!  kernel is now preconvolved for bi
; !!!!! YES !!!!
; so just pre-convolve (smooth) the hi-res PSF by the binning, before deconvolution
; this effectively deconvolves for the extra binning
; apply this to the original resolution
  tvs, conv_full - rbin(fconvolve(kern_pre3,  psf_bin3), 3, /expand, cubic=-0.5), mm=[-1,1]*5e-6, pos=0
  tvs, conv_full - rbin(fconvolve(kern_pre3,  psf_bin3), 3, /expand, interp=1), mm=[-1,1]*5e-6, pos=1

  ; ok!!!
  tvs, smooth(conv_full - rbin(fconvolve(kern_pre3,  psf_bin3), 3, /expand, interp=1),3), mm=[-1,1]*5e-6, pos=2

; rebinning both recovers the normal binned
  tvs, rbin(conv_full,3) - rbin(rbin(fconvolve(kern_pre3,  psf_bin3), 3, /expand, cubic=-0.5),3), mm=[-1,1]*5e-6, pos=1


end

; deal with fits later
;      writefits, froot[j]+'_kern.fits', kernel, hdet
;      ftcreate,3,gra.length,h,tab         ; create table header and (empty) data
;      ftaddcol,h,tab,'gra',gra.typecode   ; explicity define the
;      ftaddcol,h,tab,'gdec',gra.typecode  ; ra and dec columns
;      ftaddcol,h,tab,'gra',gra.typecode   ; explicity define the
;      ftaddcol,h,tab,'gdec',gra.typecode  ; ra and dec columns
;      ftaddcol,h,tab,'gdec',gra.typecode  ; ra and dec columns
;      ftput,h,tab,'gra',0,gra[*]          ; insert ra vector into table
;      ftput,h,tab,'gdec',0,gdec[*]        ; insert dec vector into table
;      writefits, froot[j]+'_kern.fits', tab, h, /append
;      stop
;mwrfits,kpca,'tab.fits',/create
;mwrfits,kpca,'tab.fits',/create
;a=MRDFITS('tab.fits')
;a=MRDFITS('tab.fits',alias=['basis','4'])

      ; ftput, h,tab, 'pca1',0,findgen(gra.length)   ;insert name vector with default
    ;      gcp = growthcurve( psfd, rhalf=rhp)
    ;      gct = growthcurve( target_psfd, raper=raper,rhalf=rht)
         ; cgplot,raper,gcp/gct,yr=[0.9,1.1] ,/noeras & hline,1.0,col='white',linest=2
        ; par.kern_check=1
      ;    target_psf = Mophongo.getpsf_irac(gra, gdec, filename=froot[j]+'_psf.fits', pixscl = -3600*cdelt[0], pa=pa)
      ;    target_orpa = Mophongo.getpsf_irac(gra, gdec, filename=froot[j]+'_psf.fits',pa=pa)
      ;   target_orpa_psf = Mophongo.getpsf_irac(gra, gdec, filename=froot[j]+'_psf.fits', pixscl = -3600*cdelt[0], pa=pa,/fast)
      ;!null = center(target_psf,/check)
      ;!null = center(target_orpa,/check)
      ;!null = center( target_orpa_psf,/check)
      ;   tvs, target_psf,os=0.5
      ;   tvs, target_orpa[*,*,0],pos=1
      ;   tvs,target_psf_orpa[*,*,0],pos=1
      ;  tvs,(target_psf[*,*,0]-target_orpa_psf[*,*,0])/(target_psf[*,*,0] + 0.001),pos=2,mm=[-1,1]*1e-2
       ; dscale=1e-5
 ;     stop
    ;          method[i] = me    &    rchi[i] = rr   &   maxdev[i] = dd
   ;     if n_elements(dd) ne 0 then if finite(dd) eq 0 then contine

 ; c=win(psfd.dim)
 ;     a=fft(psfd,/center)
 ;     b=fft(target_psfd,/center)
 ;     kkk = shift(real_part(fft(b/a*c,/inverse)),85,85)
 ;     kkk /= total(kkk)
 ;     tvs,convolve(psfd,kkk)
 ;     tvs,kkk, mm=[-1,1]*1e-3,pos=1
 ;     tvs,kern,mm=[-1,1]*1e-3,pos=0
  ;pu = python.import('photutils')
;kernel = pu.create_matching_kernel(psf[*,*,0], target_psf[*,*,0], window=win)
;kernel = pu.create_matching_kernel(psf[*,*,0], target_psf[*,*,0])
;psfc = fconvolve(psf[*,*,i],kernel)
;win=pu.TukeyWindow(alpha=0.3)
;plot,(win([171,171]))[85,*]
;kernel = pu.create_matching_kernel(psf[*,*,0], target_psf[*,*,0], window=win)
;psfc = fconvolve(psf[*,*,i],kernel)
;my_psfc = fconvolve(psf[*,*,i],kern)
;tvs, my_psfc, mm=[-1,1]*1e-4
;tvs, psfc, mm=[-1,1]*1e-4,pos=1
;tvs, kern, mm=[-1,1]*1e-4,pos=2
;tvs, kernel, mm=[-1,1]*1e-4,pos=3
;tvs, target_psf-my_psfc, pos=4, mm=[-1,1]*5e-5
;tvs, target_psf-psfc, pos=5, mm=[-1,1]*5e-5
;gc=growthcurve(target_psf-psfc,raper=raper)
;mygc=growthcurve(target_psf-my_psfc,raper=raper)
;plot, raper, mygc,yr=[-0.02,0.02],/xlog,xr=[0.9,85]
;oplot, raper, gc,  linest=2




; fit moffat with optional penalizing negative or positve residuals
function autopilot_moffat, p, r=r, gc=gc, w=w, showme=showme
   psf = psf_moffat(npix=round(max(r)*2),fwhm=p[0],beta=p[1],/norm)
   gc_model = growthcurve(psf,raper=r)

   if keyword_set(showme) then plot, r, gc,thick=3,/xlog, xrange=[0.5,max(r)],/xst
   if keyword_set(showme) then oplot, r, gc_model, linest=2

   res = gc - gc_model
   ineg = where(res lt 0,nneg)
   if nneg gt 0 then res[ineg] *= w  ; one-sided penalize of model (to force bigger/smaller)
   return, res
end

; interpolate kernel on grid
pro autopilot_get_kernel, fkern, x, y, kern, display=display, gx=gx, gy=gy, kernel=kernel
   if not keyword_set(kernel) then  begin
      restore, fkern
      kernel = reform(kernel, [(size(kernel))[1], (size(kernel))[1], size(gx,/dim)])
   end
   tsz = (size(kernel))[1]        ; @@@ check should take existing grid!!!!!!!
   len =  n_elements(x)
   kern = fltarr([tsz,tsz,len])
   sz=size(gx,/dim)

   ; kk = reform(kernel, [tsz,tsz,size(gx,/dim)])
   for i=0,len-1 do begin
      ckern = interpolate(kernel, 1.*(sz[0]-1)*(x[i]-min(gx))/(max(gx)-min(gx)), 1.*(sz[1]-1)*(y[i]-min(gy))/(max(gy)-min(gy)), missing=0.,cubic=-0.5)
      kern[*,*,i] = ckern/total(ckern)
   end
end

; generate the photometry images by convolving to target PSF
pro autopilot_make_phot, par, _extra=extra

   fdecomp, par.phot_image, !null, dir, fname
   froot = [par.dir_work + fname.replace('_sci',''), par.detect_image_weighted.replace('_sci.fits','')]

   npfft = python.import('numpy')
   bin = ceil(par.phot_minradius/pixscale(par.detect_image_weighted)) > 3
   h=headfits(froot[0]+'_sci.fits')
   for i=0,n_elements(froot)-1 do if file_test(froot[i]+'_con.fits') then $
      logger, 'found existing PSF matched '+froot[i]+'_con.fits', /info else begin

      ; convert kernel to fits: read from header which type.
      ; PHOTMODE - FITTING / APERTURE
      ; restore,  froot[i]+'_kern.sav'
      ; @@@ hardwire for greats. next step split be half light radius
      ; -> require measuring them for early on
       if froot[i].matches('GREATS') then begin
         logger, 'not convolving ', froot[i]+' - only fitting photometry ',/info
         continue
       end

      logger, 'convolving with kernel ', froot[i]+'_kern.sav',/info
; use Todd lauer coefficient fft trick

      ; @@@@ fixed PSF for now
      autopilot_get_kernel, froot[i]+'_kern.sav', sxpar(h,'NAXIS1')/2, sxpar(h,'NAXIS1')/2, $
                         kern, gx=gx, gy=gy, kernel=kernel

      img = readfits(froot[i]+'_sci.fits',h,/silent,nan=0.0)
      wht = readfits(froot[i]+'_wht.fits',h,/silent,nan=0.0)

; @@@ this can go a lot faster...
      iw0 = where(img eq 0, /null)
      img = fconvolve(temporary(img), kern)

      img = skysub(temporary(img),wht,rms=rmsimg,bin=bin)
      img[iw0] = 0

      ; @@@ need to test whether these convolved RMS are any good: they are probably optimistic
      logger, 'overwriting rms file with convolved rms ',froot[i]+'_rms.fits'
      writefits, froot[i]+'_wht.fits', temporary(1.0/(rmsimg^2)), h  ; @@@@ note need to catch 0's etc
      writefits, froot[i]+'_rms.fits', temporary(rmsimg), h
      writefits, froot[i]+'_con.fits', temporary(img), h
      report_memory
   end

end

; retrieve latest dust map from
pro autopilot_galactic_extinction, par

   fdust = par.detect_image_weighted.replace('detect_sci','dustmap')
   if file_test(fdust) eq 0 then begin
      h = headfits(par.detect_image_weighted)
      xyad, h, sxpar(h,'NAXIS1')/2., sxpar(h,'NAXIS2')/2., ra, dec
      url_query = par.dust_server+'&locstr='+strn(ra)+'+'+strn(dec)+'+equ+j2000'
      netObject = Obj_New('IDLnetURL')
      xml = (netObject.Get(URL=url_query, /string_array)).trim()
      url_dustmap = (xml.filter(lambda(n:n.matches('Dust.fits'))))[0]
      logger, 'retrieving '+file_basename(url_dustmap)+' from '+url_query, /info
      result = netObject -> Get(URL=url_dustmap, filename=fdust)
      dustmap = ( xml.filter(lambda(n:n.matches('Dust.fits') )) )[0]
      Obj_Destroy, netObject
   end else logger, 'found existing dustmap ', fdust, /info

 ; load catalog, apply reddening

; what about ebv: schlegel dust maps from http://irsa.ipac.caltech.edu/cgi-bin/bgTools/nph-bgExec
; suggest significant variations: e.g. EBV 0.047 - 0.074 by 60% over 20 arcmin scale
; GOODS-S similar EBV 0.006 - 0.010 over 10 arcmin
; Finally, the extinction is derived from the reddening using a standard extinction law and
; assuming a constant total-to-selective extinction parameter (Rv = 3.1). FinkBeiner suggests
; Fitzpatrick R_V=3.1 is best

; /Users/ivo/Astro/PROG/idl/idl-astro/pro/astro/fm_unred.pro
; so to replace in place
;  fm_unred, wave, flux, ebv, R_V = 3.1

end



function auto_pilot_config, config, _extra=extra
   param = dictionary()
   param['config'] = 'auto.param'      ; param  file
   param['verbose'] = 'debug'          ; ['CRITICAL','ERROR','WARNING','INFO','DEBUG'] or 0-4
   param['dir_work'] = 'work/'               ;
   param['dir_image'] = './'
   param['prefix'] = ''                ; list of prefix filtering strings to select image -> @@@ can probably merge with regexp below
   param['postfix'] = '.fits*'         ; only single
   param['image_regexp'] = 'sci'       ; regular expression pattern indicating image
   param['weight_regexp'] = 'weight|wht|exp'    ;  regular expression pattern indicating weight
   param['detect_image'] = !NULL
   param['detect_weight'] = !NULL
   param['minweight'] = 0.03           ; minimum weight acceptible weight relative to median
   param['maxweight'] = 100.0          ; minimum weight acceptible weight relative to median
   param['detect_filter'] = !NULL      ; NULL for auto, filename for custom kernel, 0 or 'no' for no filter
   param['phot_image'] = !NULL
   param['phot_weight'] = !NULL
   param['detect_band'] = ''
   param['phot_band'] = ''
   param['detect_thresh'] = !NULL
   param['detect_minarea'] = 5
   param['detect_spurious'] = 0.003   ; fraction of acceptible contaminating sources
   param['detect_filter'] = !NULL      ; give filter filename or fwhm of gaussian
   param['detect_method'] = 'variance' ; weight by inverse variance, or chi2 Szalay
   param['deblend_nthresh'] =  32       ; sextractor deblending threshold
   param['deblend_mincont'] =  0.00001  ; sextractor deblending threshold
   param['star_percentile'] = 0.1      ; use brightest percentile and most compact sources to find stars
   param['star_ellipticity'] = 0.20    ; star ellipticity must be less than
   param['star_maxfwhm'] = !NULL       ; star half light radius must be less than
   param['star_deltafwhm'] = 0.1       ; star half light radius must deviate less than this fraction

   param['register_snrlim'] = 30       ; minimum SNR for objects to include in shift calculation
   param['register_dim'] = 1           ; polynomial degree for fit
   param['skysub_det']  = 1            ; 1 = skysub on detection image
   param['skysub_phot']  = 1           ; 1 = skysub on detection image

   param['psf_cat'] = !NULL            ; catalog of star ra,dec,id. default: det[0]+'_star.cat'
   param['psf_snrlim'] = 200           ; SNR limit for psf analysis
   param['psf_maxshift'] = 0.3         ; maximum shift when recentering [arcseconds]
   param['psf_blender'] = 0
   param['psf_combine'] = 'average'    ; median or average
   param['psf_bgthresh'] = 3.0
   param['psf_tile_size'] = 10.0       ; in arcsec
   param['psf_peakthresh'] = 1.0       ; reject as saturated all stars within peakthresh * max(stars)
   param['psf_magbuf'] = 0.0           ; also remove stars magbuf fainter
                                       ; so 1.0,0.0 rejects nothing, 0.9,0.5 rejects all sources
                                       ; with peak > 0.9*max_val and all sources 0.5 mag fainter
   param['psf_max_basis'] = !null     ; maximum number KL basis functions
   param['psf_minstar'] = 2           ; minimum # stars for KL
   param['psf_check'] = 1

   param['kern_klim'] = 1e-3          ; maximum absolute deviation of growthcurve
   param['kern_maxiter'] = 20         ; maximum number of deconvolution iterations
   param['kern_method'] = 'likelihood'         ; maximum number of deconvolution iterations
   param['kern_regulate'] = 0.7     ; 0.0-1.0  ; use values closer to 1 if PSFs similar FWHM
   param['kern_check'] = 1           ; maximum number of deconvolution iterations

   param['phot_rhalf_fit'] = 0.3    ; minmum half light radius, above this limit fitting, below aperture
   param['phot_background_factor'] = 30.0  ; aperture background = factor * rhalf_fit
   param['phot_minradius'] = 0.25    ; minimum aperture [arcsec] (hard lower limit for major,minor axis)
   param['phot_kron'] = 1.0          ; kron_factor for photometry, set to 0 for aperture photometry at phot_minradius

   param['fit_snrlo_psf'] = 10.0   ; at SNR < 1.5*snrlo a PSF template of SNR = snrlo is added in
                                   ; quadrature as a prior so faint sources will converge to a PSF
   param['fit_snrhi_psf'] = 150.0  ; at SNR > snrhi an extra PSF template is added to the basis
                                    ; providing 1-D freedom in fitting structure, e.g. due to
                                    ; color gradients or a nuclear component
   param['fit_sparse_threshold'] = 1e-5
   param['fit_neg_threshold'] = -3   ; if best fit less than fit_neg_threshold sigma regularize
   param['fit_syserr'] = 0.03          ; systematic error as fraction of flux
   param['fit_memmap']  =  0    ; memory map to file ?
   param['fit_shift_sigclip'] = 3.0
   param['fit_bin'] = 0              ; binning of template (0 = auto)  ;@@@  round to whole pixels
   param['fit_tol'] = 1e-8
   param['fit_fast'] = 1
   param['fit_interp'] = 'cubic'     ; cubic/linear

   param['detect_kron'] = 1.5        ; kron factor for total magnitude (on detection image)
   param['bias_fraction'] = 0.1      ; if kron ellipse overlaps with neighbor segment by > bias_fraction*iso_area
   param['bias_shrink'] = 1.4        ;   shrink kron major/minor axes by factor
   param['blend_shrink'] = 1.4       ; if blended then shrink kron major/minor axes to factor * iso_area
   param['log'] = ''                 ; log file
   param['dust_server'] = 'http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?regSize=5.0'

   logger, verbose=param.verbose

   if not keyword_set(config) and not file_test(param.config) then begin
      logger, 'No config supplied and no default found, writing default to auto.param', /critical
      ; @@@@ todo write default config to auto.param
   end

  readparam, config, param, default=param.config, _extra=extra

  logger, verbose=param.verbose, filename=param.log


; @@@ go through entire parameter list and do sanity checking... file tests etc etc
; foreach p, (par.keys()).filter(lambda(n:n.matches('dir'))) do print,par[p]
; foreach p, par, v do print, v, ' ', p
; n_elements(psf_max_basis) eq psf.length ? psf_max_basis[i] : !NULL,
; @@@ ensure directories end in /
; self.param.fit_sparse_threshold > 0

   ; auto fill detection and phot image names
   if not keyword_set(param.detect_image) then begin
      foreach dir, param.dir_image do $
         foreach band, param.detect_band do $
            foreach pre, param.prefix, i do $
               fd = append(fd, file_search(dir,'*'+pre+'*'+band+'*'+param.postfix, /test_read, /fold))
      param.detect_image = fd[where(fd.matches(param.image_regexp),/null)]
      param.detect_weight = fd[where(fd.matches(param.weight_regexp),/null)]
   end
   if not keyword_set(param.phot_image) then begin
      foreach dir, param.dir_image do $
         foreach band, param.phot_band do $
            foreach pre, param.prefix, i do $
               fp = append(fp, file_search(dir,'*'+pre+'*'+band+'*'+param.postfix, /test_read, /fold))
      param.phot_image = fp[where(fp.matches(param.image_regexp),/null)]
      param.phot_weight = fp[where(fp.matches(param.weight_regexp),/null)]
   end

   foreach p, param[['detect_'+['image','weight'],'phot_'+['image','weight']]], key do $
      foreach pp,p do if not file_test(pp) then message, key+' '+pp+' not found!'

   if keyword_set(param.dir_work) then file_mkdir, param.dir_work

   param['BIAS'] = 1         ; constants which can not be overridden by param file
   param['BLEND'] = 2         ; constants which can not be overridden by param file

   logger, 'detection images: '+(file_basename(param.detect_image)).join(' '), /info
   logger, 'photometry images: '+(file_basename(param.phot_image)).join(' '), /info

   return, param
end

;# lists of point sources to use for PSF. Default: use detection map star list
;# building psf map with klpsf param
;## shouldn't touch below unless necessary
;r_center = AUTO     ; aper radius to center  (AUTO) r=[tile_size/r_center]
;r_bg =     AUTO     ; background aper (+second pass normalize aper) in r=[tile_size/2-r_bg]
;bgthresh = 3.5        ; reject threshold in sigma for background estimate
;mthresh = 4.5        ; reject threshold in sigma for neighbor masking
;grow = 1.            ; number of pixels to grow rejection mask
;niter = 1           ; number of iterations for rejection
;gclim = 0.2         ; reject stars whose growth curves deviate by more than gclim
;ngrid = AUTO          ; psf is spline interpolated over grid x grid map
;average = 1          ; do KL decompose relative to average rather than median
;   param['psf_ratio_thresh,'] = 1
;magbuf = 0.0         ; also reject stars that are magbuf fainter than saturated stars
;# make kernel (on the fly from PSF maps)
;#  psflo - (psfhi x kernel) OR a relative improvement 1/20. of that
;klim = 1e-3         ; maximum absolute deviation of growthcurve
;maxiter = 20         ; maximum number of deconvolution iterations


pro autopilot, config, _extra=extra
   compile_opt idl2
   !EXCEPT=0

   ; verbosity level is overwritten after loading config
   logger, verbose='debug', filename=flogname

   param = auto_pilot_config(config, _extra=extra)

   autopilot_prep_detection,  param, _extra=extra

   autopilot_make_detection, param, _extra=extra
   ; create new fancy star finding using nearest neighbors ? star finder ?

   autopilot_detect_objects, param, _extra=extra

   autopilot_register_phot,  param, _extra=extra

   autopilot_map_psf,  param, _extra=extra

   autopilot_target_psf,  param, _extra=extra

   autopilot_subtract_background, param, _extra=extra

   autopilot_map_kernel,  param, _extra=extra, /stopme

   autopilot_make_phot, param, _extra=extra

   autopilot_photo, param, _extra=extra

stop

; catalog object
   ; photometry
end

pro autopilot_test

end

; catalog class



pro autopilot_junk
!p.multi=[0,2,2]
plot, obj.gfwhm, obj.fwhm,psym=3,xr=[0,20],yr=[0,20]
plot, obj.gfwhm, obj.fwhm_ab,psym=3,xr=[0,20],yr=[0,20]
plot, obj.gfwhm, obj.rhalf,psym=3,xr=[0,20],yr=[0,20]
m = mag(obj.auto)

end


pro test
stop
maskc = smooth(mask,ceil(6*fwhm))
detimgc[iw0] = 0
detimgcs = skysub(detimgc,  rms=rmsimgc, sky=skyimgc,  bin=3, _extra=extra,/verbose)
detimgcs[iw0] = 0

print, mad(detimgc,badval=0)
print, mad(detimgc*maskc,badval=0)
print, mad(detimgcs*maskc,badval=0)
print, mad(detimg,badval=0)
print, mad(detimg*smooth(mask,3*bin),badval=0)
tvs, mask*detimg, zoom=5
tvs, maskc*detimgc, zoom=1, mm=[-0.3,0.3]
tvs, maskc*detimgcs, zoom=1, mm=[-0.3,0.3]
plothist, (mask*detimg)[where(mask*detimg ne 0 and finite(detimg))],/auto,col='red',xr=[-1.2,1.2]
plothist, (maskc*detimgc)[where(maskc*detimgc ne 0 and finite(detimgc))],/auto, col='blue', /overplot
plothist, (maskc*detimgcs)[where(maskc*detimgcs ne 0 and finite(detimgc))],/auto, col='green', /overplot
;histogauss, (mask*detimg)[where(mask*detimg ne 0 and finite(detimg))],aa
;histogauss, (maskc*detimgc)[where(maskc*detimgc ne 0 and finite(detimgc))],aa
;histogauss, (maskc*detimgc)[where(maskc*detimgcs ne 0 and finite(detimgc))],aa
histogauss, (maskc*detimgcs)[where(maskc*detimgcs ne 0 and finite(detimgcs))],aa
;print, aa

print,minmax(rmsimg[where(rmsimg gt 0)])

end



         ;if not keyword_set(psf_snrlim) then psf_snrlim = [400]
        ; run klpsf
; readparam, param, default='phot.param', _extra=extra, stopme=stopme
 ;file_mkdir, psfdir
;psf =  keyword_set(psf) ? psf : (keyword_set(phot) ? [det,phot]: det)
; if psf_cat.length lt psf.length then psf_cat = [psf_cat, replicate(psf_cat[-1], psf.length-psf_cat.length)]
 ;if psf_snrlim.length lt psf.length then psf_snrlim = [psf_snrlim, replicate(psf_snrlim[-1], psf.length-psf_snrlim.length)]

;print,
 ; fpsf=psf.replace('_sci','')
 ; fpsf=fpsf.replace('.fits','_psf.fits')
 ; if n_elements(psf_max_basis) ne 0 then if psf_max_basis.length lt psf.length then psf_max_basis = [psf_max_basis, replicate(psf_max_basis[-1], psf.length-psf_max_basis.length)]
;  for i=0,n_elements(psf)-1 do if not file_test(fpsf[i]) then print, imdir+'/'+file_basename(psf[i]), psf_cat[i], ' snrlim:', psf_snrlim[i], ' max_basis:', n_elements(psf_max_basis) eq psf.length ? psf_max_basis[i] : !NULL
;  for i=0,n_elements(psf)-1 do if not file_test(fpsf[i]) then $


