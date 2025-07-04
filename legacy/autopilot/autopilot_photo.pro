
; @@@ download from github latest eazy filters
; resolve names into sensible names
pro autopilot_resolve_filters
end

pro  autopilot_photo, par, _extra=extra
   compile_opt idl2

   logger, 'measuring objects', /info
   fdecomp, par.phot_image, !null, dir, fname, ext


   ; initialize object and load
   mp = Mophongo(par)

   mp.load_detect, par.detect_image_weighted

   mp.load_photo, par.dir_work + fname[-1]+'.'+ext[-1]

   mp.load_catalog, par.detect_image_weighted.replace('detect_sci.fits','objects.sav')

   mp.fitphot

stop


   ; @@@@ only consider unblended objects in deepest part of image here
   ;print, minmax(obj.nchildren,imm)
   ;print, obj[imm[1]].id
   ;plothist, obj.nchildren,bin=0.5,yr=[0.5,100],/ylog
  ;    obj = obj[where(obj.use)]  ; ignore previous blends
  ;    obj[where(obj.parent_id ne 0)].flag = obj[where(obj.parent_id ne 0)].flag or par.BLEND
   ;; ok looks good. 5500 sources, 29.8 5 sig total
    ;  obj = obj[where(rms[round(obj.x),round(obj.y)] lt 12e-4)]
   ; onl

   ; get psf of detection image to get aperture correction. For now just keep PSF fixed
;   getklpsf, par.detect_image_weighted.replace('sci.fits','psf.sav'), sxpar(hdet,'NAXIS1')/2, sxpar(hdet,'NAXIS1')/2, detpsf
;   autopilot_get_kernel,par.detect_image_weighted.replace('sci.fits','kern.sav'), sxpar(hdet,'NAXIS1')/2, sxpar(hdet,'NAXIS1')/2, kern, gx=gx, gy=gy, kernel=kernel
;   photpsf = convolve(detpsf, kern)

;   faper, var, obj.x, obj.y, sqrt(obj.kron_major*obj.kron_minor), ftot_var, /status
 ; force the ratio to be a point source correction at minimum
; @@@ NO     psfcor = inv_phot_apcor /  inv_det_apcor
; note that due to scatter and contamination from neighbors in PSF matched that
;  phot_auto can be higher than detect_auto

; diagnostic plots
iblend = where(obj.flag and par.BLEND)
ibias = where(obj.flag and par.BIAS)
cgplot, mag(obj.flux_auto*obj.totcor,26), alog10(sqrt(obj.kron_area/!pi)),psym=16,symsize=0.5
cgplot, mag((obj.flux_auto*obj.totcor)[ibias],26), alog10(sqrt(obj[ibias].kron_area/!pi)),psym=16,symsize=0.5, col='orange',/overplot
cgplot, mag(obj[iblend].flux_auto*obj.totcor,26), alog10(sqrt(obj[iblend].kron_area/!pi)),psym=16,symsize=0.5, col='red'

   for i=0,froot.length-1 do begin
      i=6
  logger, froot[i]
;     i=8
      img = readfits(froot[i]+'_sci.fits',h,/silent,nan=0.0)
      con = readfits(froot[i]+'_con.fits',h,/silent,nan=0.0)
      wht = readfits(froot[i]+'_wht.fits',/silent,nan=0.0)
      rms2 = readfits(froot[i]+'_con_rms.fits',/silent,nan=0.0)
      rms = readfits(froot[i]+'_rms.fits',/silent,nan=0.0)

     faper, rms, obj.x, obj.y, obj.kron_major, fpix, elon=obj.kron_major/obj.kron_minor, theta=obj.theta, /status,/mean
     frms =  sqrt(obj.kron_area)*fpix
     faper, rms2, obj.x, obj.y, obj.kron_major, fpix2, elon=obj.kron_major/obj.kron_minor, theta=obj.theta, /status,/mean
     frms2 =  sqrt(obj.kron_area)*fpix
     faper, con, obj.x, obj.y, obj.kron_major, fcon, elon=obj.kron_major/obj.kron_minor, theta=obj.theta, /status
     faper, wht, obj.x, obj.y, obj.kron_major, fwht, elon=obj.kron_major/obj.kron_minor, theta=obj.theta, /status, /mean
  snr = fcon/frms  ; seems wrong
  print, fcon[10:17]
  print, fwht[10:17]
  print, obj[10:17].kron_area, obj[10:17].kron_major

print,median(sqrt(obj.kron_area/!pi)), median(sqrt(obj.kron_area))
print,median(sqrt(obj.kron_area/!pi))*0.06*2

; @@@ fix weight edge effects -> higher threshold ?
; rms seems to high ?

   getstamps, img, obj.x,obj.y, sst
   getstamps, con, obj.x,obj.y, pst
   tvs, -pst[*,*,10:17], os=3, mm=[-1,1]*0.3e-2
   tvs, -sst[*,*,10:17], os=3, mm=[-1,1]*0.5e-2, pos=8

print,  mag(median(fpix)*8*5,26.0)
m = mag(fsci*obj.totcor,26.0)
plothist, m , /nan, xr=[23,31], bin=0.2, col='red', /ylog
;   plot, obj.x, obj.y, psym=1,/iso

print,minmax(wht), minmax(fwht), median(snr)
hfyj=3
print, mad(rbin(sst[*,*,j],3),badval=0.0)/3.0, fpix[j]

stop

      for j=0,obj.length-1 do begin
         o=obj[i]
         obj.img  = ptr_new(img[obj.xmin:obj.xmax,obj.ymin:obj.ymax], /no_copy)
         obj.seg  = ptr_new(wht[obj.xmin:obj.xmax,obj.ymin:obj.ymax], /no_copy)

         detect_kron, op, kron_factor=par.phot_kron, bias_shrink=par.bias_shrink, $
                  bias_fraction=par.bias_fraction, blend_shrink=par.blend_shrink, $
                  minradius=par.phot_minradius/pscl
          detect_showstamp, op, minradius=par.phot_minradius/pscl, title='KRON='+str(par.phot_kron)


stop
      end
   end

; autopilot_galactic_extinction, par
; aperture correction outside

end

   ; get psf of detection image to get aperture correction
   ;   getklpsf, par.detect_image_weighted.replace('sci.fits','psf.sav'), od.x, od.y, detpsf
   ;   autopilot_get_kernel, froot[i]+'_kern.sav', sxpar(h,'NAXIS1')/2, sxpar(h,'NAXIS1')/2, kern, gx=gx, gy=gy, kernel=kernel
;      faper, photpsf, !null, !null, od.kron_major, inv_phot_apcor, elon=od.kron_major/od.kron_minor, theta=od.theta
   ;      detect_kron, o, kron_factor=1.0, minradius=par.phot_minradius/pscl
;         detect_showstamp, o, minradius=par.phot_minradius/pscl, title='KRON=1.0'

