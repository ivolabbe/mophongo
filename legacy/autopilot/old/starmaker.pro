
pro starmaker, param=param
resolve_routine, "iphot",/compile
forward_function mkmodel



;getklpsf, 'psf_CDFS-1_Ks_v0.9.4_sci.sav', gx, gy, tmplpsf
;getklpsf, 'psf_CDFS_CH1.sav', gx, gy, photpsf

 tsz=99.0
 
; x = []
; y = []
; readcol, "example/out_insert1/CDFS-1_Ks_v0.9.4_sci.xy", xc, yc, idc, org
 img2 = readfits("base.fits",header,/silent,nanval=0) 
 detim = readfits("example/out_doublepass/img/CDFS-1_Ks_v0.9.4_sci.fits", dethead)
 
 filename = "simg/varystar10phot.fits"
 detfile =  "simg/varystar10det.fits"
     readcol, "cats/newcat10.cat", id,ra,dec, format="a,a,a", comment="#"
     textout=  "fluxnewcat10.cat"

; x = [x,[xc[where(org gt 9349)]]]
; y = [y,[yc[where(org gt 9349)]]]
 adxy, header,ra[9349:-1], dec[9349:-1], px,py
 adxy, dethead,ra[9349:-1],dec[9349:-1], dx, dy




; get index of x,y in regular grid
;phot ll
 pix = long(px)
 piy = long(py)
 s2 = (tsz-1)/2

 pdxy = [1#(px-pix), 1#(py-piy)]     ; fractional pixel offset
 prxy = s2 + pdxy                 ; exact coordinates in tile
 pll = long([1#pix,1#piy]-s2)      ; coordinates of lowerleft pixel  (offset of tile in image)


;det ll
 dix = long(dx)
 diy = long(dy)
 s2 = (tsz-1)/2

 ddxy = [1#(dx-dix), 1#(dy-diy)]     ; fractional pixel offset                                                                                                                                       
 drxy = s2 + ddxy                 ; exact coordinates in tile                                                                                                                                          
 dll = long([1#dix,1#diy]-s2)      ; coordinates of lowerleft pixel  (offset of tile in image)                                                                                                                                                                                                                                                                                                                  



 getklpsf, 'example/out_doublepass/psf/psf_CDFS_CH1.sav', px, py, pobj
 getklpsf, 'example/out_doublepass/psf/psf_CDFS-1_Ks_v0.9.4_sci.sav', dx, dy, dobj
;fpos = flux because obj is normalized

 
 mag = randomu(seed,200)*7+19
 fpos = 10.0^(-(mag-21.581)/2.5)
 
 dfpos= fpos*166
 
 ;fpos = fltarr((size(pobj))[3])+0.12
 ;dfpos =fltarr((size(dobj))[3])+200.0

 

 basis = make_array([99],/index)
 pshifx= 49.5-prxy[0,*]
 pshify= 49.5-prxy[1,*]

 fposf = []
 dfposf = []

 pobj2 =fltarr(99,99,200)
 for i=0, 199 do begin 
    pobj2[*,*,i] = interpolate(pobj[*,*,i], basis-pdxy[0,i], basis-pdxy[1,i], missing=0.0, /grid) 
    fposf = [fposf,[total(fpos[i]*pobj2[*,*,i])]]
 end

 dobj2 =fltarr(99,99,200)
 for i=0, 199 do begin 
    dobj2[*,*,i] = interpolate(dobj[*,*,i], basis-ddxy[0,i], basis-ddxy[1,i], missing=0.0, /grid)  
    dfposf = [dfposf,[total(dfpos[i]*dobj2[*,*,i])]]
 end
 
 


 psz = size(img2)
 dsz = size(detim)

 pmodel = mkmodel(pobj2, fpos, psz[1], psz[2], pll)
 dmodel = mkmodel(dobj2, dfpos, dsz[1], dsz[2], dll)

 outphot = img2+pmodel
 outdet = detim+dmodel

 forprint, id[9349:-1],ra[9349:-1], dec[9349:-1],fposf,dfposf, textout=textout, comment="#ID RA DEC FPOS DFPOS", format="(I,F,F,F,F)"
 print, textout, " written"
 fits_write,filename, outphot, header
 print, filename, " written"
 fits_write, detfile, outdet, dethead
 print, filename, " written"
end



pro backgroundmaker, param=param
  
;base  image to which the background is applied
  base = readfits("base.fits", header)
;x and y pixel indices used axis
  basex = readfits("simages/basex.fits")
  basey = readfits("simages/basey.fits")

;pixel rms map output from getrmsmap
  brms = readfits("example/out_doublepass/phot/out_rms.fits")
;median of rms map from getrmsmap  
 twosig =  median(brms)
  print, "two sigma: ", twosig

;sinusoidal oscillation 
  snx = sin((20*!pi*basex)/7000.0)
  sny = sin((20*!pi*basey)/8400.0)

;write fits, scaled sin wave + base image 
  fits_write, "test2.fits", sny*snx*twosig+base, header

end

pro plotty, param =param

  data = sny*snx*twosig
  lala = readfits("example/out_doublepass/img/CDFS-1_Ks_v0.9.4_exp.fits")
  lala[where(lala gt 0.0)]=1.0
  
  dataf = temporary(data)/lala
  fit = readfits("example/out_cleanbg20/phot/out_bgmed.fits")
  rms = readfits("example/out_cleanbg20/phot/out_rms.fits")
;fit = readfits("example/out_cleanbg20/phot/out_bgmed.fits")
  fitf=temporary(fit)/lala
  rmsf=temporary(rms)/lala
;histo = adata/drms
                                ;adata= abs(data)                                                                                                                                                                                                                                                                                                                                                      

                                ;mad = meanabsdev(sub, /median)                                                                                                                                                                                                                                                                                                                                        


  sub = abs(temporary(dataf)-temporary(fitf))
  hist = temporary(sub)/temporary(rmsf)

  binsize= 0.005
  mad=hist[where(finite(hist))]
  cgHistoplot, abs(hist), /Frequency, /OProbability, ProbColor='red', binsize=binsize, ProbThick=thick,xrange=[0,0.5], PolyColor='dodger blue', DataColor='navy', /nan, xtitle="Absolute Deviation [pixel rms]", probability=prob, histdata=h, locations=loc


  print, prob[where(prob gt 0.898 and prob lt 0.91)]
  print, loc[where(prob gt 0.898 and prob lt 0.91)]
   
  cgplot, [0.000092,0.000092],[0.0,0.012], /overplot
  cgtext, 0.5, 0.5, /normal, "prob(0.9)= 9.2e-05"
end
