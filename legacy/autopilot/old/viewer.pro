forward_function mkmodel, fptv
pro viewer, cid, param=param, display=display, save=save
  time1 = systime(1)
  ;param = "../example/mophongo.param"
  readcol, param, key,f='a,a',COMMENT='#',DELIM='#',/SILENT
  for i=0,n_elements(key)-1 do r=execute(key[i])

  file_mkdir, outdir+"view/" 

  fimg1 = imdir+file_basename(det)
  fseg1 = imdir+file_basename(det,'.fits')+'_seg.fits'
  fimg2 = imdir+file_basename(phot)
  fexp2 = imdir+file_basename(phot,'.fits')+'_exp.fits'
  fcat  = outdir+file_basename(det,'.fits')+'.xy'

 ; iimage, phot[(x[i]-(tsz/2.0)):(x[i]+(tsz/2.0)),(y[i]-(tsz/2.0)):(y[i]+(tsz/2.0))]
  resolve_routine, "iphot",/compile 
  

  himg = headfits(fimg1)
  pixscl = pixscale(himg)
  photbin = floor(subphot_photbin/pixscl)

  imdet = readfits(fimg1,/silent,nanval=0)
  imphot = readfits(fimg2,/silent,nanval=0)
  seg1 = long(readfits(fseg1,/silent,nanval=0))

  res = readfits(outdir+"photometry/out_res.fits",/silent,nanval=0)
  tmpl = readfits(outdir+"photometry/out_model.fits",/silent,nanval=0)

  cat = (read_ascii(fcat)).(0)

  restore, outdir+"photometry/out_obj1.sav", /verbose
  restore, outdir+"photometry/messout.sav", /verbose


  ;###could put this in a loop for (size(cid))[0] > 1

  
  md = mkmodel(obj, fpos, (size(imphot))[1], (size(imphot))[2], ll, cid)
  clean = imphot - md
  
  naper = n_elements(raper)
  



  wx = 6*tsz
  wy = 4*tsz
  xoff = (indgen(3)*2)*tsz
  yoff = intarr(3)+tsz*2
                                ; location of aperture circles are
                                ; local template coordinates                                                                                                                             
                                ; for the phot images they are shifted
                                ; by the best-fit shift p[0:1]                                                                                                                       
  xc = rxy[0,cid]
  yc = rxy[1,cid]

  xo = 2*xc+0.5 + xoff
  yo = 2*yc+0.5




                                ;extrac is used rather than index
                                ;slicing because it pads areas outside
                                ;of the original array range with
                                ;zeros

  tphot = extrac(imphot,ll[0,cid],ll[1,cid],tsz,tsz)
  tseg = extrac(seg1, ll[0,cid],ll[1,cid],tsz,tsz)
  tdet = extrac(imdet, ll[0,cid],ll[1,cid],tsz,tsz)
  ttmpl = extrac(tmpl, ll[0,cid],ll[1,cid],tsz,tsz)
  tres = extrac(res, ll[0,cid],ll[1,cid],tsz,tsz)
  tclean = extrac(clean, ll[0,cid],ll[1,cid],tsz,tsz)

  pixfac = sqrt(2)
  na = floor(raper[0]*sqrt(3.14)/pixfac)
  tile_stat, rbin(tphot, na ), bgmed, bgrms1, nsig=2., minfrac=0.25, /grow
  prms = bgrms1/na
  scl =  subphot_nsigma*prms
  ;print, subphot_nsigma, prms, bgrms1, na, pixfac, scl




  min = -scl; median(tres)-sqrt(2)*median(tres)
  max = scl ;median(tres)+sqrt(2)*median(tres)
  
  
  idcat = cat[2,*]
  xcat = cat[0,*]
  ycat = cat[1,*]
  
 ; it = awhere(idcat, (tseg[uniq(tseg,sort(tseg))])[1:*], nt)
  it = (tseg[uniq(tseg,sort(tseg))])[1:*]
  idlis = long(idcat[it])
  xlis = xcat[it] - 1      ; FITS to IDL coords
  ylis = ycat[it] - 1

  nd = sort(sqrt((xc-xlis+ll[0,cid])^2 + (yc-ylis+ll[1,cid])^2))
  lv = [0.2,0.8,0.4,0.6,1.0]#replicate(1,ceil(n_elements(nd)/5.))
  tvseg = float(tseg)
  for i=0,n_elements(nd)-1 do tvseg[where(tseg eq idlis[nd[i]])] = lv[i]







  ;### use all images to make on tiled output.

  olddev = !d.name
  set_plot,'z'
  device, set_resolution=[wx,wy]
  !p.charsize=1.5
  cleanplot,/silent
  loadct,0, /silent

                                ; @@ first do an extra "dummy"
                                ; tvcircle, otherwise we get blank
                                ; output when first starting up. dont
                                ; know why.                                                                             
 

  tvcircle, 2*raper[0], xo-[1,0,0]*2, yo+yoff-[1,0,0]*2, linestyle=2, color=cgcolor('red')
  loadct,0, /silent
  fptv, tphot, os=2, bin=photbin, mm=[min,max]
  fptv, tdet,pos=1,os=2, fac=8
 
 ;### works to here so far
                                ; coloring of segmap; rotate through 5
                                ; graylevels in a list sorted to
                                ; distance                                                                                                             


 ; for i=0,n_elements(nd)-1 do tvseg[where(tseg eq idlis[nd[i]])] = lv[i]
 ; fptv, tvseg-0.1*mask, pos=2,os=2, mm=[-0.2,1]
 ; fptv, yfit, mm=[-scl,scl], pos=3,os=2, bin=photbin ; fit = model from leve-marq non-lin fit 
 ; fptv, (tphot_fit-yfit)*(1-mask2), mm=[-scl,scl], pos=4,os=2, bin=photbin
 ; fptv, (tphot-nn_img)*(1-mask2), mm=[-scl,scl], pos=5,os=2, bin=photbin


  fptv, tvseg, pos=2,os=2, mm=[-0.2,1]
  fptv, ttmpl, pos=3,os=2, bin=photbin, mm=[min,max] ; fit = model from leve-marq non-lin fit     
  fptv, tres, pos=4,os=2, bin=photbin, mm=[min,max]
  fptv, tclean, pos=5,os=2, bin=photbin, mm=[min,max]


    ;tvcircle, 2*raper[0], xo-[1,0,0]*p[0]*2, yo+yoff-[1,0,0]*p[1]*2, linestyle=2, color=cgcolor('red')
    ;tvcircle, 2*raper[0], xo-p[0]*2, yo-p[1]*[1,1,1]*2, linestyle=2, color=cgcolor('red')
    ;tvcircle, 2*raper[naper-1], xo-[1,0,0]*p[0]*2, yo+yoff-[1,0,0]*p[1]*2, linestyle=2, color=cgcolor('red')
    ;tvcircle, 2*raper[naper-1], xo-p[0]*2, yo-p[1]*[1,1,1]*2, linestyle=2, color=cgcolor('red')




  if not keyword_set(noaper) then begin  
    tvcircle, 2*raper[0], xo-[1,0,0]*2, yo+yoff-[1,0,0]*2, linestyle=2, color=cgcolor('red') 
    tvcircle, 2*raper[0], xo, yo-[1,1,1]*2, linestyle=2, color=cgcolor('red') 
    tvcircle, 2*raper[naper-1], xo-[1,0,0]*2, yo+yoff-[1,0,0]*2, linestyle=2, color=cgcolor('red') 
    tvcircle, 2*raper[naper-1], xo, yo-[1,1,1]*2, linestyle=2, color=cgcolor('red') 
  end

  ;### maybe make color of text red to be readable.
  
  xyouts, xoff+5, yoff+ tsz*2 -20, ['img','tmpl','seg'], /device, charsize=1.5, charthick=2
  xyouts, xoff+5, yoff-20, ['model','res','clean'], /device, charsize=1.5, charthick=2

                                ; read back from z-buffer and write to
                                ; png file                                                                                                                                           
  tvlct,r,g,b,/get
  rd = tvrd()
  diag = fltarr(3,wx,wy)
  diag[0,*,*] = r[rd]
 ; diag[1,*,*] = g[rd]
 ; diag[2,*,*] = b[rd]
  set_plot, olddev
  if keyword_set(display) then begin
     iimage, diag
  end
  if keyword_set(save) then begin
     write_png, outdir+'view/'+file_basename(phot,'.fits')+'_id_'+strtrim(cid,2)+'.png', diag
  end
  print, systime(1) - time1 ,' SECONDS.'
end



; @@@ merge with tvs and make separate                                                                                                                                                      
pro fptv, inimg, os=os, mm=mm,pos=pos, fac=fac, bin=bin
; on_error, 2                                                                                                                                                                               

  if not keyword_set(os) then os=1
  if not keyword_set(pos) then pos =0.
  if not keyword_set(fac) then fac=5.
  if not keyword_set(bin) then bin=1

  sz=(size(inimg))[1]
  img = inimg
  if keyword_set(bin) then begin
    bsz = sz/bin
    tmpimg = rebin(img[0:bsz*bin-1,0:bsz*bin-1],bsz,bsz)*sqrt(bin) ; keep same s/n                                                                                                          
    img[0:bsz*bin-1,0:bsz*bin-1] = rebin(tmpimg, bsz*bin,bsz*bin,/sample)
 end

  if keyword_set(mm) then begin
    tv, rebin(bytscl(img, min=mm[0], max=mm[1]), sz*os,sz*os,/sample) ,pos
 end else begin
    rms = robust_sigma(img)
    tv, rebin(bytscl(img-median(img), min=-fac*rms, max=fac*rms), sz*os,sz*os,/sample) ,pos
 end
end
