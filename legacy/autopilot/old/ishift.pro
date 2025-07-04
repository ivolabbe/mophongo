
; recenter and renormalize image
; assumes images are square                                                                                                                         
function ipsf_recenter, img, renorm=renorm, mask=mask, verbose=verbose, dxy = dxy, gauss=gauss
  sz = (size(img))[1]
  fwhm = sz/4.
  c = (sz-1)/2.
  
                                ; get center, shift, and check again
                                                                                
  if keyword_set(gauss) then gcntrd, img, c, c, xc0, yc0, fwhm, /silent else cntrd, img, c, c, xc0, yc0, fwhm, /silent
  dxy = [c - xc0, c - yc0]

  imout = interpolate(img, indgen(sz)-(c - xc0),indgen(sz)-(c - yc0), missing=0, /GRID,cubic=-0.3)
  
;  if keyword_set(gauss) then gcntrd, imout, c, c, xc, yc, fwhm else
;  cntrd, imout, c, c, xc, yc, fwhm, /silent
;  if keyword_set(verbose) then print, string('old(x,y) ->
;  new(x,y):
;  ', xc0, yc0,' ->', xc, yc,', image size:
;  ',sz,f='(a,2f7.2,a,2f7.2,a,f7.2)')                                                                                                               

  if keyword_set(mask) then begin
    mask = 1-fix(img eq 0)
    mask = shift(growmask(mask),(c - xc0),(c - yc0))
 end
  if keyword_set(renorm) then imout=imout/total(imout)  ; normalize
  if keyword_set(mask) then imout = imout * mask   ; mask
  return, imout
end


; get info on brightness of sources, and brightness relative to other
; sources in tile                                                                
pro ipsf_getinfo, img1, img2, seg1, x, y, id, s, info, dxy, flim=flim
  len = n_elements(x)
  x1=(round(x-1-(s-1)/2.))   ; coords of lower left pixel in working tile                                                                          
  y1=(round(y-1-(s-1)/2.))   ; @@note: convert FITS to IDL coords                                                                                  
  mkcoo, s, xco, yco, d=d
  d2 = fix(d le s/2-1)
  d8 = fix(d le s/8-1)
  faper, img1, x,y, [s/8.], f1, os=1.0
  faper, img2, x,y, [s/8.], f2, os=1.0
  flim1 = percentiles(f1[where(finite(f1))], value=flim/2.)
  flim2 = percentiles(f2[where(finite(f2))], value=flim)
  j = where(f1 gt flim1 and f2 gt flim2,nok)
  dxy = fltarr(2,len)
  info = fltarr(6,len)
  for i=0,nok-1 do begin
     timg1 = extrac(img1,x1[j[i]],y1[j[i]],s,s)
     timg2 = extrac(img2,x1[j[i]],y1[j[i]],s,s)
     tseg = extrac(seg1,x1[j[i]],y1[j[i]],s,s)

     maskid = (tseg eq id[j[i]])*d2
     maskall = growmask(tseg ne 0)*d2
     mask_nn = growmask(growmask((tseg eq id[j[i]] or tseg eq 0)))*d8
                                ;    mask_blend =
                                ;    mask_nn*growmask(tseg ne id[j[i]]
                                ;    and tseg ne 0)                                                                                 
                                ; ratio flux central object to flux in
                                ; all segmentation pixels                                                                                      
     fimg1 = total(timg1*maskid)
     fimg2 = total(timg2*maskid)
     if fimg2 le 0 then continue
     rimg1 = fimg1/total(timg1*maskall)
     rimg2 = fimg2/total(timg2*maskall)
     junk = ipsf_recenter(timg1*mask_nn,dxy=dxy1,gauss=gauss)
     junk = ipsf_recenter(timg2*mask_nn,dxy=dxy2,gauss=gauss)
     dxy[*,j[i]] =  dxy1-dxy2
     if rimg2 lt 0.8 then continue
   epts = Fit_Ellipse(where(maskid),xsize=s,ysize=s, center=center,axes=axes,orientation=orientation)
   ratio1 = min(axes)/max(axes)
   info[*,j[i]] = [id[j[i]], rimg1, rimg2, fimg1, fimg2, ratio1]
   end
                                ; set any 0 value to nan
  inan = where(info eq 0,nnan)
  if nnan gt 0 then  info[inan] = !values.f_nan
end




pro ipsf_seeshift, fpos, fmap, outname=outname, pfac=pfac
  if not keyword_set(pfac) then pfac = 100.
  if not keyword_set(outname) then outname=fmap+'.ps'
                                ; read in previous list of kernels +
                                ; shifts                                                                                                        
  readcol, fpos, x, y, id, dx, dy, res0, res1 ;, form='f,f,l,f,f,f,f,f'                                                                              

                                ; read shift grid and calculate
                                ; residuals wrp to smoothed grid, and
                                ; plot                                                                           
  readcol, fmap, gx, gy, gdx, gdy
 
; reform grids                                                                                                                                       
  xout = gx[uniq(gx,sort(gx))]
  yout = gy[uniq(gy,sort(gy))]
  nx = n_elements(xout)
  ny = n_elements(yout)
  xstep = (max(gx)-min(gx))/(nx-1.)
  ystep = (max(gy)-min(gy))/(ny-1.)
  gx = reform(gx,nx,ny)
  gy = reform(gy,nx,ny)
  gdx = reform(gdx,nx,ny)
  gdy = reform(gdy,nx,ny)
 
; get index of x,y in grid                                                                                                                           
  xs = (x-xout[0])/xstep
  ys = (y-yout[0])/ystep

                                ; ok this works interpol [vec, x, y],
                                ; x, y                                                                                                          
                                ; evaluates the bilinear interpolation
                                ; of each element of vec on x,y                                                                                
; gdxy = transpose([[[gdx]],[[gdy]]],[2,0,1])                                                                                                        
; dxy_ip = interpolate(gdxy, xs,ys)                                                                                                                  
  dx_ip = interpolate(gdx,xs,ys)
  dy_ip = interpolate(gdy,xs,ys)
  resx = dx - dx_ip
  resy = dy - dy_ip
 
; print,[dxy_ip[0,*],1#dx_ip,dxy_ip[0,*]-dx_ip]                                                                                                      
; print,[dxy_ip[1,*],1#dy_ip,dxy_ip[1,*]-dy_ip]                                                                                                      

  iok = where((gdx + gdy) ne 0.)
  printps,file=outname, xs=20, ys=25
  cleanplot, /silent
  pp=plotprefs()
  col=getcolor(/load)
  !p.multi=[0,1,1]
  xrange = minmax(x)
  yrange = minmax(y)
  dxrange = xrange[1]-xrange[0]
  dyrange = yrange[1]-yrange[0]
  plot,[0], xrange=xrange+[-0.12*dxrange,0.12*dxrange], yrange=yrange+[-0.12*dyrange,0.5*dyrange], /nodata,/iso, yst=1,xst=1
  arrow, x, y, x+dx*pfac, y+dy*pfac, /data, color=col.gray, HSIZE = !D.X_SIZE / 96.
  arrow, x, y, x+resx*pfac, y+resy*pfac, /data, color=col.blue, HSIZE = !D.X_SIZE / 96.
  arrow, gx[iok], gy[iok], (gx+gdx*pfac)[iok], (gy+gdy*pfac)[iok], /data, color=col.green, HSIZE = !D.X_SIZE / 96.

  irac_histogauss, resx, ax, pos=[0.2,0.75,0.5,0.9], _extra=extra, /overplot, /noerase, charsize=0.8, yst=2
  irac_histogauss, resy, ay, pos=[0.55,0.75,0.85,0.9], _extra=extra, /overplot, /noerase, charsize=0.8, yst=2
  printps,/reset
  
;  print,ax,ay                                                                                                                                       
;  print,robust_sigma(resx), robust_sigma(resy)                                                                                                      
  
end


pro ishift, param = param  
  args=command_line_args(count=count)
  if count gt 0 then param=args[0] else if n_elements(param) eq 0 then param='phot.param'
  readcol, param, key,f='a,a',COMMENT='#',DELIM='#',/SILENT
  for i=0,n_elements(key)-1 do r=execute(key[i])


  print, 'second shift pass'
  time1=systime(1)


  nimg1 = imdir+file_basename(det)
  nseg1 = imdir+file_basename(det,'.fits')+'_seg.fits'
  nimg2 = imdir+file_basename(phot)
  fcat = outdir+file_basename(det,'.fits')+'.xy'

  sz = tsz 
  beta=beta
  sm_shift=4
  sm_kern=5
  outname=imdir+"out"
  order=[5,4,2,1]
  flim=0.7
  frlim=0.75
  adlim=0.05
  klim=0.35 
  display=1 
  xysig=2
  max_shift=5.
 
   if not keyword_set(outname) then outname=file_basename((strsplit(fcat,'.',/extract))[0])
   if not keyword_set(xysig) then xysig = 3.   ; remove shifts that are 4 sigma outside distribution
   if not keyword_set(beta) then beta = [2.5,6.,sz/6.]  ; scales of basis, critical parameter  
   if not keyword_set(order) then order=[6,3,1]         ; orders of basis, critical parameter   
   if not keyword_set(flim) then flim = 0.75   ; only consider keep 75% brightest sources
   if not keyword_set(frlim) then frlim = 0.6   ; only consider sources that contain at least 75% of total flux in tile 
   if not keyword_set(klim) then klim = 0.15       ; maximum tot abs dif of kernel wrt local average
   if not keyword_set(adlim) then adlim = 0.1   ; maximum tot abs deviation of (img2 - img1_conv)
   if not keyword_set(sm_shift) then sm_shift=3 ; smooth shift in bins of 3x3
   if not keyword_set(sm_kern) then sm_kern=3   ; smooth kernel in bins of 3x3
   if not keyword_set(max_shift) then max_shift = 3 ; maximum shift in pixels
   if not keyword_set(epslim) then epslim = 0.8

   ;use reg images 
  img1 = readfits(nimg1,/silent,nanval=0)
  img2 = readfits(nimg2,/silent,nanval=0)
  seg1 = readfits(nseg1,/silent,nanval=0)
  readcol, fcat, x,y,id, form='f,f,l', /silent

  print, 'getting info'
  ipsf_getinfo, img1, img2, seg1, x, y, id, sz, info, dxy, flim=flim
                                ; select only brightest flim percent,
                                ; and that contai least frlim fraction
                                ; of flux in tile                                                         
  f1 = info[3,*]
  f2 = info[4,*]
  f2r = info[2,*]
  rr1 = info[5,*]
  flim1 = percentiles(f1[where(finite(f1))], value=flim/2.)
  flim2 = percentiles(f2[where(finite(f2))], value=flim)
  print,' selecting to flux limits flim1,flim2: ', flim1, flim2
  
; @@@ if previous pass exists, just use it.                                                                                                          
;if not (file_test(outname+'.shift_raw') or                                                                                                          
                                ; write shifts for bright sources:
                                ; should do a 'local' avg                                                                                         
; only keep a/b ratio gt 0.8                                                                                                                         
  iok = where((f1 gt flim1) and (f2 gt flim2) and (rr1 gt 0.8) and finite(total(dxy,1)) $
               and sqrt(dxy[0,*]^2+dxy[1,*]^2) lt max_shift,nok)
  dx = dxy[0,iok] &  dy = dxy[1,iok]
  triangulate, x[iok], y[iok], connect = tl
  mdx = fltarr(nok) &  mdy= fltarr(nok) & sdx = fltarr(nok) & sdy = fltarr(nok)
  for i=0,nok-1 do mdx[i] = median(dx[0,tl[tl[i]:tl[i+1]-1]]) ;  ; connectivity list including point itself
  for i=0,nok-1 do mdy[i] = median(dy[0,tl[tl[i]:tl[i+1]-1]]) ;  ; connectivity list including point itself
  for i=0,nok-1 do sdx[i] = stddev(dx[0,tl[tl[i]:tl[i+1]-1]]) ;  ; connectivity list including point itself                                       
  for i=0,nok-1 do sdy[i] = stddev(dy[0,tl[tl[i]:tl[i+1]-1]]) ;  ; connectivity list including point itself
  is = where(abs(dx - mdx) lt xysig*sdx and abs(dy-mdy) lt xysig*sdy, ns)


  prvec, x, y, id, dxy[0,*], dxy[1,*], info[1,*], info[2,*], info[3,*], info[4,*], $
     file=outname+'.shift_raw', hdr=' x y id dx dy rimg1 rimg2 fimg1 fimg2' , form='(2f10.2,i,6g12.4)'
  prvec, x[iok[is]], y[iok[is]], id[iok[is]], dx[is], dy[is], info[1,iok[is]], info[2,iok[is]], info[3,iok[is]], info[4,iok[is]], $
     file=outname+'.shift', hdr=' x y id dx dy rimg1 rimg2 fimg1 fimg2' , form='(2f10.2,i,6g12.4)'

  ;restore, '/Users/mark/mophongo/example/out_shift/psf/kern_CDFS-1_Ks_v0.9.4_sci_CDFS_CH1.sav', /verbose 
  restore, psfdir+'/kern_'+file_basename(det,'.fits')+'_'+file_basename(phot,'.fits')+'.sav', /verbose

  remap_xy, outname+'.shift', gx, gy, nsmooth=sm_shift
  ipsf_seeshift, outname+'.shift', outname+'.shift.map'

  help,/mem, out=mem
  m=float((strsplit(mem,/ex))[3])/1e6
  print, systime(1) - time1 ,' SECONDS, ', m,' MB MEMORY'
end
