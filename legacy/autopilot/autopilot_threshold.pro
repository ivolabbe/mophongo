function auto_sample_background, img, tsz=tsz, len=len, frac=frac
   if not keyword_set(tsz) then tsz = mean(round(sqrt(img.dim)))
   if not keyword_set(len) then len = product(round(sqrt(img.dim)))
   if keyword_set(frac) then flen = round(len*frac) else flen = round(len)

   ; draw more as we reject locations within tsz/2 of border
   if len lt 1000 then buf = 1.5 else buf = 1.1  ;   ratio((sqrt(imask.length)+[0,-2*tsz])^2)
    dim = mask.dim
   imask = where(mask)

   bs,  where(mask), iloc, len=len*buf, seed=seed
   ix = iloc mod dim[0]
   iy = iloc / dim[0]
   xmin = min(ix,max=xmax)  ; defines border
   ymin = min(iy,max=ymax)

   iok = where(ix gt xmin + tsz/2 and ix lt xmax - tsz/2 and iy gt ymin + tsz/2 and iy lt ymax - tsz/2)
   iok = iok[0:len-1]

   xc = ix[iok] + randomu(seed,iok.length) - 0.5   ; @@@ shouldnt the random phase be -0.5 - 0.5 ?
   yc = iy[iok] + randomu(seed,iok.length) - 0.5   ; @@@ shouldnt the random phase be -0.5 - 0.5 ?

;  getstamps ...

  ; put in sq image here ....

   return, [1#xc, 1#yc]
end


function autopilot_threshold, det, minarea=minarea, spurious=spurious, detneg=detneg, stopme=stopme, verbose=verbose, showme=showme
  if n_elements(spurious) eq 0 then spurious=0.01
  if not keyword_set(minarea) then minarea = 5
  if not keyword_set(rlim) then rlim = 2.0
  if not keyword_set(badval) then badval=0.0

  sig = mad(det,badval=0.0)
  sz = size(det,/dim)
  level = linspace(4,1.0,nstep=20,/log)*sig
  logger, 'minarea=',minarea,' max spurious '+spurious.s()

  ; @@ if this quits after 1 iteration, something serious amiss
  for i=0,level.length-1 do begin
     np = detect_objects(det.x(sz/3,sz/3), detect_thresh=level[i], detect_minarea=minarea, /quick)
     nn = detect_objects(keyword_set(detneg) ? detneg.x(sz/3,sz/3) : -det.x(sz/3,sz/3),  detect_thresh=level[i], detect_minarea=minarea, nobj=nobj, /quick)
     nneg =  append(nneg, nn)
     npos =  append(npos, np)
     logger, i, level[i]/sig, level[i], np-nn, nn, 1.0*nn/(np-nn), /debug
     if 1.0*nn/(np-nn) gt (10*spurious < 0.2) then break
   end
   if i lt 2 then logger, ' quits first pass after ',i,' iteration ', /error

   nnok = npos-nneg
   lev0 = interpol(level[0:nneg.length-1], 1.0*nneg/nnok, spurious)
   logger, 'level pass1: lev0, /debug
   level = linspace(lev0*1.15,lev0*0.85,nstep=6,/log)
   delvarx, nneg, npos, nnok

  for i=0,level.length-1 do begin
     np = detect_objects(det, detect_thresh=level[i], detect_minarea=minarea, /quick)
     nn = detect_objects(keyword_set(detneg) ? detneg.x(sz/3,sz/3) : -det.x(sz/3,sz/3), detect_thresh=level[i], detect_minarea=minarea, /quick)
     nneg =  append(nneg, nn)
     npos =  append(npos, np)
     logger, print, i, level[i]/sig, level[i], np-nn, nn, '   -', 1.0*nn/(np-nn), /debug
   end ; while

   nnok = npos-nneg
   is = sort(level)
   lev1 = interpol(level, 1.0*nneg/nnok, spurious)
   nok1 = interpol(nnok, level, lev1)
   neg1 = interpol(nneg, level, lev1)

   if keyword_set(showme) then begin
      cgplot, level[is]/sig, npos[is], psym=-16, col='forestgreen', yrange=[0,max(npos)], xtit='sigma_bg',ytit='spurious',title='threshold'
      cgplot, level[is]/sig, nneg[is], psym=-16,/overplot, col='red'
      cgplot, level[is]/sig, nnok[is], psym=-16,/overplot, col='blue'
      cgplot, [1,1]*lev0/sig, [0,1e6], linest=1, col='black',/overplot, thick=3
      cgplot, [1,1]*lev1/sig, [0,1e6], linest=2, col='black',/overplot, thick=3

      cgplot, level[is]/sig, (1.0*nneg/nnok)[is], psym=-16, col='red', yr=[0,10*spurious], xtit='sigma_bg',ytit='spurious', title='threshold'
      cgplot, [1,1]*lev0/sig, [0,1e6], linest=1, col='black',/overplot
      cgplot, [1,1]*lev1/sig, [0,1e6], linest=2, col='black',/overplot, thick=3
   end
   if keyword_set(stopme) then stopkey

   logger, 'auto level:', lev1, /info
   return, lev1
end

