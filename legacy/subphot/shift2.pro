
pro shift2, p ,dp
  
 err = raper[0]*sqrt(!pi)*median(rms)
 flux = total(total(obj,1),1)*fpos
 

; halfstep
 ngridx = ceil((size(img2))[1]/(1.5*tsz))
 ngridy = ceil((size(img2))[2]/(1.5*tsz))
 x = (lindgen(ngridx+2))*(1.5*tsz)
 y = (lindgen(ngridy+2))*(1.5*tsz)
 
 indx = lindgen(ngridx+1)
 indy = lindgen(ngridy+1)

; nooooooooooo look at each tile and and fit all fluxes with one shift
; per tile

n = 0
pert= make_array(298,298,4000)

for i=0, n_elements(indx)-3 do begin &$
   for j=0, n_elements(indy)-3 do begin &$
     if n_elements(where( flux/err gt 20.0 and ll[1,*]+rxy[1,*] gt y[indy[j]] and ll[1,*]+rxy[1,*] lt y[indy[j]+2] and ll[0,*]+rxy[0,*] gt x[indx[i]] and ll[0,*]+rxy[0,*] lt x[indx[i]+2] )) gt 3 then begin &$
   tile = img2(x[indx[i]]:x[indx[i]+2],y[indy[j]]:y[indy[j]+2]) &$
   
   print, x[indx[i]], x[indx[i]+2], y[indy[j]], y[indy[j]+2], n_elements(where( flux/err gt 20.0 and ll[1,*]+rxy[1,*] gt y[indy[j]] and ll[1,*]+rxy[1,*] lt y[indy[j]+2] and ll[0,*]+rxy[0,*] gt x[indx[i]] and ll[0,*]+rxy[0,*] lt x[indx[i]+2])) &$
   print, size(tile) &$
   pert[*,*,n] = tile&$
   n+=1 &$
   end &$
   end &$
end





; -EXTRAct time img2   
                                ;   print, x[indx[i]-0.5],
                                ;   x[indx[i]+0.5] &$                                                                                                                                                                                   
                                ; -pick sources in location
                                ; -mkmodel in exracted tile with
                                ; sources inside mpfit, adjusting
                                ; tile shift and individual fluxes

 ;   print, x[indx[i]-0.5], x[indx[i]+0.5] &$    

 ;after selection of sources in grid
 if n_elements(where(flux/err gt 20.0)) ge 2 then begin
    
    parinfo = replicate({value:0.D, fixed:0, step:0.D, mpminstep:0.D, $
                         limited:[0,0], limits:[0.D,0]}, nfit+2)
    
    parinfo[*].value = [0.,0,lar0] ; initialize with shift 0, and the flux values from second fit
    parinfo[0:1].step = 1          ; make sure step size is large enough to measure LM differential
    parinfo[2:*].fixed = lar0/err lt 20
    parinfo[0:1].limited = 1
    parinfo[0:1].limits = [-maxshift,maxshift]
    print, 'MAXSHIFT ', [-maxshift,maxshift]
   
    if keyword_set(libnative) then begin
       eval_fcn = 'mpfit_eval'
       fargs = {libnative:libnative}
       print, 'NATIVE'
    end else eval_fcn = 'mpfit_eval_idl'
    
    p = mpfit(eval_fcn, ftol=1e-5, $
              parinfo=parinfo, STATUS=status, nfev=nfev, BESTNORM=chi2,$
              covar=covar, perror=perror, niter=niter, nfree=nfree, $
              npegged=npegged, dof=dof, ERRMSG=errmsg, /quiet,functargs=fargs)
    yfit = idl_sum3d(p,cube=cube) ; shifted model
    sh = p[0:1


 end else begin
 ; no bright sources.
    sh = [0.,0.]
 end
