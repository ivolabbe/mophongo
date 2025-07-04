
; full fit not necessary... the centroiding mostly depends on the
function centroid_multipsf, p, err=err, model=model, obs=obs, det=det, mask=mask, stopme=stopme

   dim = model.dim
   xgrid = findgen(dim[0])-p[0]
   ygrid =  findgen(dim[1])-p[1]
   if n_elements(dim) eq 2 then dim = [dim,1]
   zgrid = lindgen(dim[2])

   mod_sh = interpolate(reform(model,dim), xgrid, ygrid, zgrid, /GRID, missing=0.0, cubic=-0.5)

   m = reform(mod_sh,dim[0]*dim[1],dim[2]) # p[2:*]
   res = (obs - m)/err

 ;  res *= apermask((res.dim)[0],min(res.dim)/2-1)
   res =  res[2:-3,2:-3]  ; trim edges

  ;   mod_sh = interpolate(reform(model,dim), zgrid, yzgrid, /GRID, missing=0.0, cubic=-0.5)
   ;if n_elements(p) gt 2 then res = (obs - p[2]*mod_sh)/err else res = (obs - mod_sh)/err

  ; if keyword_set(mask) then $
  ;    res *= interpolate(byte(obs ne 0.0),  xout, yout, /GRID, missing=0.0) ne 0

   if keyword_set(stopme) then begin
      loadct,0
      wset,0
      mm = reform(m,dim[0:1])
      tvs, res, mm=[-1,1]*3*mad(res)
      tvs, obs, pos=1
      tvs, mm, pos=2
      tvs, smooth(res,3), pos=3
      tvs, det, pos=4
      logger,'p:', p,  alog10(total(res^2))
      stopkey
   end

   return, reform(res, n_elements(res))
end
