res = readfits("example/out_doublepass2/phot/out_bgsub_res.fits")
;res= redfits("example/out_doublepass2/img/CDFS_CH1.fits")
seg = readfits("example/out_doublepass2/img/CDFS-1_Ks_v0.9.4_sci_seg.fits")
readcol, "example/out_doublepass2/phot/out_phot.cat",  ox,oy,oid,oftmpl,offit,oftmpl1,oforg1,of1,oe1,ofcor1,oecor1,oapcor1,ochi1,ores1,oadev1

res = readfits("example/out_doublepass2/phot/out_bgsub_res.fits")
seg = readfits("example/out_doublepass2/img/CDFS-1_Ks_v0.9.4_sci_seg.fits")
readcol, "example/out_doublepass2/phot/out_phot.cat",  ox,oy,oid,oftmpl,offit,oftmpl1,oforg1,of1,oe1,ofcor1,oecor1,oapcor1,ochi1,ores1,oadev1


;generate 10000 random points for faper measurements 
xe = randomu(seed, 10000)*3839+2000
ye = randomu(seed, 10000)*3541+3348
f = []

xet = []  
yet = []
;make faper measurements of random locations
for i=0, n_elements(xe)-1 do begin &$
   if seg[xe[i],ye[i]] eq 0. then begin &$
   faper, res, xe[i],ye[i], 6., f2, os=3 &$
   f= [f,[f2]] &$
   xet = [xet,[xe[i]]] &$
   yet = [yet,[ye[i]]] &$
   print, i &$
      end &$
end 

   xe = xet
   ye = yet

;calc seperation distance and get flux of nearest neighbor 
 sep =[]
 flux_nn=[]
 flux=f
 for i=0, n_elements(xe)-1 do sep = [sep,[sqrt((xe[i]-ox[(sort(sqrt((xe[i]-ox)^2+(ye[i]-oy)^2)))[0]])^2 +(ye[i]-oy[(sort(sqrt((xe[i]-ox)^2+(ye[i]-oy)^2)))[0]])^2 )]]
 for i=0, n_elements(xe)-1 do flux_nn = [flux_nn,[ofcor1[(sort(sqrt((xe[i]-ox)^2+(ye[i]-oy)^2)))[0]]]]

  fluxs      = flux[where(flux_nn gt 0.0014 and finite(flux) and finite(flux_nn) )]
  flux_nns =flux_nn[where(flux_nn gt 0.0014 and finite(flux) and finite(flux_nn) )]
  seps        = sep[where(flux_nn gt 0.0014 and finite(flux) and finite(flux_nn) )]



;   data = data[where(finite(fluxs))]
;  data = (21.581-2.5*alog10(flux_nns))
;   seps = seps[where(finite(fluxs))]
;   fluxs= fluxs[where(finite(fluxs))]



  fluxs[where(fluxs le 0.00006)] = 0.00006
  time = seps/6.66667
 
    data = (22.461-2.5*alog10(flux_nns))


  fluxs = (22.461-2.5*alog10(fluxs))
  fluxs[where(fluxs gt 27.0)] = 27.0
  fluxs[where(fluxs lt 23.0)] = 23.0


  cgDisplay
  !P.Multi=[0,1,1]
  
  elevColors = Byte(Round(cgScaleVector(fluxs, 0, 255)))
  cgLoadCT, 34

  xout=findgen(11)*0.6
  yout=findgen(11)*0.7+19
  points = 100
  triangulate, time[where(finite(data))], data[where(finite(data))], tr
  result = griddata(time[where(finite(data))], data[where(finite(data))], elevcolors[where(finite(data))], method='inversedistance',power=1, xout=xout, yout=yout, min_points=points,triangles=tr,/grid)
  

 ; cgplot, time, data, /NoData,yrange =[19,26], xrange=[0,6], XTitle='Seperation [arcsec]', YTitle='Mag_nn', title=plottitle
  cgcontour, result, xout, yout, nlevels=255, /fill,  XTitle='Seperation [arcsec]', YTitle='Magnitude of Nearest Neighbour';, Title= "10,000 Random Apertures Placed on Residual Image"
  cgplots, time[where(time lt 6 and data lt 26 and data gt 19)], data[where(time lt 6 and data lt 26 and data gt 19)], col=cgcolor("black"), psym="filledcircle", symsize=0.2
  
  cgColorbar, Divisions=5, NColors=255, Range=[min(fluxs),max(fluxs)],vertical=1,  $
              TLocation='right', Format='(f10.1)', Position=[0.2, 0.92, .85, 0.96], title="Residual Magnitude in Aperture"






