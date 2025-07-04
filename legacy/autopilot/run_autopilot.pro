pro aa
  RESOLVE_ALL, CLASS=['source', 'mophongo' ] ; survey',
end

pro apr
  resolve_routine, ['klpsf','autopilot', 'autopilot_threshold','AUTOPILOT_PHOTO',$
  'mophongo__DEFINE',$
 'detect'+['_objects','_moments','_showstamp','_kron','_showseg','_region','_deblend','_resegment']], /either, /compile
  resolve_routine, ['source__define']
  resolve_routine, ['skysub'], /is_function
  stopkey
end

pro apc
  f = file_search('work/*', count=fcount)
  if fcount gt 0 then file_delete, f, /allow, /verbose
end

; THIS RUNS AUTOPILOT FOR GOODS-S dataset  ------------------------
pro apg
  apr
  autopilot, dir_image = '/Users/ivo/Astro/FIELDS/GOODS-S/HLF', dir_work = 'work_goodss'
end

; THIS RUNS AUTOPILOT FOR UDF data set  ------------------------
pro apx
  apr

  dir_image = ['/Users/ivo/Astro/FIELDS/HUDF/XDF/','/Users/ivo/Astro/FIELDS/GREATS/']
  detect_band  = ['f160w','f125w']
  phot_band = ['_f*_','*CH1*']
  prefix = ['60mas','GOODS-S']

  autopilot, dir_image=dir_image, detect_band=detect_band, phot_band=phot_band, prefix=prefix
end

pro apgc
  apc
  apg
end

pro apxc
  apc
  apx
end

