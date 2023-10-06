msname = 'redux-scan-63-avg.ms'
rootname = 'redux-scan-63'

ref_antenna = 'ea28'
antenna = 'ea06'
# AIPS
correction = [-0.0089,  0.0117, -0.0057]
# ASTROHACK
correction = [-0.00079, 0.0097, -0.0015]

################################################################################
###                                                                          ###
################################################################################
posext = '-pos.cal'

poscaltb = rootname+posext
splitms = rootname+'-split.ms'


gencal(vis       = msname,  
       caltable  = poscaltb,
       caltype   = 'antpos',
       antenna   = antenna,
       parameter = correction,
       )

applycal(vis       = msname,
         gaintable = poscaltb,
         applymode = ''
         )
