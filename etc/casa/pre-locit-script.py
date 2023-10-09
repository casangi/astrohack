################################################################################
###                                                                          ###
###                         Suggested path forward                           ###
###                                                                          ###
################################################################################

# Fringe fitting a single source i.e. BlLac seems to be the best
# compromise between flagging and having reasonable solutions.
# However a few crap scans remain, specially 57, 117 & 120 have 360 phase
# spread because of their low elevation, 120 even being below the horizon

# The pointing ms name
msname = 'pointing-2015-10-04.ms'
# Root for the products of this script
nameroot = '2015-10-04'
# Scans to flag, for example due to low elevation
scanstoflag = '' # here no scans below 5 degrees
# Reference antenna
ref_ant = 'ea04' # ea02 would also be a good choice(both are very close)
# Fringe fit source
frg_src = '0319+415' # Seems to be the brightest, old friend from the 30m pnt
# Number of channels
n_chan = 64

# Moved antennas are: 1, 6, 7, 12, 17, 20, 26, 28

# Extensions
ext_pnt = '-pnt.ms'
ext_frg = '-frg.cal'
ext_avg = '-avg.ms'
ext_cal = '-pha.cal'

# The desired intent
pnt_intent = 'CALIBRATE_POINTING#ON_SOURCE'


################################################################################
# Here starts the script, DO NOT CHANGE BELOW
################################################################################

# file names
point_only = nameroot+ext_pnt
fring_cal = nameroot+ext_frg
avg_data = nameroot+ext_avg
gaincal_tab = nameroot+ext_cal

split(vis         = msname,     # Name of input visibility file
      outputvis   = point_only, # Name of output visibility file
      keepmms     = True,       # keep Multi MS
      field       = '',         # Select field using field id(s) or field name(s)
      spw         = '',         # Select spectral window/channels
      scan        = '',         # Scan number range
      antenna     = '',         # Select data based on antenna/baseline
      correlation = '',         # Select data based on correlation
      timerange   = '',         # Select data based on time range
      intent      = pnt_intent, # Select observing intent
      array       = '',         # Select (sub)array(s) by array ID number.
      uvrange     = '',         # Select data by baseline length.
      observation = '',         # Select by observation ID(s)
      feed        = '',         # Multi-feed numbers: Not yet implemented.
      datacolumn  = 'data',     # Which data column(s) to process.
      keepflags   = True,       #
      width       = 1,          # Number of channels to average
      timebin     = '0s',       # Bin width for time averaging
)

if len(scanstoflag) > 0:
    flagdata(vis          = point_only,       # Name of input visibility file
             mode         = 'manual',         # Flagging mode
             autocorr     = False,            # Flag only the auto-correlations?
             spw          = '',               # Select spectral window/channels
             field        = '',               # Select field id(s) or field name(s)
             antenna      = '',               # Select data based on antenna/baseline
             uvrange      = '',               # Select data by baseline length.
             timerange    = '',               # Select data based on time range
             correlation  = '',               # Select data based on correlation
             scan         = scanstoflag,      # Scan number range
             intent       = '',               # Select observing intent
             array        = '',               # (Sub)array numbers
             observation  = '',               # Select by observation ID(s)
             action       =  'apply',         # Action to perform 
             display      = 'report',         # Display (data/report/both).
             flagbackup   = False,            # Save in flagversions
             savepars     = False,            # Save  parameters to the FLAG_CMD
             writeflags   = True,             # Do not modify.
             )
    # Create our new flag state under the name 'baseflags'
    flagmanager(vis=point_only, mode='save', versionname='baseflags')


    flagmanager(vis=point_only, mode='restore', versionname='baseflags')

    
fringefit(vis         = point_only,  # Name of input visibility file
          caltable    = fring_cal,   # Name of output gain calibration table
          field       = frg_src,     # field names
          solint      = 'inf',       # Solution interval: egs. 'inf', '60s'
          refant      = ref_ant,     # Reference antenna name(s)
          minsnr      = 3.0,         # Reject solutions below this snr
          zerorates   = True,        # Zero delay-rates in solution table
          globalsolve = True,        # Refine estimates with global lst-sq solver
          niter       = 100,         # Maximum number of iterations 
          corrdepflags= False,       # Respect correlation-dependent flags
          paramactive = [],          # Control which parameters are solved for
          parang      = False,       # Apply para angle correction on the fly
)

applycal(vis=point_only,
         gaintable=[fring_cal],
         interp=['nearest'],
         parang=False)


# Now we create a new dataset that is colapsed on the channel axis
# within each spw, also create a flagversion to store current flag
# state on the averaged MS
split(vis         = point_only,    # Name of input visibility file
      outputvis   = avg_data,      # Name of output visibility file
      keepmms     = True,          # keep Multi-ms
      field       = '',            # Select field  field name(s)
      spw         = '',            # Select spectral window/channels
      scan        = '',            # Scan number range
      antenna     = '',            # Select data based on antenna/baseline
      correlation = '',            # Select data based on correlation
      timerange   = '',            # Select data based on time range
      intent      = '',            # Select observing intent
      array       = '',            # Select (sub)array(s) by array ID number.
      uvrange     = '',            # Select data by baseline length.
      observation = '',            # Select by observation ID(s)
      feed        = '',            # Multi-feed numbers: Not yet implemented.
      datacolumn  = 'corrected',   # Which data column(s) to process.
      keepflags   = False,         #
      width       = n_chan,        # Number of channels to average
      timebin     = '0s',          # Bin width for time averaging
)
flagmanager(vis=avg_data, mode='save', versionname='original')


flagmanager(vis=avg_data, mode='restore', versionname='original')
gaincal(vis        = avg_data,    # Name of input visibility file
        caltable   = gaincal_tab, # Name of output gain cal table
        field      = '',          # Select field using field name(s)
        solint     = '10min',     # Solution interval
        refant     = ref_ant,     # Reference antenna name(s)
        refantmode = 'flex',      # Reference antenna mode
        minblperant= 3,           # Minimum baselines _per antenna_
        minsnr     = 3.0,         # Reject solutions below this SNR
        gaintype   = 'G',         # G is gain
        calmode    = 'p',         # Type of solution" ('ap', 'p', 'a')
        parang     = False,       # Apply parallactic angle correction
        solmode    = 'L1'         # Solution mode L1 => least squares
)

applycal(vis=avg_data,
         gaintable=[gaincal_tab],
         interp=['nearest'],
         parang=False)

