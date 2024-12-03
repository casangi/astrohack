import argparse
import casatools
import numpy as np

print()
print(80 * '-')
print()
desc = 'CASA pre-locit script\n'
desc += "Execute fringe fit, averaging and phase cal to produce the cal table to ingested by astrohack's locit"

parser = argparse.ArgumentParser(description=f'{desc}', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('input_dataset', type=str, help='Input dataset usually an ASDM')
parser.add_argument('output_name', type=str, help='Base name for output name produced by this script')
parser.add_argument('-m', '--is_ms', default=False, action='store_true', help='Input file is a MS rather than an ASDM')
parser.add_argument('-f', '--fringe_fit_source', default='0319+415',
                    help='Fringe fit source, default is 0319+415')
parser.add_argument('-r', '--reference_antenna', default='ea23',
                    help='Choose reference antenna for fringe fit and phase cal, default is ea23')
parser.add_argument('-s', '--scans_to_flag', default='', type=str,
                    help='Comma separated list of scans to flag, default is no scan to flag')
args = parser.parse_args()

print('CASA pre-locit script')
print()

# Input file names
if args.is_ms:
    ms_name = args.input_dataset
    asdm_name = ''
else:
    asdm_name = args.input_dataset
    ms_name = args.output_name + '.ms'

basename = args.output_name
scanstoflag = ''
ref_ant = args.reference_antenna
frg_src = args.fringe_fit_source
pnt_intent = 'CALIBRATE_POINTING#ON_SOURCE'

# Number of channels
n_chan = 64

# Name definitions
ext_pnt = '-pnt.ms'
ext_frg = '-frg.cal'
ext_avg = '-avg.ms'
ext_cal = '-pha.cal'
point_only = basename + ext_pnt
fring_cal = basename + ext_frg
avg_data = basename + ext_avg
gaincal_tab = basename + ext_cal

# The desired intent

if args.is_ms:
    pass
else:
    importasdm(asdm=asdm_name, vis=ms_name)

msmd = casatools.msmetadata()
msmd.open(ms_name)
spws = msmd.datadescids()
field_names = msmd.fieldnames()
nchans = []
for spw_id in spws:
    nchans.append(len(msmd.chanfreqs(spw_id)))
ant_names = msmd.antennanames()
msmd.done()

unq_chan = np.unique(nchans)
if unq_chan.size != 1:
    raise Exception('SPWs do not have a consistent number of channels')
else:
    nchan = unq_chan[0]

if frg_src not in field_names:
    print()
    print('Available sources for fringe fitting:')
    print(field_names)
    print()
    raise Exception('Chosen fringe fit source is not available in dataset')

if ref_ant not in ant_names:
    print()
    print('Available antennas:')
    print(ant_names)
    print()
    raise Exception('Chosen reference antenna is not available in dataset')

print('Splitting out relevant scans...')
split(vis=ms_name,  # Name of input visibility file
      outputvis=point_only,  # Name of output visibility file
      keepmms=True,  # keep Multi MS
      field='',  # Select field using field id(s) or field name(s)
      spw='',  # Select spectral window/channels
      scan='',  # Scan number range
      antenna='',  # Select data based on antenna/baseline
      correlation='',  # Select data based on correlation
      timerange='',  # Select data based on time range
      intent=pnt_intent,  # Select observing intent
      array='',  # Select (sub)array(s) by array ID number.
      uvrange='',  # Select data by baseline length.
      observation='',  # Select by observation ID(s)
      feed='',  # Multi-feed numbers: Not yet implemented.
      datacolumn='data',  # Which data column(s) to process.
      keepflags=True,  #
      width=1,  # Number of channels to average
      timebin='0s',  # Bin width for time averaging
      )

if len(scanstoflag) > 0:
    print('Flagging out problematic scans...')
    flagdata(vis=point_only,  # Name of input visibility file
             mode='manual',  # Flagging mode
             autocorr=False,  # Flag only the auto-correlations?
             spw='',  # Select spectral window/channels
             field='',  # Select field id(s) or field name(s)
             antenna='',  # Select data based on antenna/baseline
             uvrange='',  # Select data by baseline length.
             timerange='',  # Select data based on time range
             correlation='',  # Select data based on correlation
             scan=scanstoflag,  # Scan number range
             intent='',  # Select observing intent
             array='',  # (Sub)array numbers
             observation='',  # Select by observation ID(s)
             action='apply',  # Action to perform
             display='report',  # Display (data/report/both).
             flagbackup=False,  # Save in flagversions
             savepars=False,  # Save  parameters to the FLAG_CMD
             writeflags=True,  # Do not modify.
             )
    # Create our new flag state under the name 'baseflags'
    flagmanager(vis=point_only, mode='save', versionname='baseflags')

print('Running fringe fit...')
fringefit(vis=point_only,  # Name of input visibility file
          caltable=fring_cal,  # Name of output gain calibration table
          field=frg_src,  # field names
          solint='inf',  # Solution interval: egs. 'inf', '60s'
          refant=ref_ant,  # Reference antenna name(s)
          minsnr=3.0,  # Reject solutions below this snr
          zerorates=True,  # Zero delay-rates in solution table
          globalsolve=True,  # Refine estimates with global lst-sq solver
          niter=100,  # Maximum number of iterations
          corrdepflags=False,  # Respect correlation-dependent flags
          paramactive=[],  # Control which parameters are solved for
          parang=False,  # Apply para angle correction on the fly
          )

print('Applying fringe fit results...')
applycal(vis=point_only,
         gaintable=[fring_cal],
         interp=['nearest'],
         parang=False)

# Now we create a new dataset that is colapsed on the channel axis
# within each spw, also create a flagversion to store current flag
# state on the averaged MS
print('Averaging all channels...')
split(vis=point_only,  # Name of input visibility file
      outputvis=avg_data,  # Name of output visibility file
      keepmms=True,  # keep Multi-ms
      field='',  # Select field  field name(s)
      spw='',  # Select spectral window/channels
      scan='',  # Scan number range
      antenna='',  # Select data based on antenna/baseline
      correlation='',  # Select data based on correlation
      timerange='',  # Select data based on time range
      intent='',  # Select observing intent
      array='',  # Select (sub)array(s) by array ID number.
      uvrange='',  # Select data by baseline length.
      observation='',  # Select by observation ID(s)
      feed='',  # Multi-feed numbers: Not yet implemented.
      datacolumn='corrected',  # Which data column(s) to process.
      keepflags=False,  #
      width=n_chan,  # Number of channels to average
      timebin='0s',  # Bin width for time averaging
      )
flagmanager(vis=avg_data, mode='save', versionname='original')

print('Computing phase gains...')
gaincal(vis=avg_data,  # Name of input visibility file
        caltable=gaincal_tab,  # Name of output gain cal table
        field='',  # Select field using field name(s)
        solint='10min',  # Solution interval
        refant=ref_ant,  # Reference antenna name(s)
        refantmode='flex',  # Reference antenna mode
        minblperant=3,  # Minimum baselines _per antenna_
        minsnr=3.0,  # Reject solutions below this SNR
        gaintype='G',  # G is gain
        calmode='p',  # Type of solution" ('ap', 'p', 'a')
        parang=False,  # Apply parallactic angle correction
        solmode='L1'  # Solution mode L1 => least squares
        )

print('Applying phase gains...')
applycal(vis=avg_data,
         gaintable=[gaincal_tab],
         interp=['nearest'],
         parang=False)

# plotms(vis=avg_data, yaxis='phase', ydatacolumn='data', field='*', avgtime='10', correlation='RR,LL', coloraxis='spw',
#        antenna=ref_ant, iteraxis='baseline')
