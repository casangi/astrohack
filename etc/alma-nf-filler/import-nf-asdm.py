from alma_nf_filler_lib import asdm_to_holog, print_asdm_summary
from alma_nf_filler_lib import CALIBRATION_OPTIONS
import argparse
    
parser = argparse.ArgumentParser(description='Import an ALMA Near-Field ASDM to an AstroHACK .holog.zarr file')

parser.add_argument('nf_asdm', type=str,
                    help='Path to the root of the ALMA NF ASDM')
parser.add_argument('holog_name', type=str,
                    help='Name of the created AstroHACK file to be created '
                    '(no extension)')
parser.add_argument('-t', '--integ-time', type=float, default=None,
                    help='Integration time on the visibilities and pointings, '
                    'defaults to the largest interval of sampling '
                    '(pointing or total power)')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Print processing messages')
parser.add_argument('-q', '--print-and-quit', action='store_true', default=False,
                    help='Print ASDM summary and quit')
parser.add_argument('-p', '--phase_cal', type=str, default=CALIBRATION_OPTIONS[0],
                    help='Apply phase calibration of the specified type, "none" '
                    'means no phase cal', choices=CALIBRATION_OPTIONS)
parser.add_argument('-a', '--amplitude_cal', type=str,
                    default=CALIBRATION_OPTIONS[0], choices=CALIBRATION_OPTIONS,
                    help='Apply amplitude calibration of the specified type, '
                    '"none" means no amplitude cal')
parser.add_argument('-c', '--cal_cycle', type=int, default=3, help='How many '
                    'subscans in each calibration cycle, i.e. how many subscans '
                    'from one calibration to the next')
parser.add_argument('-pc', '--plot_cal', action='store_true', default=False,
                    help='Plot calibration to png files')
parser.add_argument('-s', '--save_cal', action='store_true', default=False,
                    help='Save calibration Xarray dataset')

fake_correlations = False
args = parser.parse_args()

if args.print_and_quit:
    print_asdm_summary(args.nf_asdm)
    exit()

asdm_to_holog(args.nf_asdm, args.holog_name, args.integ_time, args.verbose,
              fake_correlations, args.amplitude_cal, args.phase_cal,
              args.cal_cycle, args.plot_cal, args.save_cal)


