from astrohack.extract_locit import extract_locit
from astrohack.locit import locit
import argparse

desc = "Execute locit with a phase cal table produced by CASA\n\n"
desc += "This script executes a subset of locit's features, for a more detailed tutorial see:\n"
desc += "https://astrohack.readthedocs.io/en/stable/locit_tutorial.html"


parser = argparse.ArgumentParser(
    description=f"{desc}", formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument("caltable", type=str, help="Phase cal table")
parser.add_argument(
    "-d",
    "--display_plots",
    action="store_true",
    default=False,
    help="Display plots during script execution",
)
parser.add_argument(
    "-a",
    "--antennas",
    type=str,
    default="all",
    help="Comma separated list of antennas to " "be processed, default is all antennas",
)
parser.add_argument(
    "-c",
    "--combination",
    type=str,
    default="simple",
    help="How to combine different spws to for locit processing, valid values are: "
    "no, simple or difference, default is simple",
)
parser.add_argument(
    "-p",
    "--polarization",
    type=str,
    default="both",
    help="Which polarization hands to be used for locit processing, for the VLA options are: "
    "both, L or R, default is both",
)
parser.add_argument(
    "-k",
    "--fit_kterm",
    action="store_true",
    default=False,
    help="Fit antennas K term (i.e. Offset between azimuth and elevation axes)",
)
parser.add_argument(
    "-e",
    "--elevation_limit",
    default=10.0,
    help="Lowest elevation of data for consideration in degrees, default is 10",
)
parser.add_argument(
    "-f",
    "--fit_engine",
    type=str,
    default="scipy",
    help='Choose the fitting engine, default is "scipy" other available engine is "linear algebra"',
)
args = parser.parse_args()


def get_ant_list_from_input(user_ant_list):
    if "," in user_ant_list:
        # break it up into a list
        return user_ant_list.split(",")
    else:
        return user_ant_list


def get_cal_table_name(calname, cal_ext):
    if cal_ext in calname:
        return calname.replace(cal_ext, "")
    else:
        raise Exception('Cal table is badly formatted, it must end with "-pha.cal"')


caltable = args.caltable
ant_list = get_ant_list_from_input(args.antennas)


ext_cal = "-pha.cal"
ext_locit = ".locit.zarr"
ext_pos = ".position.zarr"

basename = get_cal_table_name(caltable, ext_cal)
cal_table = basename + ext_cal
locit_name = basename + ext_locit
position_name = basename + ext_pos
locit_plot_folder = position_plot_folder = f"{basename}+_locit_plots"


#################
# Extract locit #
#################
locit_mds = extract_locit(
    cal_table,  # The calibration table containing the phase gains
    locit_name=locit_name,  # The name for the created locit file
    ant="all",  # Antenna selection, None means 'All'
    ddi="all",  # DDI selection, None means 'ALL'
    overwrite=True,
)

#####################
# locit MDS exports #
#####################
locit_mds.print_array_configuration()
locit_mds.plot_source_positions(
    locit_plot_folder,  # destination for the plot
    labels=True,  # Display source labels on plot
    precessed=False,  # Plot FK5 (J2000) coordinates instead of precessed coordinates
    display=args.display_plots,
)

################
# Actual locit #
################
position_mds = locit(
    locit_name,
    position_name=position_name,  # Name of the position file to be created by locit
    elevation_limit=args.elevation_limit,  # Elevation under which no sources are considered
    polarization=args.polarization,  # Combine both R and L polarization phase gains for increased SNR
    fit_engine=args.fit_engine,  # Fit data using scipy
    fit_kterm=args.fit_kterm,  # Fit elevation axis offset
    fit_delay_rate=True,  # Fit delay rate
    ant=ant_list,  # Select all antennas
    ddi="all",  # Select all DDIs
    combine_ddis=args.combination,  # Combine delays from all DDIs to obtain a single solution with increased SNR
    parallel=False,  # Do fitting in parallel
    overwrite=True,  # Overwrite previously created position file
)

########################
# position MDS exports #
########################
position_mds.plot_delays(
    position_plot_folder,  # Folder to contain plot
    ant=ant_list,  # Selected antennas
    ddi="all",  # DDI selection irrelevant because we are combining DDIs
    time_unit="hour",  # Unit for observation duration
    angle_unit="deg",  # Unit for sky coordinates
    delay_unit="nsec",  # Unit for delays
    plot_model=True,  # Plot fitted delay model
    display=args.display_plots,
)

position_mds.export_locit_fit_results(
    position_plot_folder,  # Folder to contain antenna position corrections file
    ant=ant_list,  # selected antennas
    position_unit="m",  # Unit for the position corrections
    delay_unit="nsec",  # Unit for delays
    time_unit="hour",  # Unit for delay rate denominator
)


position_mds.plot_position_corrections(
    position_plot_folder,  # Folder to contain plot
    unit="km",  # Unit for the x and Y axes
    box_size=5,  # Size for the box containing the inner array
    scaling=250,  # scaling to be applied to corrections
    display=args.display_plots,
)
