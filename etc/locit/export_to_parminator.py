from astrohack import open_position
import argparse

desc = "Export position corrections to parminator\n\n"
desc += "This script executes a subset of locit's features, for a more detailed tutorial see:\n"
desc += "https://astrohack.readthedocs.io/en/stable/locit_tutorial.html"


parser = argparse.ArgumentParser(
    description=f"{desc}", formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    "position_file", type=str, help="position.zarr file produced by locit"
)
parser.add_argument(
    "parminator_file", type=str, help="Name for the output parminator file"
)
parser.add_argument(
    "-t",
    "--correction_threshold",
    type=float,
    default=0.01,
    help="Threshold for including corrections in meters, default is 0.01",
)
parser.add_argument(
    "-a",
    "--antennas",
    type=str,
    default="all",
    help="Comma separated list of antennas to " "be processed, default is all antennas",
)
args = parser.parse_args()


def get_ant_list_from_input(user_ant_list):
    if "," in user_ant_list:
        # break it up into a list
        return user_ant_list.split(",")
    else:
        return user_ant_list


ant_list = get_ant_list_from_input(args.antennas)

position_mds = open_position(args.position_file)
position_mds.export_results_to_parminator(
    args.parminator_file, ant=ant_list, correction_threshold=args.correction_threshold
)
