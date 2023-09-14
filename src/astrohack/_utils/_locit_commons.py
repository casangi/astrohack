from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from astrohack._utils._panel_classes.telescope import Telescope
from astrohack._utils._constants import *
from astrohack.client import _get_astrohack_logger


def _open_telescope(telname):
    """Open correct telescope based on the telescope string"""
    if 'VLA' in telname:
        telname = 'VLA'
    elif 'ALMA' in telname:
        telname = 'ALMA_DA'  # It does not matter which ALMA layout since the array center is the same
    telescope = Telescope(telname)
    return telescope


def _compute_antenna_relative_off(antenna, tel_lon, tel_lat, tel_rad, scaling=1.0):
    antenna_off_east = tel_rad * (antenna['longitude'] - tel_lon) * np.cos(tel_lat)
    antenna_off_north = tel_rad * (antenna['latitude'] - tel_lat)
    antenna_off_ele = antenna['radius'] - tel_rad
    antenna_dist = np.sqrt(antenna_off_east ** 2 + antenna_off_north ** 2 + antenna_off_ele ** 2)
    return antenna_off_east * scaling, antenna_off_north * scaling, antenna_off_ele * scaling, antenna_dist * scaling


def _get_telescope_lat_lon_rad(telescope):
    if telescope.array_center['refer'] == 'ITRF':
        lon = telescope.array_center['m0']['value']
        lat = telescope.array_center['m1']['value']
        rad = telescope.array_center['m2']['value']
    else:
        logger = _get_astrohack_logger()
        msg = f'Unsupported telescope position reference :{telescope.array_center["refer"]}'
        logger.error(msg)
        raise Exception(msg)

    return lon, lat, rad


def _create_figure_and_axes(figure_size, boxes, default_figsize=figsize):
    if figure_size is None or figure_size == 'None':
        fig, axes = plt.subplots(boxes[0], boxes[1], figsize=default_figsize)
    else:
        fig, axes = plt.subplots(boxes[0], boxes[1], figsize=figure_size)

    return fig, axes


def _plot_antenna_position(outerax, innerax, xpos, ypos, text, box_size, marker='+', color='black'):
    half_box = box_size/2
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.plot(xpos, ypos, marker=marker, color=color)
        outerax.text(xpos, ypos, text, fontsize=fontsize, ha='left', va='center')
    else:
        innerax.plot(xpos, ypos, marker=marker, color=color)
        innerax.text(xpos, ypos, text, fontsize=fontsize, ha='left', va='center')


def _plot_corrections(outerax, innerax, xpos, ypos, xcorr, ycorr, box_size, color='red', linewidth=0.5):
    half_box = box_size/2
    size = np.sqrt(xcorr**2 + ycorr**2)
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.arrow(xpos, ypos, xcorr, ycorr, color=color, linewidth=linewidth, head_width=size/4)
    else:
        innerax.arrow(xpos, ypos, xcorr, ycorr, color=color, linewidth=linewidth, head_width=size/4)


def _plot_boxes_limits_and_labels(outerax, innerax, xlabel, ylabel, box_size, outertitle, innertitle, marker='x',
                                  marker_color='blue', rectangle_color='red', ):
    half_box = box_size/2.
    x_lim, y_lim = outerax.get_xlim(), outerax.get_ylim()
    x_half, x_mid = (x_lim[1] - x_lim[0])/2, (x_lim[1] + x_lim[0]) / 2
    y_half, y_mid = (y_lim[1] - y_lim[0])/2, (y_lim[1] + y_lim[0]) / 2
    if x_half > y_half:
        y_lim = [y_mid-x_half, y_mid+x_half]
    else:
        x_lim = [x_mid-y_half, x_mid+y_half]
    outerax.set_xlim(x_lim)
    outerax.set_ylim(y_lim)
    outerax.set_xlabel(xlabel)
    outerax.set_ylabel(ylabel)
    outerax.plot(0, 0, marker=marker, color=marker_color)
    box = Rectangle((-half_box, -half_box), box_size, box_size, linewidth=0.5, edgecolor=rectangle_color,
                    facecolor='none')
    outerax.add_patch(box)
    outerax.set_title(outertitle)
    outerax.set_aspect(1)

    # Smaller box limits and labels
    innerax.set_xlim([-half_box, half_box])
    innerax.set_ylim([-half_box, half_box])
    innerax.set_xlabel(xlabel)
    innerax.set_ylabel(ylabel)
    innerax.plot(0, 0, marker=marker, color=marker_color)
    innerax.set_title(innertitle)
    innerax.set_aspect(1)


def _close_figure(figure, title, filename, dpi, display, tight_layout=True):
    if title is not None:
        figure.suptitle(title)
    if tight_layout:
        figure.tight_layout()
    plt.savefig(filename, dpi=dpi)
    if not display:
        plt.close()
    return

