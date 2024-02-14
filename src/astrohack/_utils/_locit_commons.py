from matplotlib.patches import Rectangle

from astrohack._utils._constants import *

import graphviper.utils.logger as logger


def _compute_antenna_relative_off(antenna, tel_lon, tel_lat, tel_rad, scaling=1.0):
    """
    Computes an antenna offset to the array center
    Args:
        antenna: Antenna information dictionary
        tel_lon: array center longitude
        tel_lat: array center latitude
        tel_rad: array center's distance to the center of the earth
        scaling: scale factor

    Returns:
    Offset to the east, Offset to the North, elevation offset and distance to array center
    """
    antenna_off_east = tel_rad * (antenna['longitude'] - tel_lon) * np.cos(tel_lat)
    antenna_off_north = tel_rad * (antenna['latitude'] - tel_lat)
    antenna_off_ele = antenna['radius'] - tel_rad
    antenna_dist = np.sqrt(antenna_off_east ** 2 + antenna_off_north ** 2 + antenna_off_ele ** 2)
    return antenna_off_east * scaling, antenna_off_north * scaling, antenna_off_ele * scaling, antenna_dist * scaling


def _get_telescope_lat_lon_rad(telescope):
    """
    Return array center's latitude, longitude and distance to the center of the earth based on the coordinate reference
    Args:
        telescope: Telescope object

    Returns:
    Array center  latitude, longitude and distance to the center of the Earth in meters
    """
    if telescope.array_center['refer'] == 'ITRF':
        lon = telescope.array_center['m0']['value']
        lat = telescope.array_center['m1']['value']
        rad = telescope.array_center['m2']['value']
    else:

        msg = f'Unsupported telescope position reference :{telescope.array_center["refer"]}'
        logger.error(msg)
        raise Exception(msg)

    return lon, lat, rad


def _plot_antenna_position(outerax, innerax, xpos, ypos, text, box_size, marker='+', color='black'):
    """
    Plot an antenna to either the inner or outer array boxes
    Args:
        outerax: Plotting axis for the outer array box
        innerax: Plotting axis for the inner array box
        xpos: X antenna position (east-west)
        ypos: Y antenna position (north-south)
        text: Antenna label
        box_size: Size of the inner array box
        marker: Antenna position marker
        color: Color for the antenna position marker
    """
    half_box = box_size / 2
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.plot(xpos, ypos, marker=marker, color=color)
        outerax.text(xpos, ypos, text, fontsize=fontsize, ha='left', va='center')
    else:
        outerax.plot(xpos, ypos, marker=marker, color=color)
        innerax.plot(xpos, ypos, marker=marker, color=color)
        innerax.text(xpos, ypos, text, fontsize=fontsize, ha='left', va='center')


def _plot_corrections(outerax, innerax, xpos, ypos, xcorr, ycorr, box_size, color='red', linewidth=0.5):
    """
    Plot an antenna position corrections as a vector to the antenna position
    Args:
        outerax: Plotting axis for the outer array box
        innerax: Plotting axis for the inner array box
        xpos: X antenna position (east-west)
        ypos: Y antenna position (north-south)
        xcorr: X axis correction (horizontal on plot)
        ycorr: Y axis correction (vectical on plot)
        box_size: inner array box size
        color: vector color
        linewidth: vector line width
    """
    half_box = box_size / 2
    head_size = np.sqrt(xcorr ** 2 + ycorr ** 2) / 4
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.arrow(xpos, ypos, xcorr, ycorr, color=color, linewidth=linewidth, head_width=head_size)
    else:
        outerax.arrow(xpos, ypos, xcorr, ycorr, color=color, linewidth=linewidth, head_width=head_size)
        innerax.arrow(xpos, ypos, xcorr, ycorr, color=color, linewidth=linewidth, head_width=head_size)


def _plot_boxes_limits_and_labels(outerax, innerax, xlabel, ylabel, box_size, outertitle, innertitle, marker='x',
                                  marker_color='blue', rectangle_color='red', fixed_aspect=None):
    """
    Set limits and axis labels to array configuration boxes
    Args:
        outerax: Plotting axis for the outer array box
        innerax: Plotting axis for the inner array box
        xlabel: X axis label
        ylabel: Y axis label
        box_size: inner array box size
        outertitle: Title for the outer array box
        innertitle: Title for the inner array box
        marker: Marker for the array center
        marker_color: Color for the array center marker
        rectangle_color: Color of the rectangle representing the inner array box in the outer array plot
    """
    half_box = box_size / 2.
    x_lim, y_lim = outerax.get_xlim(), outerax.get_ylim()
    x_half, x_mid = (x_lim[1] - x_lim[0]) / 2, (x_lim[1] + x_lim[0]) / 2
    y_half, y_mid = (y_lim[1] - y_lim[0]) / 2, (y_lim[1] + y_lim[0]) / 2
    if x_half > y_half:
        y_lim = [y_mid - x_half, y_mid + x_half]
    else:
        x_lim = [x_mid - y_half, x_mid + y_half]
    outerax.set_xlim(x_lim)
    outerax.set_ylim(y_lim)
    outerax.set_xlabel(xlabel)
    outerax.set_ylabel(ylabel)
    outerax.plot(0, 0, marker=marker, color=marker_color)
    box = Rectangle((-half_box, -half_box), box_size, box_size, linewidth=0.5, edgecolor=rectangle_color,
                    facecolor='none')
    outerax.add_patch(box)
    outerax.set_title(outertitle)
    if fixed_aspect is not None:
        outerax.set_aspect(fixed_aspect)

    # Smaller box limits and labels
    innerax.set_xlim([-half_box, half_box])
    innerax.set_ylim([-half_box, half_box])
    innerax.set_xlabel(xlabel)
    innerax.set_ylabel(ylabel)
    innerax.plot(0, 0, marker=marker, color=marker_color)
    innerax.set_title(innertitle)
    if fixed_aspect is not None:
        innerax.set_aspect(fixed_aspect)


def _time_label(unit):
    return f'Time from observation start [{unit}]'


def _elevation_label(unit):
    return f'Elevation [{unit}]'


def _declination_label(unit):
    return f'Declination [{unit}]'


def _hour_angle_label(unit):
    return f'Hour Angle [{unit}]'
