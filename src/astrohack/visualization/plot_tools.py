from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps as matplotlib_cmaps
from matplotlib.colors import ListedColormap
from astrohack.utils import figsize

astrohack_cmaps = list(matplotlib_cmaps.keys())
astrohack_cmaps.append('AIPS')


def create_figure_and_axes(figure_size, boxes, default_figsize=figsize, sharex=False, sharey=False):
    """
    Create a figures and plotting axes within according to a desired figure size and number of boxes
    Args:
        figure_size: Desired figure size in inches
        boxes: How many subplots in the horizontal and vertical directions
        default_figsize: Default figure size for when the user specifies no figure size

    Returns:
    Figure and plotting axes array
    """
    if figure_size is None or figure_size == 'None':
        fig, axes = plt.subplots(boxes[0], boxes[1], figsize=default_figsize, sharex=sharex, sharey=sharey)
    else:
        fig, axes = plt.subplots(boxes[0], boxes[1], figsize=figure_size, sharex=sharex, sharey=sharey)

    return fig, axes


def close_figure(figure, title, filename, dpi, display, tight_layout=True):
    """
    Set title, save to disk and optionally close the figure
    Args:
        figure: The matplotlib figure object
        title: The superior title to be added to the figures
        filename: The file name to which save the figure
        dpi: dots per inch (resolution)
        display: Keep the plotting window open?
        tight_layout: Plots in the figure are tightly packed?
    """
    if title is not None:
        figure.suptitle(title)
    if tight_layout:
        figure.tight_layout()
    plt.savefig(filename, dpi=dpi)
    if display:
        plt.show()
    plt.close()
    return


def well_positioned_colorbar(ax, fig, image, label, location='right', size='5%', pad=0.05):
    """
    Adds a well positioned colorbar to a plot
    Args:
        ax: Axes instance to add the colorbar
        fig: Figure in which the axes are embedded
        image: The plt.imshow instance associated to the colorbar
        label: Colorbar label
        location: Colorbar location
        size: Colorbar size
        pad: Colorbar padding

    Returns: the well positioned colorbar

    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    return fig.colorbar(image, label=label, cax=cax)


def get_proper_color_map(user_cmap, default_cmap='viridis'):
    if user_cmap is None or user_cmap == 'None':
        return matplotlib_cmaps[default_cmap]
    elif user_cmap == 'AIPS':
        # 8-bit color values picked from AIPS plots using GIMP
        cmap = ListedColormap([[ 71/255.,  71/255.,  71/255., 1],  # Grey
                               [104/255.,   0/255., 142/255., 1],  # Purple/ dark blue?
                               [  0/255.,   0/255., 186/255., 1],  # Blue
                               [ 71/255., 147/255., 230/255., 1],  # Pink
                               [  0/255., 130/255.,   0/255., 1],  # Green
                               [  0/255., 243/255.,   0/255., 1],  # Light Green
                               [255/255., 255/255.,   0/255., 1],  # Yellow
                               [255/255., 158/255.,   0/255., 1],  # Orange
                               [255/255.,   0/255.,   0/255., 1]   # Red
                               ])
        return cmap
    else:
        return matplotlib_cmaps[user_cmap]


def plot_boxes_limits_and_labels(
        outerax,
        innerax,
        xlabel,
        ylabel,
        box_size,
        outertitle,
        innertitle,
        marker='x',
        marker_color='blue',
        rectangle_color='red',
        fixed_aspect=None
):
    """
    Set limits and axis labels to array configuration boxes
    Args:
        fixed_aspect ():
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

    box = Rectangle(
        (-half_box, -half_box),
        box_size,
        box_size,
        linewidth=0.5,
        edgecolor=rectangle_color,
        facecolor='none'
    )

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
