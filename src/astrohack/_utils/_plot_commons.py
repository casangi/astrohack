from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps as matplotlib_cmaps
from matplotlib.colors import ListedColormap
from astrohack._utils import figsize, fontsize
from astrohack._utils._constants import custom_split_checker, custom_unit_checker


astrohack_cmaps = list(matplotlib_cmaps.keys())
astrohack_cmaps.append('AIPS')


def _create_figure_and_axes(figure_size, boxes, default_figsize=figsize, sharex=False, sharey=False):
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


def _close_figure(figure, title, filename, dpi, display, tight_layout=True):
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


def _scatter_plot(ax, xdata, xlabel, ydata, ylabel, title=None, labels=None, xlim=None, ylim=None, hlines=None,
                  vlines=None, model=None, data_marker='+', data_color='red', data_linestyle='', data_label='data',
                  hv_linestyle='--', hv_color='black', model_marker='x', model_color='blue', model_linestyle='',
                  model_label='model'):
    """
    Do scatter simple scatter plots of data to a plotting axis
    Args:
        ax: The plotting axis
        xdata: X axis data
        xlabel: X axis data label
        ydata: Y axis data
        ylabel: Y axis datal label
        title: Plotting axis title
        labels: labels to be added to data
        xlim: X axis limits
        ylim: Y axis limits
        hlines: Horizontal lines to be drawn
        vlines: Vertical lines to be drawn
        model: Model to be overplotted to the data
        data_marker: Marker for data points
        data_color: Color of the data marker
        data_linestyle: Line style for connecting data points
        data_label: Label for data points when displayed along a model
        hv_linestyle: Line style for the horizontal or vertical lines displayed in the plot
        hv_color: Line color for the horizontal or vertical lines displayed in the plot
        model_marker: Marker for the model points
        model_color: Color of the model marker
        model_linestyle: Line style for connecting model points
        model_label: Label for model points
    """
    ax.plot(xdata, ydata, ls=data_linestyle, marker=data_marker, color=data_color, label=data_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if hlines is not None:
        for hline in hlines:
            ax.axhline(hline, color=hv_color, ls=hv_linestyle)
    if vlines is not None:
        for vline in vlines:
            ax.axvline(vline, color=hv_color, ls=hv_linestyle)
    if labels is not None:
        nlabels = len(labels)
        for ilabel in range(nlabels):
            ax.text(xdata[ilabel], ydata[ilabel], labels[ilabel], fontsize=.8*fontsize, ha='left', va='center',
                    rotation=20)
    if model is not None:
        ax.plot(xdata, model, ls=model_linestyle, marker=model_marker, color=model_color, label=model_label)
        ax.legend()
    return


def _well_positioned_colorbar(ax, fig, image, label, location='right', size='5%', pad=0.05):
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


def _get_proper_color_map(user_cmap, default_cmap='viridis'):
    if user_cmap is None or user_cmap == 'None':
        return matplotlib_cmaps[default_cmap]
    elif user_cmap == 'AIPS':
        # 8 bit color values picked from AIPS plots using GIMP
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


def custom_plots_checker(allowed_type):
    if allowed_type == 'colormaps':
        return astrohack_cmaps
    elif 'split' in allowed_type:
        return custom_split_checker(allowed_type)
    elif 'units' in allowed_type:
        return custom_unit_checker(allowed_type)
    else:
        return "Not found"
