import matplotlib.image
import numpy as np
from scipy.stats import linregress

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps as matplotlib_cmaps
from matplotlib.colors import ListedColormap
from astrohack.utils import figsize, fontsize

astrohack_cmaps = list(matplotlib_cmaps.keys())
astrohack_cmaps.append("AIPS")


def create_figure_and_axes(
    figure_size,
    boxes,
    default_figsize=figsize,
    sharex=False,
    sharey=False,
    plot_is_3d=False,
):
    """
    Create a figures and plotting axes within according to a desired figure size and number of boxes
    Args:
        figure_size: Desired figure size in inches
        boxes: How many subplots in the horizontal and vertical directions
        default_figsize: Default figure size for when the user specifies no figure size

    Returns:
    Figure and plotting axes array
    """
    if figure_size is None or figure_size == "None":
        prog_fig_size = default_figsize
    else:
        prog_fig_size = figure_size

    if plot_is_3d:
        fig, axes = plt.subplots(
            boxes[0],
            boxes[1],
            figsize=prog_fig_size,
            sharex=sharex,
            sharey=sharey,
            subplot_kw={"projection": "3d"},
        )
    else:
        fig, axes = plt.subplots(
            boxes[0], boxes[1], figsize=prog_fig_size, sharex=sharex, sharey=sharey
        )
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


def well_positioned_colorbar(
    ax, fig, mappable, label, location="right", size="5%", pad=0.05
):
    """
    Adds a well positioned colorbar to a plot
    Args:
        ax: Axes instance to add the colorbar
        fig: Figure in which the axes are embedded
        mappable: The plt.imshow or colormap instance associated to the colorbar
        label: Colorbar label
        location: Colorbar location
        size: Colorbar size
        pad: Colorbar padding

    Returns: the well positioned colorbar

    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)

    if isinstance(mappable, matplotlib.image.AxesImage):
        return fig.colorbar(mappable, label=label, cax=cax)
    else:  # mappable is a colormap
        sm = plt.cm.ScalarMappable(cmap=mappable)
        sm.set_array([])
        return fig.colorbar(sm, label=label, cax=cax)


def compute_extent(x_axis, y_axis, margin=0.0):
    """
    Compute extent from the arrays representing the X and Y axes
    Args:
        x_axis: X axis np array
        y_axis: Y axis np array
        margin: Optional margin to add to plots

    Returns:
        len=4 list with [xmin, xmax, ymin, ymax]
    """
    mins = np.array([np.min(x_axis), np.min(y_axis)])
    maxs = np.array([np.max(x_axis), np.max(y_axis)])
    data_range = maxs - mins
    mins -= margin * data_range
    maxs += margin * data_range
    extent = [mins[0], maxs[0], mins[1], maxs[1]]
    return extent


def get_proper_color_map(user_cmap, default_cmap="viridis"):
    if user_cmap is None or user_cmap == "None":
        return matplotlib_cmaps[default_cmap]
    elif user_cmap == "AIPS":
        # 8-bit color values picked from AIPS plots using GIMP
        cmap = ListedColormap(
            [
                [71 / 255.0, 71 / 255.0, 71 / 255.0, 1],  # Grey
                [104 / 255.0, 0 / 255.0, 142 / 255.0, 1],  # Purple/ dark blue?
                [0 / 255.0, 0 / 255.0, 186 / 255.0, 1],  # Blue
                [71 / 255.0, 147 / 255.0, 230 / 255.0, 1],  # Pink
                [0 / 255.0, 130 / 255.0, 0 / 255.0, 1],  # Green
                [0 / 255.0, 243 / 255.0, 0 / 255.0, 1],  # Light Green
                [255 / 255.0, 255 / 255.0, 0 / 255.0, 1],  # Yellow
                [255 / 255.0, 158 / 255.0, 0 / 255.0, 1],  # Orange
                [255 / 255.0, 0 / 255.0, 0 / 255.0, 1],  # Red
            ]
        )
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
    marker="x",
    marker_color="blue",
    rectangle_color="red",
    fixed_aspect=None,
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
    half_box = box_size / 2.0
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
        facecolor="none",
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


def scatter_plot(
    ax,
    xdata,
    xlabel,
    ydata,
    ylabel,
    title=None,
    labels=None,
    xlim=None,
    ylim=None,
    hlines=None,
    vlines=None,
    model=None,
    data_marker="+",
    data_color="red",
    data_linestyle="",
    data_label="data",
    hv_linestyle="--",
    hv_color="black",
    model_marker="x",
    model_color="blue",
    model_linestyle="",
    model_label="model",
    plot_residuals=True,
    residuals_marker="+",
    residuals_color="black",
    residuals_linestyle="",
    residuals_label="residuals",
    add_regression=False,
    regression_linestyle="-",
    regression_color="black",
):
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
        plot_residuals: Add a residuals subplot at the bottom when a model is provided
        residuals_marker: Marker for residuals
        residuals_color: Color for residual markers
        residuals_linestyle: Line style for residuals
        residuals_label: Label for residuals
        add_regression: Add a linear regression between X and y data
        regression_linestyle: Line style for the regression plot
        regression_color: Color for the regression plot
    """
    ax.plot(
        xdata,
        ydata,
        ls=data_linestyle,
        marker=data_marker,
        color=data_color,
        label=data_label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
            ax.text(
                xdata[ilabel],
                ydata[ilabel],
                labels[ilabel],
                fontsize=0.8 * fontsize,
                ha="left",
                va="center",
                rotation=20,
            )

    if add_regression:
        slope, intercept, _, _, _ = linregress(xdata, ydata)
        regression_label = f"y = {slope:.4f}*x + {intercept:.4f}"
        yregress = slope * xdata + intercept
        ax.plot(
            xdata,
            yregress,
            ls=regression_linestyle,
            color=regression_color,
            label=regression_label,
            lw=2,
        )
        ax.legend()

    if model is not None:
        ax.plot(
            xdata,
            model,
            ls=model_linestyle,
            marker=model_marker,
            color=model_color,
            label=model_label,
        )
        ax.legend()
        if plot_residuals:
            divider = make_axes_locatable(ax)
            ax_res = divider.append_axes("bottom", size="20%", pad=0)
            ax.figure.add_axes(ax_res)
            residuals = ydata - model
            ax.set_xticks([])
            ax_res.plot(
                xdata,
                residuals,
                ls=residuals_linestyle,
                marker=residuals_marker,
                color=residuals_color,
                label=residuals_label,
            )
            if xlim is not None:
                ax_res.set_xlim(xlim)

            minmax = np.nanmax(np.absolute(residuals))
            ax_res.set_ylim([-minmax, minmax])
            if vlines is not None:
                for vline in vlines:
                    ax_res.axvline(vline, color=hv_color, ls=hv_linestyle)

            ax_res.axhline(0, color=hv_color, ls=hv_linestyle)
            ax_res.set_ylabel("Residuals")

    if title is not None:
        ax.set_title(title)

    return


def simple_imshow_map_plot(
    ax,
    fig,
    x_axis,
    y_axis,
    gridded_2d_arr,
    title,
    cmap,
    zlim,
    x_label="X axis [m]",
    y_label="Y axis [m]",
    z_label="Z Scale",
    transpose=False,
    extent=None,
    extent_margin=0,
    add_colorbar=True,
):
    if zlim is None:
        minmax = [np.nanmin(gridded_2d_arr), np.nanmax(gridded_2d_arr)]
    else:
        minmax = zlim
    if extent is None:
        extent = compute_extent(x_axis, y_axis, margin=extent_margin)

    ax.set_title(title)
    if transpose:
        im = ax.imshow(
            gridded_2d_arr.T,
            cmap=cmap,
            extent=extent,
            interpolation="nearest",
            vmin=minmax[0],
            vmax=minmax[1],
            origin="lower",
        )
    else:
        im = ax.imshow(
            gridded_2d_arr,
            cmap=cmap,
            extent=extent,
            interpolation="nearest",
            vmin=minmax[0],
            vmax=minmax[1],
        )

    if add_colorbar:
        well_positioned_colorbar(ax, fig, im, z_label)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return im
