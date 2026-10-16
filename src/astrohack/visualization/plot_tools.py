from typing import Any

import matplotlib.image
import numpy as np
import toolviper.utils.logger as logger
from matplotlib import colormaps as matplotlib_cmaps
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress, siegelslopes, theilslopes

from astrohack.utils.constants import figsize, fontsize

astrohack_cmaps = list(matplotlib_cmaps.keys())
astrohack_cmaps.append("AIPS")


def get_execution_environment():
    try:
        # Check if get_ipython is available in the global namespace
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__

        if shell == "ZMQInteractiveShell":
            return "jupyter"  # Jupyter Notebook or JupyterLab
        elif shell == "TerminalInteractiveShell":
            return "ipython"  # Terminal-based IPython
        else:
            return "other"  # Other IDE backends
    except (NameError, ImportError):
        return "terminal"  # Standard Python interpreter


def create_figure_and_axes(
    figure_size: list | tuple | None,
    boxes: list | tuple | np.ndarray,
    default_figsize: list | tuple = figsize,
    sharex: bool = False,
    sharey: bool = False,
    plot_is_3d: bool = False,
    force_2d_axes_array: bool = False,
):
    """
    Create a figures and plotting axes within according to a desired figure size and number of boxes
    Args:
        figure_size: Desired figure size in inches
        boxes: How many subplots in the horizontal and vertical directions
        default_figsize: Default figure size for when the user specifies no figure size
        sharex: Subplots share the X axis
        sharey: Subplots share the Y axis
        plot_is_3d: Subplots will contain 3d data.
        force_2d_axes_array:

    Returns:
    Figure and plotting axes array
    """
    if figure_size is None or figure_size == "None":
        prog_fig_size = default_figsize
    else:
        prog_fig_size = figure_size

    fig = Figure(figsize=prog_fig_size)

    subplots_kwargs = {
        "nrows": boxes[0],
        "ncols": boxes[1],
        "sharex": sharex,
        "sharey": sharey,
        "squeeze": not force_2d_axes_array,
    }
    if plot_is_3d:
        subplots_kwargs["subplot_kw"] = {"projection": "3d"}

    axes = fig.subplots(**subplots_kwargs)

    return fig, axes


def close_figure(
    figure: Figure,
    title: str,
    filename: str,
    dpi: int,
    display: bool,
    tight_layout: bool = True,
):
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
    grid_spec = figure.axes[0].get_gridspec()
    nrows = grid_spec.nrows
    if nrows > 4 and tight_layout:
        y_off = 0.985
    else:
        y_off = None
    if title is not None:
        figure.suptitle(title, x=0.5, y=y_off)

    if tight_layout:
        if nrows > 4:
            figure.set_layout_engine("tight", rect=[0.015, 0.015, 0.985, 0.97])
        else:
            figure.tight_layout()

    # Force filename to contain .png in case no extension is present
    if len(filename.split(".")) == 1:
        filename += ".png"
    figure.savefig(filename, dpi=dpi)

    if display:
        # figure.show()
        py_env = get_execution_environment()
        if py_env in ["terminal", "ipython", "other"]:
            mpl_backend = matplotlib.get_backend()
            if mpl_backend.lower() == "tkagg":
                import tkinter as tk

                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

                # 2. Instantiate canvas without explicit master (defaults to an internal Tk root)
                canvas = FigureCanvasTkAgg(figure)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill=tk.BOTH, expand=True)

                # 3. Retrieve the automatically generated top-level window and run it
                window = canvas_widget.winfo_toplevel()
                window.title(f"Astrohack: {title}")
                window.mainloop()
            elif mpl_backend == "macosx":
                import time

                from matplotlib.backends.backend_macosx import (
                    FigureCanvasMac,
                    FigureManagerMac,
                )

                running = True

                def on_close(event):
                    nonlocal running
                    running = False

                canvas = FigureCanvasMac(figure)
                canvas.mpl_connect("close_event", on_close)
                manager = FigureManagerMac(canvas, 1)
                manager.show()
                while running:
                    canvas.flush_events()
                    time.sleep(0.01)

            else:
                logger.warning(
                    f"'{mpl_backend}' backend not supported for interactive plots"
                )
        elif py_env in ["jupyter"]:
            from IPython import display

            display.display(display.Image(filename, embed=True))
            # display(HTML(f'<img src="{filename}" style="max-width:60%; height:auto;">'))
        else:
            logger.warning(f"Unrecognized python environment '{py_env}'")

    figure.clear()
    del figure
    return


def well_positioned_colorbar(
    ax: Axes,
    fig: Figure,
    mappable,
    label: str,
    location: str = "right",
    size: str = "5%",
    pad: float | int = 0.05,
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
        sm = matplotlib.cm.ScalarMappable(cmap=mappable)
        sm.set_array([])
        return fig.colorbar(sm, label=label, cax=cax)


def compute_extent(x_axis: np.ndarray, y_axis: np.ndarray, margin: float | int = 0.0):
    """
    Compute extent from the arrays representing the X and Y axes
    Args:
        x_axis: X axis np array
        y_axis: Y axis np array
        margin: Optional margin to add to plots

    Returns:
        len=4 list with [xmin, xmax, ymin, ymax]
    """
    mins = np.array([x_axis[0], y_axis[0]])
    maxs = np.array([x_axis[-1], y_axis[-1]])
    data_range = maxs - mins
    mins -= margin * data_range
    maxs += margin * data_range
    extent = [mins[0], maxs[0], mins[1], maxs[1]]
    return extent


def get_proper_color_map(user_cmap: str | None, default_cmap: str = "viridis"):
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
    outerax: Axes,
    innerax: Axes,
    xlabel: str,
    ylabel: str,
    box_size: float | int,
    outertitle: str,
    innertitle: str,
    marker: str = "x",
    marker_color: str = "blue",
    rectangle_color: str = "red",
    fixed_aspect: float | int | None = None,
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
    x_lim, y_lim = outerax.get_xlim(), outerax.get_ylim()
    x_half, x_mid = (x_lim[1] - x_lim[0]) / 2, (x_lim[1] + x_lim[0]) / 2
    y_half, y_mid = (y_lim[1] - y_lim[0]) / 2, (y_lim[1] + y_lim[0]) / 2

    if box_size == 0:
        half_box = np.min([x_half, y_half])
    else:
        half_box = box_size / 2.0

    if x_half > y_half:
        y_lim = (y_mid - x_half, y_mid + x_half)

    else:
        x_lim = (x_mid - y_half, x_mid + y_half)

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
    innerax.set_xlim((-half_box, half_box))
    innerax.set_ylim((-half_box, half_box))
    innerax.set_xlabel(xlabel)
    innerax.set_ylabel(ylabel)
    innerax.plot(0, 0, marker=marker, color=marker_color)
    innerax.set_title(innertitle)

    if fixed_aspect is not None:
        innerax.set_aspect(fixed_aspect)


def scatter_plot(
    ax: Axes,
    xdata: np.ndarray,
    xlabel: str,
    ydata: np.ndarray,
    ylabel: str,
    title: str | None = None,
    labels: list | tuple | None = None,
    xlim: list | tuple | None = None,
    ylim: list | tuple | None = None,
    hlines: list | tuple | np.ndarray | None = None,
    vlines: list | tuple | np.ndarray | None = None,
    model: np.ndarray | None = None,
    data_marker: str = "+",
    data_color: str = "red",
    data_linestyle: str = "",
    data_label: str = "data",
    hv_linestyle: str = "--",
    hv_color: str = "black",
    model_marker: str = "x",
    model_color: str = "blue",
    model_linestyle: str = "",
    model_label: str = "model",
    plot_residuals: bool = True,
    residuals_marker: str = "+",
    residuals_color: str = "black",
    residuals_linestyle: str = "",
    residuals_label: str = "residuals",
    add_regression: bool = False,
    regression_linestyle: str = "-",
    regression_color: str = "black",
    regression_method: str = "linregress",
    add_regression_reference: bool = False,
    regression_reference: Any = (1.0, 0.0),
    regression_reference_color: str = "orange",
    regression_reference_label: str = "Regression refrence",
    force_equal_aspect: bool = False,
    add_legend: bool = True,
    legend_location: str = "best",
    title_fontsize: int = 10,
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
        regression_method: Which scipy function to use for the linear regression: linregress, theilslopes or siegelslopes
        add_regression_reference: Add reference for the expected regression result
        regression_reference: 2 value array/tuple/list with a slope and intercept for reference
        regression_reference_color: Color for reference regression
        regression_reference_label: Label for reference regression
        force_equal_aspect: Force equal aspect on plot box
        add_legend: add legend to the plot
        legend_location: Location of the legend in the plot
        title_fontsize: Font size of the title
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
        if regression_method == "linregress":
            slope, intercept, _, _, _ = linregress(xdata, ydata)
        elif regression_method == "theilslopes":
            slope, intercept, _, _ = theilslopes(ydata, xdata)
        elif regression_method == "siegelslopes":
            slope, intercept = siegelslopes(ydata, xdata)
        else:
            raise RuntimeError(f"Unknown linear regression method: {regression_method}")
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
        if add_regression_reference:
            reg_ref = regression_reference[0] * xdata + regression_reference[1]
            ax.plot(
                xdata,
                reg_ref,
                ls=regression_linestyle,
                color=regression_reference_color,
                label=regression_reference_label,
            )

    if model is not None:
        ax.plot(
            xdata,
            model,
            ls=model_linestyle,
            marker=model_marker,
            color=model_color,
            label=model_label,
        )
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

            minmax = float(np.nanmax(np.absolute(residuals)))
            ax_res.set_ylim([-minmax, minmax])
            if vlines is not None:
                for vline in vlines:
                    ax_res.axvline(vline, color=hv_color, ls=hv_linestyle)

            ax_res.axhline(0, color=hv_color, ls=hv_linestyle)
            ax_res.set_ylabel("Residuals")
            ax_res.set_xlabel(xlabel)

    if force_equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if add_legend:
        ax.legend(loc=legend_location)

    return


def simple_imshow_map_plot(
    ax: Axes,
    fig: Figure,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    gridded_2d_arr: np.ndarray,
    title: str,
    colormap: str,
    zlim: list | tuple | np.ndarray,
    x_label: str = "X axis [m]",
    y_label: str = "Y axis [m]",
    z_label: str = "Z Scale",
    transpose: bool = False,
    extent: list | tuple | np.ndarray | None = None,
    extent_margin: float | int = 0,
    add_colorbar: bool = True,
    interpolation: str = "nearest",
):
    cmap = get_proper_color_map(colormap)
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
            interpolation=interpolation,
            vmin=minmax[0],
            vmax=minmax[1],
            origin="lower",
        )
    else:
        im = ax.imshow(
            gridded_2d_arr,
            cmap=cmap,
            extent=extent,
            interpolation=interpolation,
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


def set_y_axis_lims_from_default(
    ax: Axes, user_y_scale: tuple | list, prog_defaults: tuple | list
):
    if user_y_scale is None:
        applied_scale = list(prog_defaults)
    else:
        applied_scale = list(user_y_scale)

    sub_lims = [-1, 1]
    for i_lim in range(2):
        if not np.isfinite(applied_scale[i_lim]):
            applied_scale[i_lim] = sub_lims[i_lim]

    ax.set_ylim(applied_scale)


def filter_duplicates_in_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
