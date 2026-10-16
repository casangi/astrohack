import pathlib
from copy import deepcopy

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils.conversion import convert_unit
from astrohack.utils.text import param_to_list, undscr
from astrohack.utils.validation import custom_unit_checker
from astrohack.visualization.array_cfg_plot import plot_array_configuration
from astrohack.visualization.plot_tools import close_figure, create_figure_and_axes


class AstrohackPointFile(AstrohackBaseFile):
    """Data class for point data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackPointFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPointFile object
        :rtype: AstrohackPointFile
        """
        super().__init__(file=file)

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_pointing_in_time(
        self,
        destination: str,
        ant: str | list[str] = "all",
        pointing_key: str = "DIRECTIONAL_COSINES",
        plot_antennas_separately: bool = False,
        azel_unit: str = "deg",
        time_unit: str = "hour",
        az_scale: tuple | list[float] | np.ndarray | None = None,
        el_scale: tuple | list[float] | np.ndarray | None = None,
        time_scale: tuple | list[float] | np.ndarray | None = None,
        figure_size: tuple | list[float] | np.ndarray = (5.0, 6.4),
        display: bool = False,
        dpi: int = 300,
    ) -> None:
        """Plot Pointing for antennas in time.

        :param destination: Name of the destination folder to contain plot(s)
        :type destination: str

        :param ant: Antenna(s) to plot, default is "all"
        :type ant: str, list, optional

        :param pointing_key: Which xds pointing data key to plot, defaults to "DIRECTIONAL_COSINES"
        :type pointing_key: str, optional

        :param plot_antennas_separately: Create an individual plot file for each antenna?
        :type plot_antennas_separately: bool, optional

        :param azel_unit: Unit for Azimuth and Elevation in the plot(s), valid values are trigonometric units, default \
        is deg
        :type azel_unit: str, optional

        :param time_unit: Unit for time in the plot(s), valid values are time units, default is hour
        :type time_unit: str, optional

        :param az_scale: Azimuth plot limits, defaults to all Azimuths present when None.
        :type az_scale: Union[Tuple, List[float], np.array], optional

        :param el_scale: Elevation plot limits, defaults to all Elevations present when None.
        :type el_scale: Union[Tuple, List[float], np.array], optional

        :param time_scale: Time plot limits, defaults to all times present when None
        :type time_scale: Union[Tuple, List[float], np.array], optional

        :param display: Display plot(s) inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:
        Plot antenna pointing info in time together in one plot, or individually for each antenna.
        """

        pathlib.Path(destination).mkdir(exist_ok=True)
        input_params = locals()

        if plot_antennas_separately:
            self._plot_pointing_in_time_separately(input_params)
        else:
            self._plot_pointing_in_time_together(input_params)
        return

    def _get_plot_configuration(self, input_params):
        ant_list = param_to_list(input_params["ant"], self, "ant")
        time_fac = convert_unit("sec", input_params["time_unit"], "time")
        ang_fac = convert_unit("rad", input_params["azel_unit"], "trigonometric")
        target_column = input_params["pointing_key"].upper()
        return ant_list, time_fac, ang_fac, target_column

    def _plot_pointing_in_time_separately(self, input_params):
        ant_list, time_fac, ang_fac, target_column = self._get_plot_configuration(
            input_params
        )

        n_use_ants = 0
        for ant_key in ant_list:
            ant_name = ant_key.split(undscr)[1]
            if ant_key in self.keys():
                n_use_ants = n_use_ants + 1
                fig, axes, y_labels = _create_pointing_figure(input_params)
                _plot_one_pnt_xds(
                    time_fac,
                    ang_fac,
                    ant_name,
                    self[ant_key].dataset,
                    target_column,
                    y_labels,
                    axes,
                )
                _finalize_pointing_figure(
                    input_params, target_column, ant_name, y_labels, axes, fig
                )
            else:
                logger.warning(f"Antenna {ant_name} not found in dataset")

        if n_use_ants <= 0:
            logger.warning("No valid antennas selected, no plot produced.")
        return

    def _plot_pointing_in_time_together(self, input_params):
        ant_list, time_fac, ang_fac, target_column = self._get_plot_configuration(
            input_params,
        )
        fig, axes, y_labels = _create_pointing_figure(input_params)

        n_use_ants = 0
        for ant_key in ant_list:
            ant_name = ant_key.split(undscr)[1]
            if ant_key in self.keys():
                n_use_ants = n_use_ants + 1
                _plot_one_pnt_xds(
                    time_fac,
                    ang_fac,
                    ant_name,
                    self[ant_key].dataset,
                    target_column,
                    y_labels,
                    axes,
                )
            else:
                logger.warning(f"Antenna {ant_name} not found in dataset")

        if n_use_ants > 0:
            simple_ant_list = [ant_key.split(undscr)[1] for ant_key in ant_list]
            _finalize_pointing_figure(
                input_params,
                target_column,
                ", ".join(simple_ant_list),
                y_labels,
                axes,
                fig,
            )
        else:
            logger.warning("No valid antennas selected, no plot produced.")
        return

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_array_configuration(
        self,
        destination: str,
        stations: bool = True,
        zoff: bool = False,
        unit: str = "km",
        box_size: int | float | None = None,
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
        display: bool = False,
        dpi: int = 300,
    ) -> None:
        """Plot antenna positions.

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param stations: Add station names to the plot, defaults to True
        :type stations: bool, optional

        :param zoff: Add Elevation offsets to the plots, defaults to False
        :type zoff: bool, optional

        :param unit: Unit for the plot, valid values are length units, default is km
        :type unit: str, optional

        :param box_size: Size of the box for plotting the inner part of the array in unit, when none the box size is \
        20% of the total size of the array, default is None
        :type box_size: int, float, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:
        Plot the array configuration from the antenna positions.
        """

        pathlib.Path(destination).mkdir(exist_ok=True)
        input_params = locals()
        plot_array_configuration(input_params, self.root, "point")

    @toolviper.utils.parameter.validate()
    def set_antennas_as_reference(
        self,
        reference_antennas: str | list[str] | tuple[str],
        write_changes: bool = True,
    ) -> None:
        """
        Modify point_mds data to make specific antennas reference antennas, useful for older datasets that contain \
        wrong pointing data for reference antennas.

        :param reference_antennas: Antennas to transform into reference antennas
        :type reference_antennas: Union[str, list[str], tuple[str]]

        :param write_changes: Write the modified mds to disk?
        :type write_changes: bool, optional
        """

        if isinstance(reference_antennas, str):
            reference_antennas = [reference_antennas]

        for ref_ant in reference_antennas:
            ant_key = f"ant_{ref_ant}"
            try:
                ant_xds = self[ant_key].dataset
            except KeyError:
                logger.warning(f"Antenna {ref_ant} not found in dataset")
                continue
            logger.info(f"Making {ref_ant} a reference antenna")
            zeroed_pnt = np.zeros_like(ant_xds["POINTING_OFFSET"])
            ant_xds["POINTING_OFFSET"].values = zeroed_pnt
            ant_xds["DIRECTIONAL_COSINES"].values = zeroed_pnt
            ant_xds["TARGET"].values = ant_xds["DIRECTION"].values

            mapping_scans_list = ant_xds.attrs["mapping_scans_obs_dict"]

            for mapping_dict in mapping_scans_list:
                for ddi_key in mapping_dict:
                    mapping_dict[ddi_key] = {}

            ant_xds.attrs["mapping_scans_obs_dict"] = mapping_scans_list
            self[ant_key].dataset = ant_xds

        if write_changes:
            self.write(mode="a")
        return


def _create_pointing_figure(input_params):
    y_labels = ["azimuth", "elevation"]
    fig, axes = create_figure_and_axes(input_params["figure_size"], [2, 1])
    return fig, axes, y_labels


def _finalize_pointing_figure(
    input_params, target_column, ant_label, y_labels, axes, fig
):
    azel_unit = input_params["azel_unit"]
    time_unit = input_params["time_unit"]
    title = f"Pointing [{target_column}] data for: {ant_label}"
    filename = f"{input_params['destination']}/point_{target_column.lower()}_"
    if len(ant_label.split(",")) > 1:
        filename += "combined.png"
    else:
        filename += f"ant_{ant_label}.png"
    for i_coord, y_label in enumerate(y_labels):
        axes[i_coord].set_ylabel(f"{y_label.capitalize()} [{azel_unit}]")
        if y_label == "Azimuth":
            if input_params["az_scale"] is not None:
                axes[i_coord].set_ylim(input_params["az_scale"])
        else:
            if input_params["el_scale"] is not None:
                axes[i_coord].set_ylim(input_params["el_scale"])
        if input_params["time_scale"] is not None:
            axes[i_coord].set_xlim(input_params["time_scale"])
        axes[i_coord].set_xlabel(f"Time Since Observation start [{time_unit}]")
        axes[i_coord].legend()
    close_figure(fig, title, filename, input_params["dpi"], input_params["display"])


def _plot_one_pnt_xds(
    time_fac, ang_fac, ant_name, pnt_xds, target_column, y_labels, axes
):
    time_ax = deepcopy(pnt_xds.coords["time"].values)
    # Set time from obs start
    time_ax -= time_ax[0]
    plot_data = pnt_xds[target_column].values
    for i_coord, y_label in enumerate(y_labels):
        axes[i_coord].plot(
            time_fac * time_ax,
            ang_fac * plot_data[:, i_coord],
            label=ant_name,
            ls="",
            marker="o",
            ms=5,
        )
