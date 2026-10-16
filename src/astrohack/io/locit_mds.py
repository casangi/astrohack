import pathlib

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter
from astropy.time import Time

from astrohack.antenna.telescope import get_proper_telescope
from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils.algorithms import compute_antenna_relative_off
from astrohack.utils.constants import figsize, notavail
from astrohack.utils.text import (
    convert_unit,
    create_pretty_table,
    lnbr,
    rad_to_deg_str,
    rad_to_hour_str,
)
from astrohack.utils.tools import get_telescope_lat_lon_rad
from astrohack.utils.validation import custom_unit_checker
from astrohack.visualization.array_cfg_plot import plot_array_configuration
from astrohack.visualization.plot_tools import (
    close_figure,
    create_figure_and_axes,
    scatter_plot,
)


class AstrohackLocitFile(AstrohackBaseFile):
    """Data class for locit data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackLocitFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackLocitFile object
        :rtype: AstrohackLocitFile
        """
        super().__init__(file=file)

    @toolviper.utils.parameter.validate()
    def print_source_table(self, save_to: str | None = None) -> None:
        """Prints a table with the sources observed for antenna location determination

        :param save_to: File to save the table to
        :type save_to: str, optional
        """

        field_names = [
            "Id",
            "Name",
            "RA FK5",
            "DEC FK5",
            "RA precessed",
            "DEC precessed",
        ]
        table = create_pretty_table(field_names, "l")
        src_dict = self.root.attrs["source_dict"]

        for source in src_dict.values():
            table.add_row(
                [
                    source["id"],
                    source["name"],
                    rad_to_hour_str(source["fk5"][0]),
                    rad_to_deg_str(source["fk5"][1]),
                    rad_to_hour_str(source["precessed"][0]),
                    rad_to_deg_str(source["precessed"][1]),
                ]
            )
        out_str = f"{lnbr}Sources:{lnbr}"
        out_str += table.get_string()
        print(out_str)
        if save_to is not None:
            with open(save_to, "w") as out_file:
                out_file.write(out_str)

    @toolviper.utils.parameter.validate()
    def print_array_configuration(
        self, relative: bool = True, save_to: str | None = None
    ) -> None:
        """Prints a table containing the array configuration

        :param relative: Print antenna coordinates relative to array center or in geocentric coordinates, default is True
        :type relative: bool, optional
        :param save_to: File to save the table to
        :type save_to: str, optional

        .. _Description:

        Print arrayx configuration in the dataset. Also marks the reference antenna and the antennas that are
        absent from the dataset. Coordinates of antenna stations can be relative to the array center or Geocentric
        (longitude, latitude and radius)

        """

        telescope_name = self.root.attrs["telescope_name"]
        telescope = get_proper_telescope(telescope_name)

        out_str = f"{lnbr}{telescope_name} antennas, # of antennas {len(self.root.keys())}:{lnbr}"
        if relative:
            nfields = 5
            field_names = [
                "Name",
                "Station",
                "East [m]",
                "North [m]",
                "Elevation [m]",
                "Distance [m]",
            ]
            tel_lon, tel_lat, tel_rad = get_telescope_lat_lon_rad(telescope)
        else:
            nfields = 4
            field_names = ["Name", "Station", "Longitude", "Latitude", "Radius [m]"]
            tel_lon, tel_lat, tel_rad = None, None, None

        table = create_pretty_table(field_names)
        for ant_name in telescope.antenna_list:
            row = [ant_name]
            try:
                ant_key = f"ant_{ant_name}"
                ant_info = self.root[ant_key].attrs["antenna_info"]
                row.append(ant_info["station"])
                if relative:
                    offsets = compute_antenna_relative_off(
                        ant_info, tel_lon, tel_lat, tel_rad
                    )
                    row.extend(
                        [
                            f"{offsets[0]:.4f}",
                            f"{offsets[1]:.4f}",
                            f"{offsets[2]:.4f}",
                            f"{offsets[3]:.4f}",
                        ]
                    )
                else:
                    row.extend(
                        [
                            rad_to_deg_str(ant_info["longitude"]),
                            rad_to_deg_str(ant_info["latitude"]),
                            f"{ant_info['radius']:.4f}",
                        ]
                    )
            except KeyError:
                for i_col in range(nfields):
                    row.append(notavail)

            table.add_row(row)
        out_str += table.get_string()
        print(out_str)
        if save_to is not None:
            with open(save_to, "w") as out_file:
                out_file.write(out_str)
        return

    @toolviper.utils.parameter.validate()
    def plot_source_positions(
        self,
        destination: str,
        labels: bool = True,
        precessed: bool = False,
        display: bool = False,
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
        dpi: int = 300,
    ) -> None:
        """Plot source positions in either FK5 or precessed right ascension and declination.

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param labels: Add source labels to the plot, defaults to False
        :type labels: bool, optional

        :param precessed: Plot in precessed coordinates? defaults to False (FK5)
        :type precessed: bool, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:

        Plot the sources on the source list to a full 24 hours 180 degrees flat 2D representation of the full sky.
        If precessed is set to True the coordinates precessd to the midpoint of the observations is plotted, otherwise
        the FK5 coordinates are plotted.
        The source names can be plotted next to their positions if label is True, however plots may become too crowded
        if that is the case.

        """
        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)

        if precessed:
            filename = str(
                pathlib.Path(destination).joinpath("locit_source_table_precessed.png")
            )
            time_range = self.root.attrs["time_range"]
            obs_midpoint = (time_range[1] + time_range[0]) / 2.0

        else:
            filename = str(
                pathlib.Path(destination).joinpath("locit_source_table_fk5.png")
            )
            obs_midpoint = None

        _plot_source_positions_sub(
            filename,
            self.root.attrs["source_dict"],
            precessed=precessed,
            obs_midpoint=obs_midpoint,
            display=display,
            figure_size=figure_size,
            dpi=dpi,
            label=labels,
        )

        return

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_array_configuration(
        self,
        destination: str,
        stations: bool = True,
        zoff: bool = False,
        unit: str = "km",
        box_size: int | float | None = None,
        display: bool = False,
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
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
        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
        plot_array_configuration(param_dict, self.root, "locit")
        return


def _plot_source_positions_sub(
    filename,
    src_dict,
    label=True,
    precessed=False,
    obs_midpoint=None,
    display=True,
    figure_size=figsize,
    dpi=300,
):
    """Backend function for plotting the source table
    Args:
        filename: Name for the png plot file
        src_dict: The dictionary containing the observed sources
        label: Add source labels
        precessed: Plot sources with precessed coordinates
        obs_midpoint: Time to which precesses the coordiantes
        display: Display plots in matplotlib
        figure_size: plot dimensions in inches
        dpi: Dots per inch (plot resolution)
    """

    n_src = len(src_dict)
    radec = np.ndarray((n_src, 2))
    name = []
    if precessed:
        if obs_midpoint is None:
            msg = "Observation midpoint is missing"
            logger.error(msg)
            raise RuntimeError(msg)
        coorkey = "precessed"
        time = Time(obs_midpoint, format="mjd")
        title = f"Coordinates precessed to {time.iso}"
    else:
        coorkey = "fk5"
        title = "FK5 reference frame"

    for i_src, src in src_dict.items():
        radec[int(i_src)] = src[coorkey]
        name.append(src["name"])

    fig, ax = create_figure_and_axes(figure_size, [1, 1])
    radec[:, 0] *= convert_unit("rad", "hour", "trigonometric")
    radec[:, 1] *= convert_unit("rad", "deg", "trigonometric")

    xlabel = "Right Ascension [h]"
    ylabel = "Declination [\u00b0]"
    if label:
        labels = name
    else:
        labels = None

    scatter_plot(
        ax,
        radec[:, 0],
        xlabel,
        radec[:, 1],
        ylabel,
        title=None,
        labels=labels,
        xlim=[-0.5, 24.5],
        ylim=[-95, 95],
        add_legend=False,
    )

    close_figure(fig, title, filename, dpi, display)
    return
