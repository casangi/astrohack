import json
from typing import Any

import numpy as np
import pandas as pd
import toolviper.utils.logger as logger
from rich.console import Console

from astrohack.io.dio import open_holog
from astrohack.utils.text import get_data_name


def _add_prefix_to_keys(prefix, key_list):
    def add_prefix_lambda(list_item):
        return f"{prefix}_{list_item}"

    return list(map(add_prefix_lambda, key_list))


def _check_if_array_in_dict(array_dict, array):
    for key, val in array_dict.items():
        if np.array_equiv(val, array):
            return key
    return False


class HologObsDict(dict):
    def __init__(self, dict_obj: dict = None):
        if dict_obj is None:
            super().__init__()
        else:
            super().__init__(dict_obj)

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    def print(self, style: str = "static"):
        if style == "dynamic":
            from IPython.display import JSON

            return JSON(self)
        else:
            console = Console()
            console.log(self, log_locals=False)
            return None

    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath) as file:
            json_dict = json.load(file)
            return cls(json_dict)

    def to_json_file(self, filepath):
        with open(filepath, "w") as file:
            json.dump(self, file, indent=4)

    @classmethod
    def from_holog_file(cls, filepath):
        holog_mds = open_holog(filepath)
        return cls(holog_mds.root.attrs["holog_obs_dict"])

    @classmethod
    def create_from_pointing_file(
        cls,
        pnt_mds,
        exclude_antennas,
        baseline_average_distance,
        baseline_average_nearest,
    ):

        ant_names_main = pnt_mds.root.attrs["antenna_names"]
        dist_matrix_dict = pnt_mds.root.attrs["baseline_dist_matrix"]
        mapping_scans_dict = {}
        holog_obs_dict = cls()
        map_id = 0
        ant_names_set = set()

        if exclude_antennas is None:
            exclude_antennas = []
        elif isinstance(exclude_antennas, str):
            exclude_antennas = [exclude_antennas]
        else:
            pass

        for ant_name in exclude_antennas:
            prefixed = "ant_" + ant_name
            if prefixed not in pnt_mds.keys():
                logger.warning(
                    f"Bad reference antenna {ant_name} is not present in the data."
                )

        # Generate {ddi: {map: {scan:[i ...], ant:{ant_map_0:[], ...}}}} structure. No reference antennas are added
        # because we first need to populate all mapping antennas.
        for ant_name, ant_ds in pnt_mds.items():
            if "ant" in ant_name:
                ant_name = ant_name.replace("ant_", "")
                if ant_name in exclude_antennas:
                    pass
                else:
                    if ant_name in ant_names_main:  # Check if antenna in main table.
                        ant_names_set.add(ant_name)
                        for ddi, map_dict in ant_ds.attrs["mapping_scans_obs_dict"][
                            0
                        ].items():
                            if ddi not in holog_obs_dict:
                                holog_obs_dict[ddi] = {}
                            for ant_map_id, scan_list in map_dict.items():
                                if scan_list:
                                    map_key = _check_if_array_in_dict(
                                        mapping_scans_dict, scan_list
                                    )
                                    if not map_key:
                                        map_key = "map_" + str(map_id)
                                        mapping_scans_dict[map_key] = scan_list
                                        map_id = map_id + 1

                                    if map_key not in holog_obs_dict[ddi]:
                                        holog_obs_dict[ddi][map_key] = {
                                            "scans": scan_list,
                                            "ant": {},
                                        }

                                    holog_obs_dict[ddi][map_key]["ant"][ant_name] = []

        df_mat = pd.DataFrame.from_dict(dist_matrix_dict, orient="index")

        if (baseline_average_distance != "all") and (baseline_average_nearest != "all"):
            logger.error(
                "baseline_average_distance and baseline_average_nearest can not both be specified."
            )

            raise RuntimeError("Too many baseline parameters specified.")

        n_ant_total = len(ant_names_set)
        # The reference antennas are then given by ref_ant_set = ant_names_set - map_ant_set.
        for ddi, ddi_dict in holog_obs_dict.items():
            if len(ddi_dict.items()) == 0:
                logger.warning(f"No mapping antennas for DDI {get_data_name(ddi)}")
                continue
            for map_id, map_dict in ddi_dict.items():
                map_ant_set = set(map_dict["ant"].keys())

                # Need a copy because of del holog_obs_dict[ddi][map_id]['ant'][map_ant_key] below.
                map_ant_keys = list(map_dict["ant"].keys())
                n_map_ant = len(map_ant_keys)
                if n_map_ant == n_ant_total:
                    logger.warning(
                        f"No reference antennas for DDI {get_data_name(ddi)}"
                    )
                    continue

                for map_ant_key in map_ant_keys:
                    ref_ant_set = ant_names_set - map_ant_set

                    # Select reference antennas by distance from mapping antenna
                    if baseline_average_distance != "all":
                        sub_ref_ant_set = []

                        for ref_ant in ref_ant_set:
                            if (
                                df_mat.loc[map_ant_key, ref_ant]
                                < baseline_average_distance
                            ):
                                sub_ref_ant_set.append(ref_ant)

                        if (not sub_ref_ant_set) and ref_ant_set:
                            logger.warning(
                                "DDI "
                                + str(ddi)
                                + " and mapping antenna "
                                + str(map_ant_key)
                                + " has no reference antennas. If baseline_average_distance was specified "
                                "increase this distance. See antenna distance matrix in log by setting "
                                "debug level to DEBUG in client function."
                            )

                        ref_ant_set = sub_ref_ant_set

                    # Select reference antennas by the n-closest antennas
                    if baseline_average_nearest != "all":
                        sub_ref_ant_set = []
                        nearest_ant_list = (
                            df_mat.loc[map_ant_key, :]
                            .loc[list(ref_ant_set)]
                            .sort_values()
                            .index.tolist()[0:baseline_average_nearest]
                        )

                        logger.debug(nearest_ant_list)
                        for ref_ant in ref_ant_set:
                            if ref_ant in nearest_ant_list:
                                sub_ref_ant_set.append(ref_ant)

                        ref_ant_set = sub_ref_ant_set
                    ##################################################

                    if ref_ant_set:
                        holog_obs_dict[ddi][map_id]["ant"][map_ant_key] = list(
                            ref_ant_set
                        )
                    else:
                        del holog_obs_dict[ddi][map_id]["ant"][
                            map_ant_key
                        ]  # Don't want mapping antennas with no reference antennas.
                        logger.warning(
                            "DDI "
                            + str(ddi)
                            + " and mapping antenna "
                            + str(map_ant_key)
                            + " has no reference antennas."
                        )

        return holog_obs_dict

    def _select_ddi(self, selected_values: int | list[int]):
        prefixed_selected_values = _add_prefix_to_keys("ddi", selected_values)
        ddi_list = list(self.keys())
        for ddi_key in ddi_list:
            if ddi_key not in prefixed_selected_values:
                self.pop(ddi_key)
        return

    def _select_antenna(self, selected_values: str | list[str]):
        for ddi_key in self.keys():
            for map_key in self[ddi_key].keys():
                ant_list = list(self[ddi_key][map_key]["ant"].keys())
                for ant_key in ant_list:
                    if ant_key not in selected_values:
                        self[ddi_key][map_key]["ant"].pop(ant_key)
        return

    def _select_map(self, selected_values: int | list[int]):
        prefixed_selected_values = _add_prefix_to_keys("map", selected_values)
        for ddi_key in self.keys():
            map_list = list(self[ddi_key].keys())
            for map_key in map_list:
                if map_key not in prefixed_selected_values:
                    self[ddi_key].pop(map_key)

    def _select_scan(self, selected_values: int | list[int]):
        for ddi_key in self.keys():
            for map_key in self[ddi_key].keys():
                self[ddi_key][map_key]["scan"] = selected_values
        return

    @staticmethod
    def get_nearest_baselines(
        antenna: str, dist_matrix_dict, n_baselines: int = None
    ) -> object:

        df_matrix = pd.DataFrame.from_dict(dist_matrix_dict, orient="index")

        # Skip the first index because it is a self distance
        if n_baselines is None:
            return (
                df_matrix[antenna].sort_values(ascending=True).index[1:].values.tolist()
            )

        return (
            df_matrix[antenna]
            .sort_values(ascending=True)
            .index[1:n_baselines]
            .values.tolist()
        )

    def select_baseline(
        self, selected_values, n_baselines, dist_matrix_dict, reference=None
    ):
        if reference is not None:
            if not isinstance(reference, list):
                reference = [reference]

        ddi_list = list(self.keys())

        for ddi in ddi_list:
            map_list = list(self[ddi].keys())
            for mp in map_list:
                ant_list = list(self[ddi][mp]["ant"].keys())
                for ant in ant_list:
                    if ant not in selected_values:
                        self[ddi][mp]["ant"].pop(ant)
                        continue

                    if reference is None and n_baselines is not None:
                        reference_antennas = self[ddi][mp]["ant"][ant]

                        if n_baselines > len(reference_antennas):
                            n_baselines = len(reference_antennas)

                        sorted_antennas = np.array(
                            self.get_nearest_baselines(ant, dist_matrix_dict)
                        )

                        values, i, j = np.intersect1d(
                            reference_antennas, sorted_antennas, return_indices=True
                        )
                        index = np.sort(j)

                        self[ddi][mp]["ant"][ant] = sorted_antennas[index][:n_baselines]

                    else:
                        self[ddi][mp]["ant"][ant] = reference
        return

    def select(self, key, selected_values):
        if not isinstance(selected_values, (list, tuple)):
            selected_values = [selected_values]

        match key:
            case "ddi":
                self._select_ddi(selected_values)
            case "antenna":
                self._select_antenna(selected_values)
            case "map":
                self._select_map(selected_values)
            case "scan":
                self._select_scan(selected_values)
        return
