"""
Functions and classes to do the following :

(a) Read/write Zernike coefficients into Zarr
(b) Convert between CSV and Zarr formats for the Zernike coefficients
(c) Fit the holog apertures with Zernike polynomials and store the fitten
models and coefficients in one of the above formats.
"""

import pandas as pd
import xarray as xr
import numpy as np
import numba as nb

import os

from astrohack._utils._imaging import _mask_circular_disk


class ZernikeCoefficients:
    """
    Class to encapsulate the Zernike coefficients and their associated
    operations, such as reading and writing to Zarr and CSV files.
    """

    def __init__(self):
        pass

    def read_csv(self, infile):
        """
        Read the Zernike coefficients from a CSV file.
        """

        if not os.path.exists(infile):
            raise FileNotFoundError(f"File {infile} not found")

        df = pd.read_csv(infile)
        return df

    def _stok_num_to_str(self, stok_num):
        """
        Convert from CASA number based indexing to the corresponding
        string.

        :param stok_num: The CASA Stokes index
        :type stok_num: int

        :return: The Stokes type as a string
        :rtype: str
        """

        stokes_map = {
            0: "Undefined",
            1: "I",
            2: "Q",
            3: "U",
            4: "V",
            5: "RR",
            6: "RL",
            7: "LR",
            8: "LL",
            9: "XX",
            10: "XY",
            11: "YX",
            12: "YY",
            13: "RX",
            14: "RY",
            15: "LX",
            16: "LY",
            17: "XR",
            18: "XL",
            19: "YR",
            20: "YL",
            21: "PP",
            22: "PQ",
            23: "QP",
            24: "QQ",
            25: "RCircular",
            26: "LCircular",
            27: "Linear",
            28: "Ptotal",
            29: "Plinear",
            30: "PFtotal",
            31: "PFlinear",
            32: "PAngle",
        }

        return stokes_map[stok_num]

    def csv_to_zarr(self, infile, outfile="", xds_attrs={}):
        """
        Convert the Zernike coefficients from CSV to Zarr format.
        The frequencies in the CSV file are assumed to be in MHz.


        :param infile: The name of the CSV file to read from.
        :type infile: str

        :param outfile: The name of the Zarr file to write to. If none is given, the
        Dataset is returned but not written to disk.
        :type outfile: str

        :param xds_attrs: A dictionary containing the Dataset attributes
        :type xds_attrs: dict

        :param outname: The name of the dataset to write to. If not provided, the
        Dataset is returned but is not written to disk.
        :type outname: str

        :return: The corresponding xarray dataset of the Zernike coefficients.
        :rtype: xarray.Dataset
        """

        df = self.read_csv(infile)
        colnames = df.columns
        colnames = [col.strip("#") for col in colnames]
        df.columns = colnames

        ds = xr.Dataset()

        # Setup the input attributes
        # TODO : Check that the attributes follow schema and are valid
        if len(xds_attrs) > 0:
            for key, value in xds_attrs.items():
                ds.attrs[key] = value

        coeffs = df["ind"].unique()
        stok_num = df["stokes"].unique()
        stok_str = [self._stok_num_to_str(stok) for stok in stok_num]
        freqs = df["freq"].unique()
        freqs /= 1e3

        if "antenna" in df.columns:
            ants = df["antenna"].unique()
        else:
            ants = [0]

        coords = {
            "zernike_index": coeffs,
            "polarization": stok_str,
            "frequency": freqs,
            "antenna_index": ants,
        }
        ds = ds.assign_coords(coords)

        # Shape coefficients appropriately
        coeff_arr = np.zeros(
            (len(ants), len(freqs), len(stok_str), len(coeffs)), dtype=complex
        )
        eta_arr_p = np.zeros(
            (len(ants), len(freqs), len(stok_str), len(coeffs)), dtype=float
        )
        eta_arr_q = np.zeros(
            (len(ants), len(freqs), len(stok_str), len(coeffs)), dtype=float
        )

        for antidx, ant in enumerate(ants):
            for freqidx, freq in enumerate(freqs):
                for stokidx, stok in enumerate(stok_num):
                    for coeffidx, coeff in enumerate(coeffs):

                        if "antenna" in df.columns:
                            tdf = df[
                                (df["antenna"] == ant)
                                & (df["freq"] == freq * 1e3)
                                & (df["stokes"] == stok)
                                & (df["ind"] == coeff)
                            ]
                        else:
                            tdf = df[
                                (df["freq"] == freq * 1e3)
                                & (df["stokes"] == stok)
                                & (df["ind"] == coeff)
                            ]

                        coeff_arr[antidx, freqidx, stokidx, coeffidx] = (
                            tdf["real"].values[0] + 1j * tdf["imag"].values[0]
                        )
                        eta_arr_p[antidx, freqidx, stokidx, coeffidx] = tdf[
                            "etax"
                        ].values[0]
                        eta_arr_q[antidx, freqidx, stokidx, coeffidx] = tdf[
                            "etay"
                        ].values[0]

        ds["coefficients"] = xr.DataArray(
            coeff_arr,
            dims=["antenna_index", "frequency", "polarization", "zernike_index"],
            coords=coords,
        )
        ds["eta_p"] = xr.DataArray(
            eta_arr_p,
            dims=["antenna_index", "frequency", "polarization", "zernike_index"],
            coords=coords,
        )
        ds["eta_q"] = xr.DataArray(
            eta_arr_q,
            dims=["antenna_index", "frequency", "polarization", "zernike_index"],
            coords=coords,
        )

        if outfile != "":
            ds.to_zarr(outfile, mode="w", compute=True, consolidated=True)

        return ds

    def read_zarr(self):
        """
        Read the Zernike coefficients from a Zarr file.
        """
        ds = xr.open_zarr(self.infile)
        return ds


class ZernikeFitter:
    """
    Class to fit the holography apertures with Zernike polynomials and store the
    fitted models and coefficients in Zarr or CSV formats.
    """

    def __init__(self, aperture_mds, fmt="zarr"):
        self.aperture_mds = aperture_mds
        self.fmt = fmt

        self.ntermsreal = 67
        self.ntermsimag = 67

        self.npol = 1  # Number of polarization planes in the data

        if not os.path.exists(self.aperture_mds):
            raise FileNotFoundError(f"File {self.aperture_mds} not found")

    def powl(self, base, exp):
        """
        Algorithm taken from https://stackoverflow.com/questions/2198138/calculating-powers-e-g-211-quickly
        """
        if exp == 0:
            return 1
        elif exp == 1:
            return base
        elif (exp & 1) != 0:
            return base * self.powl(base * base, exp // 2)
        else:
            return self.powl(base * base, exp // 2)

    def gen_zernike_surface(self, coeffs, x, y):
        """
        Use polynomial approximations to generate Zernike surface fast
        """

        # Setting coefficients array
        Z = np.zeros(67)
        if len(coeffs) < len(Z):
            c = Z.copy()
            c[: len(coeffs)] += coeffs
        else:
            c = Z.copy()
            c[: len(coeffs)] += coeffs

        # Setting the equations for the Zernike polynomials
        # r = np.sqrt(powl(x,2) + powl(y,2))
        # Summon cthulhu
        Z1 = c[0] * 1  # m = 0    n = 0
        Z2 = c[1] * x  # m = -1   n = 1
        Z3 = c[2] * y  # m = 1    n = 1
        Z4 = c[3] * 2 * x * y  # m = -2   n = 2
        Z5 = c[4] * (2 * self.powl(x, 2) + 2 * self.powl(y, 2) - 1)  # m = 0  n = 2
        Z6 = c[5] * (-1 * self.powl(x, 2) + self.powl(y, 2))  # m = 2  n = 2
        Z7 = c[6] * (-1 * self.powl(x, 3) + 3 * x * self.powl(y, 2))  # m = -3     n = 3
        Z8 = c[7] * (
            -2 * x + 3 * (self.powl(x, 3)) + 3 * x * (self.powl(y, 2))
        )  # m = -1   n = 3
        Z9 = c[8] * (
            -2 * y + 3 * self.powl(y, 3) + 3 * (self.powl(x, 2)) * y
        )  # m = 1    n = 3
        Z10 = c[9] * (self.powl(y, 3) - 3 * (self.powl(x, 2)) * y)  # m = 3 n =3
        Z11 = c[10] * (
            -4 * (self.powl(x, 3)) * y + 4 * x * (self.powl(y, 3))
        )  # m = -4    n = 4
        Z12 = c[11] * (
            -6 * x * y + 8 * (self.powl(x, 3)) * y + 8 * x * (self.powl(y, 3))
        )  # m = -2   n = 4
        Z13 = c[12] * (
            1
            - 6 * self.powl(x, 2)
            - 6 * self.powl(y, 2)
            + 6 * self.powl(x, 4)
            + 12 * (self.powl(x, 2)) * (self.powl(y, 2))
            + 6 * self.powl(y, 4)
        )  # m = 0  n = 4
        Z14 = c[13] * (
            3 * self.powl(x, 2)
            - 3 * self.powl(y, 2)
            - 4 * self.powl(x, 4)
            + 4 * self.powl(y, 4)
        )  # m = 2    n = 4
        Z15 = c[14] * (
            self.powl(x, 4)
            - 6 * (self.powl(x, 2)) * (self.powl(y, 2))
            + self.powl(y, 4)
        )  # m = 4   n = 4
        Z16 = c[15] * (
            self.powl(x, 5)
            - 10 * (self.powl(x, 3)) * self.powl(y, 2)
            + 5 * x * (self.powl(y, 4))
        )  # m = -5   n = 5
        Z17 = c[16] * (
            4 * self.powl(x, 3)
            - 12 * x * (self.powl(y, 2))
            - 5 * self.powl(x, 5)
            + 10 * (self.powl(x, 3)) * (self.powl(y, 2))
            + 15 * x * self.powl(y, 4)
        )  # m =-3     n = 5
        Z18 = c[17] * (
            3 * x
            - 12 * self.powl(x, 3)
            - 12 * x * (self.powl(y, 2))
            + 10 * self.powl(x, 5)
            + 20 * (self.powl(x, 3)) * (self.powl(y, 2))
            + 10 * x * (self.powl(y, 4))
        )  # m= -1  n = 5
        Z19 = c[18] * (
            3 * y
            - 12 * self.powl(y, 3)
            - 12 * y * (self.powl(x, 2))
            + 10 * self.powl(y, 5)
            + 20 * (self.powl(y, 3)) * (self.powl(x, 2))
            + 10 * y * (self.powl(x, 4))
        )  # m = 1  n = 5
        Z20 = c[19] * (
            -4 * self.powl(y, 3)
            + 12 * y * (self.powl(x, 2))
            + 5 * self.powl(y, 5)
            - 10 * (self.powl(y, 3)) * (self.powl(x, 2))
            - 15 * y * self.powl(x, 4)
        )  # m = 3   n = 5
        Z21 = c[20] * (
            self.powl(y, 5)
            - 10 * (self.powl(y, 3)) * self.powl(x, 2)
            + 5 * y * (self.powl(x, 4))
        )  # m = 5 n = 5
        Z22 = c[21] * (
            6 * (self.powl(x, 5)) * y
            - 20 * (self.powl(x, 3)) * (self.powl(y, 3))
            + 6 * x * (self.powl(y, 5))
        )  # m = -6 n = 6
        Z23 = c[22] * (
            20 * (self.powl(x, 3)) * y
            - 20 * x * (self.powl(y, 3))
            - 24 * (self.powl(x, 5)) * y
            + 24 * x * (self.powl(y, 5))
        )  # m = -4   n = 6
        Z24 = c[23] * (
            12 * x * y
            + 40 * (self.powl(x, 3)) * y
            - 40 * x * (self.powl(y, 3))
            + 30 * (self.powl(x, 5)) * y
            + 60 * (self.powl(x, 3)) * (self.powl(y, 3))
            - 30 * x * (self.powl(y, 5))
        )  # m = -2   n = 6
        Z25 = c[24] * (
            -1
            + 12 * (self.powl(x, 2))
            + 12 * (self.powl(y, 2))
            - 30 * (self.powl(x, 4))
            - 60 * (self.powl(x, 2)) * (self.powl(y, 2))
            - 30 * (self.powl(y, 4))
            + 20 * (self.powl(x, 6))
            + 60 * (self.powl(x, 4)) * self.powl(y, 2)
            + 60 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 20 * (self.powl(y, 6))
        )  # m = 0   n = 6
        Z26 = c[25] * (
            -6 * (self.powl(x, 2))
            + 6 * (self.powl(y, 2))
            + 20 * (self.powl(x, 4))
            - 20 * (self.powl(y, 4))
            - 15 * (self.powl(x, 6))
            - 15 * (self.powl(x, 4)) * (self.powl(y, 2))
            + 15 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 15 * (self.powl(y, 6))
        )  # m = 2   n = 6
        Z27 = c[26] * (
            -5 * (self.powl(x, 4))
            + 30 * (self.powl(x, 2)) * (self.powl(y, 2))
            - 5 * (self.powl(y, 4))
            + 6 * (self.powl(x, 6))
            - 30 * (self.powl(x, 4)) * self.powl(y, 2)
            - 30 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 6 * (self.powl(y, 6))
        )  # m = 4    n = 6
        Z28 = c[27] * (
            -1 * (self.powl(x, 6))
            + 15 * (self.powl(x, 4)) * (self.powl(y, 2))
            - 15 * (self.powl(x, 2)) * (self.powl(y, 4))
            + self.powl(y, 6)
        )  # m = 6   n = 6
        Z29 = c[28] * (
            -1 * (self.powl(x, 7))
            + 21 * (self.powl(x, 5)) * (self.powl(y, 2))
            - 35 * (self.powl(x, 3)) * (self.powl(y, 4))
            + 7 * x * (self.powl(y, 6))
        )  # m = -7    n = 7
        Z30 = c[29] * (
            -6 * (self.powl(x, 5))
            + 60 * (self.powl(x, 3)) * (self.powl(y, 2))
            - 30 * x * (self.powl(y, 4))
            + 7 * self.powl(x, 7)
            - 63 * (self.powl(x, 5)) * (self.powl(y, 2))
            - 35 * (self.powl(x, 3)) * (self.powl(y, 4))
            + 35 * x * (self.powl(y, 6))
        )  # m = -5    n = 7
        Z31 = c[30] * (
            -10 * (self.powl(x, 3))
            + 30 * x * (self.powl(y, 2))
            + 30 * self.powl(x, 5)
            - 60 * (self.powl(x, 3)) * (self.powl(y, 2))
            - 90 * x * (self.powl(y, 4))
            - 21 * self.powl(x, 7)
            + 21 * (self.powl(x, 5)) * (self.powl(y, 2))
            + 105 * (self.powl(x, 3)) * (self.powl(y, 4))
            + 63 * x * (self.powl(y, 6))
        )  # m =-3       n = 7
        Z32 = c[31] * (
            -4 * x
            + 30 * self.powl(x, 3)
            + 30 * x * (self.powl(y, 2))
            - 60 * (self.powl(x, 5))
            - 120 * (self.powl(x, 3)) * (self.powl(y, 2))
            - 60 * x * (self.powl(y, 4))
            + 35 * self.powl(x, 7)
            + 105 * (self.powl(x, 5)) * (self.powl(y, 2))
            + 105 * (self.powl(x, 3)) * (self.powl(y, 4))
            + 35 * x * (self.powl(y, 6))
        )  # m = -1  n = 7
        Z33 = c[32] * (
            -4 * y
            + 30 * self.powl(y, 3)
            + 30 * y * (self.powl(x, 2))
            - 60 * (self.powl(y, 5))
            - 120 * (self.powl(y, 3)) * (self.powl(x, 2))
            - 60 * y * (self.powl(x, 4))
            + 35 * self.powl(y, 7)
            + 105 * (self.powl(y, 5)) * (self.powl(x, 2))
            + 105 * (self.powl(y, 3)) * (self.powl(x, 4))
            + 35 * y * (self.powl(x, 6))
        )  # m = 1   n = 7
        Z34 = c[33] * (
            10 * (self.powl(y, 3))
            - 30 * y * (self.powl(x, 2))
            - 30 * self.powl(y, 5)
            + 60 * (self.powl(y, 3)) * (self.powl(x, 2))
            + 90 * y * (self.powl(x, 4))
            + 21 * self.powl(y, 7)
            - 21 * (self.powl(y, 5)) * (self.powl(x, 2))
            - 105 * (self.powl(y, 3)) * (self.powl(x, 4))
            - 63 * y * (self.powl(x, 6))
        )  # m =3     n = 7
        Z35 = c[34] * (
            -6 * (self.powl(y, 5))
            + 60 * (self.powl(y, 3)) * (self.powl(x, 2))
            - 30 * y * (self.powl(x, 4))
            + 7 * self.powl(y, 7)
            - 63 * (self.powl(y, 5)) * (self.powl(x, 2))
            - 35 * (self.powl(y, 3)) * (self.powl(x, 4))
            + 35 * y * (self.powl(x, 6))
        )  # m = 5  n = 7
        Z36 = c[35] * (
            self.powl(y, 7)
            - 21 * (self.powl(y, 5)) * (self.powl(x, 2))
            + 35 * (self.powl(y, 3)) * (self.powl(x, 4))
            - 7 * y * (self.powl(x, 6))
        )  # m = 7  n = 7
        Z37 = c[36] * (
            -8 * (self.powl(x, 7)) * y
            + 56 * (self.powl(x, 5)) * (self.powl(y, 3))
            - 56 * (self.powl(x, 3)) * (self.powl(y, 5))
            + 8 * x * (self.powl(y, 7))
        )  # m = -8  n = 8
        Z38 = c[37] * (
            -42 * (self.powl(x, 5)) * y
            + 140 * (self.powl(x, 3)) * (self.powl(y, 3))
            - 42 * x * (self.powl(y, 5))
            + 48 * (self.powl(x, 7)) * y
            - 112 * (self.powl(x, 5)) * (self.powl(y, 3))
            - 112 * (self.powl(x, 3)) * (self.powl(y, 5))
            + 48 * x * (self.powl(y, 7))
        )  # m = -6  n = 8
        Z39 = c[38] * (
            -60 * (self.powl(x, 3)) * y
            + 60 * x * (self.powl(y, 3))
            + 168 * (self.powl(x, 5)) * y
            - 168 * x * (self.powl(y, 5))
            - 112 * (self.powl(x, 7)) * y
            - 112 * (self.powl(x, 5)) * (self.powl(y, 3))
            + 112 * (self.powl(x, 3)) * (self.powl(y, 5))
            + 112 * x * (self.powl(y, 7))
        )  # m = -4   n = 8
        Z40 = c[39] * (
            -20 * x * y
            + 120 * (self.powl(x, 3)) * y
            + 120 * x * (self.powl(y, 3))
            - 210 * (self.powl(x, 5)) * y
            - 420 * (self.powl(x, 3)) * (self.powl(y, 3))
            - 210 * x * (self.powl(y, 5))
            - 112 * (self.powl(x, 7)) * y
            + 336 * (self.powl(x, 5)) * (self.powl(y, 3))
            + 336 * (self.powl(x, 3)) * (self.powl(y, 5))
            + 112 * x * (self.powl(y, 7))
        )  # m = -2   n = 8
        Z41 = c[40] * (
            1
            - 20 * self.powl(x, 2)
            - 20 * self.powl(y, 2)
            + 90 * self.powl(x, 4)
            + 180 * (self.powl(x, 2)) * (self.powl(y, 2))
            + 90 * self.powl(y, 4)
            - 140 * self.powl(x, 6)
            - 420 * (self.powl(x, 4)) * (self.powl(y, 2))
            - 420 * (self.powl(x, 2)) * (self.powl(y, 4))
            - 140 * (self.powl(y, 6))
            + 70 * self.powl(x, 8)
            + 280 * (self.powl(x, 6)) * (self.powl(y, 2))
            + 420 * (self.powl(x, 4)) * (self.powl(y, 4))
            + 280 * (self.powl(x, 2)) * (self.powl(y, 6))
            + 70 * self.powl(y, 8)
        )  # m = 0    n = 8
        Z42 = c[41] * (
            10 * self.powl(x, 2)
            - 10 * self.powl(y, 2)
            - 60 * self.powl(x, 4)
            + 105 * (self.powl(x, 4)) * (self.powl(y, 2))
            - 105 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 60 * self.powl(y, 4)
            + 105 * self.powl(x, 6)
            - 105 * self.powl(y, 6)
            - 56 * self.powl(x, 8)
            - 112 * (self.powl(x, 6)) * (self.powl(y, 2))
            + 112 * (self.powl(x, 2)) * (self.powl(y, 6))
            + 56 * self.powl(y, 8)
        )  # m = 2  n = 8
        Z43 = c[42] * (
            15 * self.powl(x, 4)
            - 90 * (self.powl(x, 2)) * (self.powl(y, 2))
            + 15 * self.powl(y, 4)
            - 42 * self.powl(x, 6)
            + 210 * (self.powl(x, 4)) * (self.powl(y, 2))
            + 210 * (self.powl(x, 2)) * (self.powl(y, 4))
            - 42 * self.powl(y, 6)
            + 28 * self.powl(x, 8)
            - 112 * (self.powl(x, 6)) * (self.powl(y, 2))
            - 280 * (self.powl(x, 4)) * (self.powl(y, 4))
            - 112 * (self.powl(x, 2)) * (self.powl(y, 6))
            + 28 * self.powl(y, 8)
        )  # m = 4     n = 8
        Z44 = c[43] * (
            7 * self.powl(x, 6)
            - 105 * (self.powl(x, 4)) * (self.powl(y, 2))
            + 105 * (self.powl(x, 2)) * (self.powl(y, 4))
            - 7 * self.powl(y, 6)
            - 8 * self.powl(x, 8)
            + 112 * (self.powl(x, 6)) * (self.powl(y, 2))
            - 112 * (self.powl(x, 2)) * (self.powl(y, 6))
            + 8 * self.powl(y, 8)
        )  # m = 6    n = 8
        Z45 = c[44] * (
            self.powl(x, 8)
            - 28 * (self.powl(x, 6)) * (self.powl(y, 2))
            + 70 * (self.powl(x, 4)) * (self.powl(y, 4))
            - 28 * (self.powl(x, 2)) * (self.powl(y, 6))
            + self.powl(y, 8)
        )  # m = 8     n = 9
        Z46 = c[45] * (
            self.powl(x, 9)
            - 36 * (self.powl(x, 7)) * (self.powl(y, 2))
            + 126 * (self.powl(x, 5)) * (self.powl(y, 4))
            - 84 * (self.powl(x, 3)) * (self.powl(y, 6))
            + 9 * x * (self.powl(y, 8))
        )  # m = -9     n = 9
        Z47 = c[46] * (
            8 * self.powl(x, 7)
            - 168 * (self.powl(x, 5)) * (self.powl(y, 2))
            + 280 * (self.powl(x, 3)) * (self.powl(y, 4))
            - 56 * x * (self.powl(y, 6))
            - 9 * self.powl(x, 9)
            + 180 * (self.powl(x, 7)) * (self.powl(y, 2))
            - 126 * (self.powl(x, 5)) * (self.powl(y, 4))
            - 252 * (self.powl(x, 3)) * (self.powl(y, 6))
            + 63 * x * (self.powl(y, 8))
        )  # m = -7    n = 9
        Z48 = c[47] * (
            21 * self.powl(x, 5)
            - 210 * (self.powl(x, 3)) * (self.powl(y, 2))
            + 105 * x * (self.powl(y, 4))
            - 56 * self.powl(x, 7)
            + 504 * (self.powl(x, 5)) * (self.powl(y, 2))
            + 280 * (self.powl(x, 3)) * (self.powl(y, 4))
            - 280 * x * (self.powl(y, 6))
            + 36 * self.powl(x, 9)
            - 288 * (self.powl(x, 7)) * (self.powl(y, 2))
            - 504 * (self.powl(x, 5)) * (self.powl(y, 4))
            + 180 * x * (self.powl(y, 8))
        )  # m = -5    n = 9
        Z49 = c[48] * (
            20 * self.powl(x, 3)
            - 60 * x * (self.powl(y, 2))
            - 105 * self.powl(x, 5)
            + 210 * (self.powl(x, 3)) * (self.powl(y, 2))
            + 315 * x * (self.powl(y, 4))
            + 168 * self.powl(x, 7)
            - 168 * (self.powl(x, 5)) * (self.powl(y, 2))
            - 840 * (self.powl(x, 3)) * (self.powl(y, 4))
            - 504 * x * (self.powl(y, 6))
            - 84 * self.powl(x, 9)
            + 504 * (self.powl(x, 5)) * (self.powl(y, 4))
            + 672 * (self.powl(x, 3)) * (self.powl(y, 6))
            + 252 * x * (self.powl(y, 8))
        )  # m = -3  n = 9
        Z50 = c[49] * (
            5 * x
            - 60 * self.powl(x, 3)
            - 60 * x * (self.powl(y, 2))
            + 210 * self.powl(x, 5)
            + 420 * (self.powl(x, 3)) * (self.powl(y, 2))
            + 210 * x * (self.powl(y, 4))
            - 280 * self.powl(x, 7)
            - 840 * (self.powl(x, 5)) * (self.powl(y, 2))
            - 840 * (self.powl(x, 3)) * (self.powl(y, 4))
            - 280 * x * (self.powl(y, 6))
            + 126 * self.powl(x, 9)
            + 504 * (self.powl(x, 7)) * (self.powl(y, 2))
            + 756 * (self.powl(x, 5)) * (self.powl(y, 4))
            + 504 * (self.powl(x, 3)) * (self.powl(y, 6))
            + 126 * x * (self.powl(y, 8))
        )  # m = -1   n = 9
        Z51 = c[50] * (
            5 * y
            - 60 * self.powl(y, 3)
            - 60 * y * (self.powl(x, 2))
            + 210 * self.powl(y, 5)
            + 420 * (self.powl(y, 3)) * (self.powl(x, 2))
            + 210 * y * (self.powl(x, 4))
            - 280 * self.powl(y, 7)
            - 840 * (self.powl(y, 5)) * (self.powl(x, 2))
            - 840 * (self.powl(y, 3)) * (self.powl(x, 4))
            - 280 * y * (self.powl(x, 6))
            + 126 * self.powl(y, 9)
            + 504 * (self.powl(y, 7)) * (self.powl(x, 2))
            + 756 * (self.powl(y, 5)) * (self.powl(x, 4))
            + 504 * (self.powl(y, 3)) * (self.powl(x, 6))
            + 126 * y * (self.powl(x, 8))
        )  # m = -1   n = 9
        Z52 = c[51] * (
            -20 * self.powl(y, 3)
            + 60 * y * (self.powl(x, 2))
            + 105 * self.powl(y, 5)
            - 210 * (self.powl(y, 3)) * (self.powl(x, 2))
            - 315 * y * (self.powl(x, 4))
            - 168 * self.powl(y, 7)
            + 168 * (self.powl(y, 5)) * (self.powl(x, 2))
            + 840 * (self.powl(y, 3)) * (self.powl(x, 4))
            + 504 * y * (self.powl(x, 6))
            + 84 * self.powl(y, 9)
            - 504 * (self.powl(y, 5)) * (self.powl(x, 4))
            - 672 * (self.powl(y, 3)) * (self.powl(x, 6))
            - 252 * y * (self.powl(x, 8))
        )  # m = 3  n = 9
        Z53 = c[52] * (
            21 * self.powl(y, 5)
            - 210 * (self.powl(y, 3)) * (self.powl(x, 2))
            + 105 * y * (self.powl(x, 4))
            - 56 * self.powl(y, 7)
            + 504 * (self.powl(y, 5)) * (self.powl(x, 2))
            + 280 * (self.powl(y, 3)) * (self.powl(x, 4))
            - 280 * y * (self.powl(x, 6))
            + 36 * self.powl(y, 9)
            - 288 * (self.powl(y, 7)) * (self.powl(x, 2))
            - 504 * (self.powl(y, 5)) * (self.powl(x, 4))
            + 180 * y * (self.powl(x, 8))
        )  # m = 5     n = 9
        Z54 = c[53] * (
            -8 * self.powl(y, 7)
            + 168 * (self.powl(y, 5)) * (self.powl(x, 2))
            - 280 * (self.powl(y, 3)) * (self.powl(x, 4))
            + 56 * y * (self.powl(x, 6))
            + 9 * self.powl(y, 9)
            - 180 * (self.powl(y, 7)) * (self.powl(x, 2))
            + 126 * (self.powl(y, 5)) * (self.powl(x, 4))
            - 252 * (self.powl(y, 3)) * (self.powl(x, 6))
            - 63 * y * (self.powl(x, 8))
        )  # m = 7     n = 9
        Z55 = c[54] * (
            self.powl(y, 9)
            - 36 * (self.powl(y, 7)) * (self.powl(x, 2))
            + 126 * (self.powl(y, 5)) * (self.powl(x, 4))
            - 84 * (self.powl(y, 3)) * (self.powl(x, 6))
            + 9 * y * (self.powl(x, 8))
        )  # m = 9       n = 9
        Z56 = c[55] * (
            10 * (self.powl(x, 9)) * y
            - 120 * (self.powl(x, 7)) * (self.powl(y, 3))
            + 252 * (self.powl(x, 5)) * (self.powl(y, 5))
            - 120 * (self.powl(x, 3)) * (self.powl(y, 7))
            + 10 * x * (self.powl(y, 9))
        )  # m = -10   n = 10
        Z57 = c[56] * (
            72 * (self.powl(x, 7)) * y
            - 504 * (self.powl(x, 5)) * (self.powl(y, 3))
            + 504 * (self.powl(x, 3)) * (self.powl(y, 5))
            - 72 * x * (self.powl(y, 7))
            - 80 * (self.powl(x, 9)) * y
            + 480 * (self.powl(x, 7)) * (self.powl(y, 3))
            - 480 * (self.powl(x, 3)) * (self.powl(y, 7))
            + 80 * x * (self.powl(y, 9))
        )  # m = -8    n = 10
        Z58 = c[57] * (
            270 * (self.powl(x, 9)) * y
            - 360 * (self.powl(x, 7)) * (self.powl(y, 3))
            - 1260 * (self.powl(x, 5)) * (self.powl(y, 5))
            - 360 * (self.powl(x, 3)) * (self.powl(y, 7))
            + 270 * x * (self.powl(y, 9))
            - 432 * (self.powl(x, 7)) * y
            + 1008 * (self.powl(x, 5)) * (self.powl(y, 3))
            + 1008 * (self.powl(x, 3)) * (self.powl(y, 5))
            - 432 * x * (self.powl(y, 7))
            + 168 * (self.powl(x, 5)) * y
            - 560 * (self.powl(x, 3)) * (self.powl(y, 3))
            + 168 * x * (self.powl(y, 5))
        )  # m = -6   n = 10
        Z59 = c[58] * (
            140 * (self.powl(x, 3)) * y
            - 140 * x * (self.powl(y, 3))
            - 672 * (self.powl(x, 5)) * y
            + 672 * x * (self.powl(y, 5))
            + 1008 * (self.powl(x, 7)) * y
            + 1008 * (self.powl(x, 5)) * (self.powl(y, 3))
            - 1008 * (self.powl(x, 3)) * (self.powl(y, 5))
            - 1008 * x * (self.powl(y, 7))
            - 480 * (self.powl(x, 9)) * y
            - 960 * (self.powl(x, 7)) * (self.powl(y, 3))
            + 960 * (self.powl(x, 3)) * (self.powl(y, 7))
            + 480 * x * (self.powl(y, 9))
        )  # m = -4   n = 10
        Z60 = c[59] * (
            30 * x * y
            - 280 * (self.powl(x, 3)) * y
            - 280 * x * (self.powl(y, 3))
            + 840 * (self.powl(x, 5)) * y
            + 1680 * (self.powl(x, 3)) * (self.powl(y, 3))
            + 840 * x * (self.powl(y, 5))
            - 1008 * (self.powl(x, 7)) * y
            - 3024 * (self.powl(x, 5)) * (self.powl(y, 3))
            - 3024 * (self.powl(x, 3)) * (self.powl(y, 5))
            - 1008 * x * (self.powl(y, 7))
            + 420 * (self.powl(x, 9)) * y
            + 1680 * (self.powl(x, 7)) * (self.powl(y, 3))
            + 2520 * (self.powl(x, 5)) * (self.powl(y, 5))
            + 1680 * (self.powl(x, 3)) * (self.powl(y, 7))
            + 420 * x * (self.powl(y, 9))
        )  # m = -2   n = 10
        Z61 = c[60] * (
            -1
            + 30 * self.powl(x, 2)
            + 30 * self.powl(y, 2)
            - 210 * self.powl(x, 4)
            - 420 * (self.powl(x, 2)) * (self.powl(y, 2))
            - 210 * self.powl(y, 4)
            + 560 * self.powl(x, 6)
            + 1680 * (self.powl(x, 4)) * (self.powl(y, 2))
            + 1680 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 560 * self.powl(y, 6)
            - 630 * self.powl(x, 8)
            - 2520 * (self.powl(x, 6)) * (self.powl(y, 2))
            - 3780 * (self.powl(x, 4)) * (self.powl(y, 4))
            - 2520 * (self.powl(x, 2)) * (self.powl(y, 6))
            - 630 * self.powl(y, 8)
            + 252 * self.powl(x, 10)
            + 1260 * (self.powl(x, 8)) * (self.powl(y, 2))
            + 2520 * (self.powl(x, 6)) * (self.powl(y, 4))
            + 2520 * (self.powl(x, 4)) * (self.powl(y, 6))
            + 1260 * (self.powl(x, 2)) * (self.powl(y, 8))
            + 252 * self.powl(y, 10)
        )  # m = 0    n = 10
        Z62 = c[61] * (
            -15 * self.powl(x, 2)
            + 15 * self.powl(y, 2)
            + 140 * self.powl(x, 4)
            - 140 * self.powl(y, 4)
            - 420 * self.powl(x, 6)
            - 420 * (self.powl(x, 4)) * (self.powl(y, 2))
            + 420 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 420 * self.powl(y, 6)
            + 504 * self.powl(x, 8)
            + 1008 * (self.powl(x, 6)) * (self.powl(y, 2))
            - 1008 * (self.powl(x, 2)) * (self.powl(y, 6))
            - 504 * self.powl(y, 8)
            - 210 * self.powl(x, 10)
            - 630 * (self.powl(x, 8)) * (self.powl(y, 2))
            - 420 * (self.powl(x, 6)) * (self.powl(y, 4))
            + 420 * (self.powl(x, 4)) * (self.powl(y, 6))
            + 630 * (self.powl(x, 2)) * (self.powl(y, 8))
            + 210 * self.powl(y, 10)
        )  # m = 2  n = 10
        Z63 = c[62] * (
            -35 * self.powl(x, 4)
            + 210 * (self.powl(x, 2)) * (self.powl(y, 2))
            - 35 * self.powl(y, 4)
            + 168 * self.powl(x, 6)
            - 840 * (self.powl(x, 4)) * (self.powl(y, 2))
            - 840 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 168 * self.powl(y, 6)
            - 252 * self.powl(x, 8)
            + 1008 * (self.powl(x, 6)) * (self.powl(y, 2))
            + 2520 * (self.powl(x, 4)) * (self.powl(y, 4))
            + 1008 * (self.powl(x, 2)) * (self.powl(y, 6))
            - 252 * (self.powl(y, 8))
            + 120 * self.powl(x, 10)
            - 360 * (self.powl(x, 8)) * (self.powl(y, 2))
            - 1680 * (self.powl(x, 6)) * (self.powl(y, 4))
            - 1680 * (self.powl(x, 4)) * (self.powl(y, 6))
            - 360 * (self.powl(x, 2)) * (self.powl(y, 8))
            + 120 * self.powl(y, 10)
        )  # m = 4     n = 10
        Z64 = c[63] * (
            -28 * self.powl(x, 6)
            + 420 * (self.powl(x, 4)) * (self.powl(y, 2))
            - 420 * (self.powl(x, 2)) * (self.powl(y, 4))
            + 28 * self.powl(y, 6)
            + 72 * self.powl(x, 8)
            - 1008 * (self.powl(x, 6)) * (self.powl(y, 2))
            + 1008 * (self.powl(x, 2)) * (self.powl(y, 6))
            - 72 * self.powl(y, 8)
            - 45 * self.powl(x, 10)
            + 585 * (self.powl(x, 8)) * (self.powl(y, 2))
            + 630 * (self.powl(x, 6)) * (self.powl(y, 4))
            - 630 * (self.powl(x, 4)) * (self.powl(y, 6))
            - 585 * (self.powl(x, 2)) * (self.powl(y, 8))
            + 45 * self.powl(y, 10)
        )  # m = 6    n = 10
        Z65 = c[64] * (
            -9 * self.powl(x, 8)
            + 252 * (self.powl(x, 6)) * (self.powl(y, 2))
            - 630 * (self.powl(x, 4)) * (self.powl(y, 4))
            + 252 * (self.powl(x, 2)) * (self.powl(y, 6))
            - 9 * self.powl(y, 8)
            + 10 * self.powl(x, 10)
            - 270 * (self.powl(x, 8)) * (self.powl(y, 2))
            + 420 * (self.powl(x, 6)) * (self.powl(y, 4))
            + 420 * (self.powl(x, 4)) * (self.powl(y, 6))
            - 270 * (self.powl(x, 2)) * (self.powl(y, 8))
            + 10 * self.powl(y, 10)
        )  # m = 8    n = 10
        Z66 = c[65] * (
            -1 * self.powl(x, 10)
            + 45 * (self.powl(x, 8)) * (self.powl(y, 2))
            - 210 * (self.powl(x, 6)) * (self.powl(y, 4))
            + 210 * (self.powl(x, 4)) * (self.powl(y, 6))
            - 45 * (self.powl(x, 2)) * (self.powl(y, 8))
            + self.powl(y, 10)
        )  # m = 10   n = 10

        ZW = (
            Z1
            + Z2
            + Z3
            + Z4
            + Z5
            + Z6
            + Z7
            + Z8
            + Z9
            + Z10
            + Z11
            + Z12
            + Z13
            + Z14
            + Z15
            + Z16
            + Z17
            + Z18
            + Z19
            + Z20
            + Z21
            + Z22
            + Z23
            + Z24
            + Z25
            + Z26
            + Z27
            + Z28
            + Z29
            + Z30
            + Z31
            + Z32
            + Z33
            + Z34
            + Z35
            + Z36
            + Z37
            + Z38
            + Z39
            + Z40
            + Z41
            + Z42
            + Z43
            + Z44
            + Z45
            + Z46
            + Z47
            + Z48
            + Z49
            + Z50
            + Z51
            + Z52
            + Z53
            + Z54
            + Z55
            + Z56
            + Z57
            + Z58
            + Z59
            + Z60
            + Z61
            + Z62
            + Z63
            + Z64
            + Z65
            + Z66
        )
        return ZW

    def fit_minimize(self, coeffs, aperture, xx, yy, maskidx):
        surf = self.gen_zernike_surface(coeffs, xx, yy)
        surf[maskidx] = 0
        return (aperture - surf).flatten()

    def fit_complex_aperture(self, antenna, ddi="ddi_0"):
        """
        Given the antenna and DDI, fit the full Jones complex aperture.
        The fit is performed to the phase corrected aperture plane, i.e.,
        the AMPLITUDE and CORRECTED_PHASE columns of the input Zarr.

        :param antenna: The antenna name to fit
        :type antenna: str

        :param ddi: The DDI to select, default is 'ddi_0'
        :type ddi: str

        :return: The Zarr file with the Zernike coefficients for the fitted
        aperture
        :rtype: Xarray.Dataset
        """

        from astrohack.dio import open_image
        from astrohack._utils import Telescope
        from scipy.optimize import leastsq

        image_mds = open_image(self.aperture_mds)
        self.npol = image_mds[antenna][ddi].coords["pol"].size

        fit_aperture = np.zeros_like(
            image_mds[antenna][ddi]["AMPLITUDE"], dtype=complex
        )
        res_aperture = np.zeros_like(
            image_mds[antenna][ddi]["AMPLITUDE"], dtype=complex
        )

        for pp in range(1):
            tmp_mds = image_mds[antenna][ddi].isel(pol=pp)

            aperture_amp = np.asarray(np.squeeze(tmp_mds["AMPLITUDE"]))
            aperture_phase = np.asarray(np.squeeze(tmp_mds["CORRECTED_PHASE"]))

            aperture_real = np.asarray(aperture_amp * np.cos(aperture_phase))
            aperture_imag = np.asarray(aperture_amp * np.sin(aperture_phase))

            print("aperture_real.shape, aperture_imag.shape")
            print(aperture_real.shape, aperture_imag.shape)

            npixx = aperture_amp.shape[0]
            npixy = aperture_amp.shape[1]

            # Generate mask
            x = np.linspace(-1, 1, npixx)
            y = np.linspace(-1, 1, npixy)

            yy, xx = np.meshgrid(x, y)

            # _holog.py trims at 1.1 * radius, so we need to trim more aggressively before fitting
            maskidx = np.where(np.hypot(xx, yy) > 0.9)

            coeffsreal = np.ones(self.ntermsreal)
            coeffsimag = np.ones(self.ntermsimag)

            # Fit real and imaginary parts separately
            coeffsreal = leastsq(
                self.fit_minimize,
                np.ones(self.ntermsreal),
                args=(aperture_amp, xx, yy, maskidx),
            )[0]
            coeffsimag = leastsq(
                self.fit_minimize,
                np.ones(self.ntermsimag),
                args=(aperture_phase, xx, yy, maskidx),
            )[0]

            # Only keep the terms we want, even if real and imag
            # have different number of terms
            maxterms = max(self.ntermsreal, self.ntermsimag)
            initreal = np.zeros(maxterms)
            initimag = np.zeros(maxterms)

            initreal[: self.ntermsreal] = coeffsreal
            initimag[: self.ntermsimag] = coeffsimag

            coeffsreal = np.copy(initreal)
            coeffsimag = np.copy(initimag)

            fitreal = self.gen_zernike_surface(coeffsreal, xx, yy)
            fitimag = self.gen_zernike_surface(coeffsimag, xx, yy)

            fitreal[maskidx] = 0
            fitimag[maskidx] = 0

            fit_aperture[0, 0, pp, :, :] = fitreal + 1j * fitimag
            aperture_complex = aperture_real + 1j * aperture_imag
            res_aperture[0, 0, pp, :, :] = (
                aperture_complex - fit_aperture[0, 0, pp, :, :]
            )

            print(fit_aperture.shape, image_mds[antenna][ddi]["AMPLITUDE"].shape)

            # image_mds[antenna][ddi]['APERTURE_FIT'] = fitaperture
            # image_mds[antenna][ddi]['APERTURE_RESIDUAL'] = resaperture

        return coeffsreal, coeffsimag, fit_aperture, res_aperture
