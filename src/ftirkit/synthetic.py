# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: synthetic.py                                                      #
# Description: Tools to generate synthetic FTIR data                          #
#                                                                             #
# SPDX-License-Identifier: Apache-2.0                                         #
# Copyright (c) 2025 Marco A. Lopez-Sanchez. All rights reserved.             #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the License);               #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at                                     #
#                                                                             #
#     http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an AS IS BASIS,             #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
# Author: Marco A. Lopez-Sanchez                                              #
# ORCID: http://orcid.org/0000-0002-0261-9267                                 #
# Website: https://marcoalopez.github.io/FTIRkit /                            #
# Repository: https://github.com/marcoalopez/FTIRkit                          #
# =========================================================================== #


# Import statements
import numpy as np
import pandas as pd
from contourpy import contour_generator

from .transmittance_model import calc_transmittance
from .geometry import sph2cart, regular_S2_grid, rotate
from .spectrum_tools import transmittance_to_absorbance


# Function definitions

def generate_spectra(
    theta_deg: float | np.ndarray,
    phi_deg: float | np.ndarray,
    standard: pd.DataFrame,
    thickness: float = 1.0,
    output: str = "absorbance",
    wavelength_colname: str = "wavenumber",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate FTIR spectra for one or more orientations based on a FTIR standard.

    Parameters
    ----------
    theta_deg : float or array-like of float
        Azimuthal angle(s) theta in Asimov's reference frame, in degrees.
        May be given as a scalar (single orientation) or 1D array-like.
        All values must lie within [0, 360] degrees.
    phi_deg : float or array-like of float
        Polar angle(s) phi in Asimov's reference frame, in degrees.
        May be given as a scalar (single orientation) or 1D array-like.
        All values must lie within [0, 180] degrees.
    standard : pandas.DataFrame
        DataFrame containing transmittance spectra along the principal
        crystal axes. It must contain the columns "Ta", "Tb", "Tc"
        and a wavelength column given by wavelength_colname.
    thickness : float, optional
        Sample thickness relative to the standard, by default 1.0.
        This value is passed directly to calc_transmittance as d.
    output : {"absorbance", "transmittance"}, optional
        Desired output type. If "absorbance" (default), spectra are
        returned as absorbance (-log10(T)). If "transmittance",
        raw transmittance is returned.
    wavelength_colname : str, optional
        Name of the wavelength/wavenumber column in standard to be
        copied into the output spectra DataFrame, by default "wavenumber".

    Returns
    -------
    spectra : pandas.DataFrame
        FTIR spectra for each requested orientation. The first column is
        the wavelength/wavenumber (copied from standard), followed by
        one column per spectrum: "spec1", "spec2", etc.
    angles : pandas.DataFrame
        Table describing the orientations used to generate each spectrum,
        with columns "reference", "phi_deg" and "theta_deg".
    """

    # CHECKS ##################################################
    theta_arr = np.atleast_1d(np.asarray(theta_deg, dtype=float))
    phi_arr = np.atleast_1d(np.asarray(phi_deg, dtype=float))

    if theta_arr.shape != phi_arr.shape:
        raise ValueError("theta_deg and phi_deg must have the same number of angles")

    if theta_arr.ndim != 1:
        raise ValueError(
            "theta_deg and phi_deg must be scalars or 1D array-like objects."
        )

    if np.any((phi_arr < 0) | (phi_arr > 180)):
        raise ValueError("All phi angles must lie within [0, 180] degrees.")
    if np.any((theta_arr < 0) | (theta_arr > 360)):
        raise ValueError("All theta angles must lie within [0, 360] degrees.")

    # validate standard DataFrame
    required_cols = {"Ta", "Tb", "Tc"}
    missing = required_cols.difference(standard.columns)

    if missing:
        raise KeyError(f"`standard` is missing required column(s): {sorted(missing)}.")

    if wavelength_colname not in standard.columns:
        raise KeyError(
            f"`standard` does not contain the wavelength column '{wavelength_colname}'."
        )

    output_lower = output.lower()
    if output_lower not in {"absorbance", "transmittance"}:
        raise ValueError(
            "Parameter `output` must be either 'absorbance' or 'transmittance'."
        )

    # PROCEDURE ##########################################
    spectra = pd.DataFrame(index=standard.index)
    column_names: list[str] = []

    ta = standard["Ta"]
    tb = standard["Tb"]
    tc = standard["Tc"]

    for i, (phi, theta) in enumerate(zip(phi_arr, theta_arr), start=1):
        col_name = f"spec{i}"
        column_names.append(col_name)

        spectra[col_name] = calc_transmittance(
            standard=(ta, tb, tc),
            theta_rad=np.deg2rad(theta),
            phi_rad=np.deg2rad(phi),
            d=thickness,
        )

    if output_lower == "absorbance":
        # conversion clips T to the physical/numerical range internally
        spectra = spectra.apply(transmittance_to_absorbance)
    else:
        # Clip to physical/numerical range to avoid degenerate values
        eps = np.finfo(float).eps
        spectra = spectra.clip(lower=eps, upper=1.0)

    spectra.insert(
        0,
        wavelength_colname,
        standard[wavelength_colname].to_numpy(),
    )

    # build angles table
    angles = pd.DataFrame(
        {
            "reference": column_names,
            "phi_deg": phi_arr,
            "theta_deg": theta_arr,
        }
    )

    return spectra, angles


def generate_section(
    euler_deg: np.ndarray,
    standard_Ts_1mm: np.ndarray,
    thickness: float = 1.0,
    sample_size: int = 16,
    azimuth_range_deg: int = 360,
    noise=None,
    grid_resolution: int = 288,
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    euler_deg : np.ndarray
        Euler angles in degrees
    standard_Ts_1mm : np.ndarray
        transmittance reference principal values at a
        specific wavenumber
    thickness : float, optional
        thickness relative to 1 mm standard, by default 1.
    sample_size : int, optional
        _description_, by default 16
    azimuth_range_deg : int, optional
        _description_, by default 360
    noise : _type_, optional
        _description_, by default None

    Returns
    -------
    pd.DataFrame
        _description_
    """

    # Generate a mesh of orientations
    polar_ang, azimuths = regular_S2_grid(n_squared=grid_resolution)

    # Calculate the transmittance envelope
    T = calc_transmittance(
        standard=standard_Ts_1mm, theta_rad=azimuths, phi_rad=polar_ang, d=thickness
    )
    x, y, z = sph2cart(T, azimuths, polar_ang)

    # generate an array with a set of angles respect to polarizer
    step = azimuth_range_deg / sample_size
    ang2pol = np.arange(0, azimuth_range_deg, step)

    # replace zero Euler values to avoid artifacts with contourpy
    euler_deg = np.asarray(euler_deg)  # ensure a numpy array
    euler_deg[np.isclose(euler_deg, 0, atol=1e-8)] = 1e-3

    # rotate the envelope to the defined orientation
    x2, y2, z2 = rotate(coordinates=(x, y, z), euler_deg=euler_deg)

    # get the XY section and append specific values
    xy_section = extract_XY_section_ang(x2, y2, z2)
    indexes = _find_nearest(xy_section["angles"], ang2pol)
    T_vals = xy_section.loc[indexes, ["T"]].values.T.tolist()[0]
    angle_deg = xy_section.loc[indexes, ["angles"]].values.T.tolist()[0]

    # create a dataframe
    data = pd.DataFrame({"T_values": T_vals, "angle_deg": angle_deg})

    if noise is not None:
        rgn = np.random.default_rng()
        std_T, std_azimuths = noise
        data["T_vals_noise"] = data["T_values"] + rgn.normal(
            0, std_T, data["T_values"].shape
        )
        data["angle_noise"] = data["angle_deg"] + rgn.normal(
            0, std_azimuths, data["angle_deg"].shape
        )

    return data


# ============================================================================ #
# EXTRACT ENVELOPE SECTIONS                                                    #
# ============================================================================ #


def extract_XY_section_ang(x, y, z) -> pd.DataFrame:
    """
    It uses ContourPy to determine the section of an envelope at
    z=0 (XY plate). It returns a DataFrame with the x and y
    coordinates, the distance from the origin (the T value),
    and the angle with respect to x in radians.

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    z : _type_
        _description_
    Returns
    -------
    pandas.Dataframe
        _description_
    """

    # estimate the contour at z=0 (i.e. XY plane)
    sections = contour_generator(x, y, z)

    # get the vertice coordinates of the contour z=0 (array-like)
    coordinates = sections.lines(0)[0]

    # get vector lengths (i.e. T values within the XY plane)
    T = np.linalg.norm(coordinates, axis=1)

    # get the angle of the vector (in radians)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    # Convert angles to the range 0-2π (0-360 degrees)
    angles = np.degrees(angles) % 360

    df = pd.DataFrame(
        {"x": coordinates[:, 0], "y": coordinates[:, 1], "T": T, "angles": angles}
    )

    return df


# ============================================================================ #
# AUXILIARY FUNCTIONS. Not intended for direct user access.                    #
# ============================================================================ #

def _find_nearest(df, values):
    """find the index of the nearest value in
    a pandas dataframe

    Parameters
    ----------
    df : _type_
        _description_
    values : _type_
        _description_
    """

    indexes = []
    for value in values:
        indexes.append((np.abs(df - value)).idxmin())
    return indexes


# End of function definitions

# End of file
