# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: orientation/spectrum_method.py                                    #
# Description: Estimate crystal orientation based on a single spectrum using  #
# a modified version of the method by Asimow et al. (2006)                    #
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
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Dict, Any

from ..transmittance_model import calc_transmittance
from ..crystallography import convert_Euler_to_Asimow_angles
from ..spectrum_tools import smooth_spectrum
from .common import get_initial_guesses, summarize_results

# Function definitions


def find_orientation_based_on_spectrum(
    standard: pd.DataFrame,
    spectrum: np.ndarray,
    angles: str = "Asimow",
    algorithm: str | Tuple[str] = "all",
    thickness_bound: None | Tuple[float] = None,
    smooth_window: int = 21,
    num_guesses: int = 15,
) -> Dict[str, Any]:
    """
    Estimate Asimow angles (θ, φ) (and optionally thickness d)
    by minimizing derivative misfit.

    Parameters
    ----------
    standard : pd.DataFrame
        A pandas dataframe containing the normalized transmitance
        spectra for principal directions (Ta, Tb, Tc) as a function
        of wavelengths. Is assumes that column names are "Ta", "Tb",
        "Tc" and "wavenumber", respectively.
    spectrum : array-like or pd.Series
        Measured transmittance spectrum.
    angles : str, optional
        either orientation in Asimow angles, or in Euler
        (Bunge convention). Can be "Asimow" or "Euler"
    algorithm : str or Tuple[str], optional
        the minimization algorithm to use, by default "all"
        Can be: "gradient", "diffevol", "all" or a Tuple
        with different algorithms, by default "all".
    thickness_bound : Tuple of size 2 or None, optional
        Whether to fit thickness as well, defaults to None
    smooth_window : int, optional
        Number of samples to include in the moving average
        (must be odd), defaults to 21
    num_guesses : int, optional
        Number of initial guesses when using the gradient-based
        algorithm, by default 15

    Returns
    -------
    dict
        Best-fit parameters and corresponding misfit.
    """

    # Sanity checks
    algorithm = algorithm.lower()
    if algorithm not in {"all", "gradient", "diffevol"}:
        raise ValueError('Unsupported algorithm. Use "all", "gradienr" or "diffevol"')
    if algorithm == "all":
        algorithm = ["gradient", "diffevol"]

    results = {}  # initialize dict

    # Set parameter bounds
    if thickness_bound is not None:
        thickness = thickness_bound
    else:
        thickness = (1, 1)  # set thickness to 1

    if angles == "Asimow":
        theta_ang_rad = (0, 2 * np.pi)
        phi_ang_rad = (0, np.pi)
        bounds = [theta_ang_rad, phi_ang_rad, thickness]
        angle_format = "asimow_rad"
    elif angles == "Euler":  # the bounds below assumes an orthorhombic symmetry
        euler1_rad = (0, np.pi / 2)
        euler2_rad = (0, np.pi / 2)
        euler3_rad = (0, np.pi)
        bounds = [euler1_rad, euler2_rad, euler3_rad, thickness]
        angle_format = "euler_rad"
    else:
        raise ValueError('Unsupported orientation method. Use "Asimow" or "Euler"')

    # MINIMIZATION BASED ON L-BFGS-B algorithm (gradient-based optimization)
    if "gradient" in algorithm:

        best_result = None
        best_error = float("inf")

        # compute initial guesses
        initial_guesses = get_initial_guesses(bounds, num_guesses)

        start_time = time.time()
        # run optimization
        for guess in initial_guesses:
            # generate initial guesses
            result_bfgs = minimize(
                fun=_error_function,
                x0=guess,
                args=(
                    standard["wavenumber"],
                    spectrum,
                    standard["Ta"],
                    standard["Tb"],
                    standard["Tc"],
                    smooth_window,
                ),
                bounds=bounds,
                method="L-BFGS-B",
            )

            # Update result if the current one is better
            if result_bfgs.fun < best_error:
                best_result = result_bfgs
                best_error = result_bfgs.fun

        end_time = time.time()
        results["gradient_based"] = best_result
        elapsed_time = end_time - start_time

        # print/summarize results
        summarize_results(
            best_result,
            elapsed_time,
            alg_name="L-BFGS-B",
            angle_format=angle_format,
            num_guesses=num_guesses,
        )

    # MINIMIZATION BASED ON differential evolution algorithm
    if "diffevol" in algorithm:
        start_time = time.time()

        # run optimization
        result_diff = differential_evolution(
            func=_error_function,
            bounds=bounds,
            args=(
                standard["wavenumber"],
                spectrum,
                standard["Ta"],
                standard["Tb"],
                standard["Tc"],
                smooth_window,
            ),
        )

        end_time = time.time()
        results["differential_evol"] = result_diff
        elapsed_time = end_time - start_time

        summarize_results(
            result_diff, elapsed_time, alg_name="diffEvol", angle_format=angle_format
        )

    print("DONE")
    print(" ")

    return results


# ============================================================================ #
# AUXILIARY FUNCTIONS. Not intended for direct user access.                    #
# ============================================================================ #


def _error_function(
    params: np.ndarray,
    wavelengths: np.ndarray,
    T_measured: np.ndarray,
    Ta_values: np.ndarray,
    Tb_values: np.ndarray,
    Tc_values: np.ndarray,
    smooth_window: int = 9,
) -> float:
    """
    Compute misfit between measured and synthetic derivatives.

    Parameters
    ----------
    params : numpy.ndarray
        Parameters to optimize: [theta_rad, phi_rad, thickness]
        or [euler1, euler2, euler3, thickness]
    wavelengths : numpy.ndarray
        λ values for data points.
    T_measured : numpy.ndarray
        Measured transmittance spectrum.
    Ta_values : numpy.ndarray
        Array of Ta values corresponding to each wavelength
    Tb_values : numpy.ndarray
        Array of Tb values corresponding to each wavelength
    Tc_values : numpy.ndarray
        Array of Tc values corresponding to each wavelength
    smooth_window : int
        Window size for moving average smoothing(odd integer).

    Returns
    -------
    float
        Sum of squared differences between derivatives of measured
        and calculated spectra
    """
    # Extract parameters
    if len(params) == 3:
        theta_rad, phi_rad, d = params
    else:
        e1, e2, e3, d = params
        phi_rad, theta_rad = convert_Euler_to_Asimow_angles([e1, e2, e3], degrees=False)

    # Calculate synthetic spectrum, smooth and derive
    T_calculated = calc_transmittance(
        standard=(Ta_values, Tb_values, Tc_values),
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        d=d,
    )
    T_calculated_smooth = smooth_spectrum(T_calculated, window_size=smooth_window)
    dT_calculated = _calc_derivative(wavelengths, T_calculated_smooth)

    # Smooth and derive measured spectrum
    T_measured_smooth = smooth_spectrum(T_measured, window_size=smooth_window)
    dT_measured = _calc_derivative(wavelengths, T_measured_smooth)

    # calculate the misfit
    return np.sum((dT_measured - dT_calculated) ** 2)


def _calc_derivative(wavelengths: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Compute the numerical derivative dT/dλ using
    central differences.

    Parameters
    ----------
    wavelengths : array-like
        Increasing sequence of λ values.
    values : array-like
        Spectrum values at each λ.

    Returns
    -------
    numpy.ndarray
        First derivative of transmittance with
        respect to wavelength
    """

    return np.gradient(values, wavelengths)


# End of function definitions

# End of file
