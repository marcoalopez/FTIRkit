# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: orientation/lambda_method.py                                      #
# Description: Estimate crystal orientation based on the section-wavelength   #
# method of Lopez-Sanchez and Padrón-Navarta (2026)                           #
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
from scipy.optimize import minimize, differential_evolution, dual_annealing
from typing import Tuple, Dict, Any, List

from ..transmittance_model import calc_transmittance
from ..geometry import sph2cart, cart2sph, rotate
from .common import get_initial_guesses, explore_Euler_space, summarize_results

# Function definitions

def extract_section_from_spectra(
    spectra: pd.DataFrame,
    angles2pol_deg: np.array,
    wavenumber: float = 1987.29,
) -> pd.DataFrame:
    """
    Extract a section of the transmittance envelope
    from a set of spectra at a specific wavenumber
    and returns a data frame containing two columns:
    the transmittance values and the angle to the
    polariser in degrees. The data frame is ready
    to be used for section-based orientation analysis.
    
    TODO: automatically select the wavenumber closest
    to the chosen number without having to specify the
    exact number

    Parameters
    ----------
    spectra : pd.DataFrame
        _description_
    angles2pol_deg : np.array | pd.Series
        _description_
    wavenumber : float, optional
        _description_, by default 1987.29

    Returns
    -------
    pd.DataFrame
        _description_
    """

    # TODO: input checking

    # extract transmitance values for specific wavenumber
    t_values = (
        spectra.query("wavenumber == @wavenumber")
        .drop(["wavenumber"], axis=1)
        .to_numpy()
        .flatten()
    )

    # check
    if len(t_values) != len(angles2pol_deg):
        raise ValueError(
            "The number of angles must correspond to the number of spectra. Check your inputs!"
        )

    table = {"T_values": t_values, "ang2pol_deg": angles2pol_deg}

    return pd.DataFrame(table)


# ============================================================================ #
# MINIMIZATION PROCEDURE                                                       #
# ============================================================================ #


def find_orientation_based_on_lambda(
    transmitances: np.ndarray,
    angles2pol_deg: np.ndarray,
    principal_Ts: np.ndarray,
    algorithm: str | List[str] = "all",
    num_guesses: int = 20,
    upper_bounds: Tuple = (90., 89.99, 180.),
    thickness_bounds: None | Tuple[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Determine the crystallographic orientation from a set of
    FTIR data polarised at a specific wavelength using the
    envelope section method. It can use one or several minimisation
    algorithms in parallel.

    Parameters
    ----------
    transmitances : array-like
        Transmittance measurements at a specific wavelength.
    angles2pol_deg : array-like
        The angle between the polarization direction and the specimen
        reference in degrees at which the transmittance measurements
        were taken.
    principal_Ts : tuple of size 3
        tuple containing the transmittance values along a-axis (Ta),
        b-axis (Tb), and c-axis (Tc). -> (Ta, Tb, Tc)
    algorithm : str or list[str], optional
        the minimization algorithm to use, by default "all"
        Can be: "gradient", "diffEvol", "annealing", "all"
        or a list with different algorithms, by default "all"
    num_guesses : int, optional
        Number of initial guesses when using the gradient-based
        algorithm, by default 20
    upper_bounds : tuple, optional
        the upper bounds of Euler angles defining the fundamental
        zone of solutions, by default (90., 89.99, 180.)
    thickness_bounds : Tuple of size 2 or None, optional
        Whether to fit thickness as well, defaults to None

    Returns
    -------
    dict
        a Python dict with all the minimization outpus
    """

    # set some initial conditions
    results = {}  # initialize dict
    if algorithm == "all":
        algorithm = ["gradient", "diffEvol", "annealing"]

    # organize mesurements in an Numpy array of shape (n, 3)
    measurements = np.column_stack(
        (transmitances, angles2pol_deg, np.full_like(angles2pol_deg, 90))
    )

    # Set parameter bounds
    bounds = [
        (0, upper_bounds[0]),  # euler1
        (0, upper_bounds[1]),  # euler2
        (0, upper_bounds[2]),  # euler3
    ]
    if thickness_bounds is not None:
        bounds.append(thickness_bounds)  # user-adjustable thickness range
    else:
        thickness_bounds = (0.9999, 1.0001)  # set thickness to (almost) 1
        bounds.append(thickness_bounds)

    # MINIMIZATION BASED ON L-BFGS-B algorithm (gradient-based optimization)
    if "gradient" in algorithm:
        start_time = time.time()

        # Initialize variables to track the best solution
        best_result = None
        best_error = float("inf")

        # generate initial guesses
        initial_guesses = get_initial_guesses(bounds, num_guesses)

        # run optimization
        for guess in initial_guesses:
            result_bfgs = minimize(
                fun=_misfit_function,
                x0=guess,
                args=(measurements, principal_Ts),
                bounds=bounds,
                method="L-BFGS-B",
                **kwargs,
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
            best_result, elapsed_time, alg_name="L-BFGS-B", num_guesses=num_guesses
        )

    # MINIMIZATION BASED ON differential evolution algorithm
    if "diffEvol" in algorithm:
        start_time = time.time()

        # run optimization
        result_diff = differential_evolution(
            func=_misfit_function,
            bounds=bounds,
            args=(measurements, principal_Ts),
            **kwargs,
        )

        end_time = time.time()
        results["differential_evol"] = result_diff
        elapsed_time = end_time - start_time

        summarize_results(result_diff, elapsed_time, alg_name="diffEvol")

    # MINIMIZATION BASED ON dual annealing algorithm
    if "annealing" in algorithm:
        start_time = time.time()

        # run optimization
        result_anne = dual_annealing(
            func=_misfit_function,
            bounds=bounds,
            args=(measurements, principal_Ts),
            **kwargs,
        )

        end_time = time.time()
        results["dual_annealing"] = result_anne
        elapsed_time = end_time - start_time

        summarize_results(result_anne, elapsed_time, alg_name="annealing")

    print("DONE")
    print(" ")

    return results


def bruteforce_algorithm(
    measurements,
    standard_Ts_1mm,
    step: int = 3
):
    """
    Brute force algorithm that finds the orientation that best
    fits within a defined accuracy.

    Warning: Use it only for testing purposes.
    
    TODO: need to 

    Parameters
    ----------
    measurements : _type_
        _description_
    standard_Ts_1mm : _type_
        tuple containing the standard transmittance values (1 mm)
        for a specific wavenumber along a-axis, b-axis, and
        c-axis. -> (Ta, Tb, Tc)
    step : int, optional
        _description_, by default 3

    Returns
    -------
    _type_
        _description_
    """

    euler = explore_Euler_space(step)
    diff = np.empty(euler.shape[0])

    for index, euler_ang in enumerate(euler):
        val = _misfit_function(euler_ang, measurements, standard_Ts_1mm)
        diff[index] = val

    print(f"Calculated Orientation: {euler[diff.argmin()]}")
    print(f"diff = {diff.min()}")

    return euler[diff.argmin()]


# ============================================================================ #
# AUXILIARY FUNCTIONS. Not intended for direct user access.                    #
# ============================================================================ #


def _misfit_function(
    params: np.ndarray,
    measurements: np.ndarray,
    principal_Ts: np.ndarray,
) -> float:
    """
    Computes misfit between measured and theoretical
    transmittance values for a given wavelength and orientation
    defined in Euler angles.

    Parameters
    ----------
    params : array-like
        Array of length 4 containing the euler angles (extrinsic,
        Bunge convention, degrees) and the sample thickness.
    measurements : array-like
        Measured transmitances and spherical angles respect
        to polarizer in the reference frame.
    principal_Ts : array-like
        Transmitance reference values along crystal directions
        a, b and c for the wavelength used.

    Returns
    -------
    float
        Sum of squared differences between measured and calculated
    """
    # extract parameters
    e1, e2, e3, d = params
    Ta, Tb, Tc = principal_Ts
    T_measured = measurements[:, 0]
    azimuths = np.deg2rad(measurements[:, 1])
    polar = np.deg2rad(measurements[:, 2])

    # convert from spherical to cartesian coordinates
    x, y, z = sph2cart(T_measured, azimuths, polar)

    # apply rotation to measures using Eules angles (Bunge convention)
    # Note that the order of euler angles are inverted and the sign changed
    x2, y2, z2 = rotate(coordinates=(x, y, z), euler_deg=(-e3, -e2, -e1))

    # convert back to spherical coordinates
    T_measured, azimuths, polar = cart2sph(x2.ravel(), y2.ravel(), z2.ravel())

    # estimate theoretical T values
    T_theoretical = calc_transmittance(
        standard=(Ta, Tb, Tc), theta_rad=azimuths, phi_rad=polar, d=d
    )

    # calculate the misfit
    return np.sum((T_measured - T_theoretical) ** 2)


# End of function definitions

# End of file
