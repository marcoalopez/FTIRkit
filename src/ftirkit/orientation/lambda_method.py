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
    angles2pol_deg: np.ndarray,
    wavenumber: float | List[float] | np.ndarray = 1987.29,
) -> pd.DataFrame:
    """
    Extract one or more sections of the transmittance envelope
    from a set of spectra, one section per requested wavenumber.
    For each requested wavenumber the closest wavenumber available
    in the spectra is used, so exact values are not required. It
    returns a data frame with one transmittance column per
    wavenumber, named "T_<wavenumber>", plus a common column with
    the angle to the polariser in degrees. The data frame is ready
    to be used for section-based orientation analysis.

    Parameters
    ----------
    spectra : pd.DataFrame
        Transmittance spectra as a function of wavenumber. It must
        contain a "wavenumber" column plus one column per measured
        spectrum.
    angles2pol_deg : array-like
        The angle between the polarization direction and the
        specimen reference in degrees, one per spectrum (i.e. per
        non-wavenumber column of ``spectra``).
    wavenumber : float or array-like of float, optional
        Wavenumber(s) at which the section(s) are extracted,
        by default 1987.29.

    Returns
    -------
    pd.DataFrame
        One "T_<wavenumber>" column per requested wavenumber (named
        after the closest wavenumber actually found in the spectra)
        and a common "ang2pol_deg" column.
    """

    # input checking
    if "wavenumber" not in spectra.columns:
        raise KeyError("`spectra` must contain a 'wavenumber' column.")
    angles2pol_deg = np.asarray(angles2pol_deg)
    if spectra.shape[1] - 1 != angles2pol_deg.size:
        raise ValueError(
            "The number of angles must correspond to the number of spectra. Check your inputs!"
        )

    # find the closest available wavenumber to each requested value
    requested = np.atleast_1d(np.asarray(wavenumber, dtype=float))
    available = spectra["wavenumber"].to_numpy(dtype=float)
    nearest_indexes = np.abs(available[:, np.newaxis] - requested).argmin(axis=0)

    if np.unique(nearest_indexes).size != nearest_indexes.size:
        raise ValueError(
            "Two or more requested wavenumbers map to the same spectral "
            "point. Request more widely spaced wavenumbers."
        )

    # extract transmittance values (one row per selected wavenumber)
    t_values = spectra.drop(columns="wavenumber").to_numpy()[nearest_indexes]

    table = {
        f"T_{available[index]:.2f}": row
        for index, row in zip(nearest_indexes, t_values)
    }
    table["ang2pol_deg"] = angles2pol_deg

    return pd.DataFrame(table)


# ============================================================================ #
# MINIMIZATION PROCEDURE                                                       #
# ============================================================================ #


def find_orientation_based_on_lambda(
    transmittances: np.ndarray,
    angles2pol_deg: np.ndarray,
    standard_Ts_1mm: np.ndarray,
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
    transmittances : array-like
        Transmittance measurements at a specific wavelength.
    angles2pol_deg : array-like
        The angle between the polarization direction and the specimen
        reference in degrees at which the transmittance measurements
        were taken.
    standard_Ts_1mm : tuple of size 3
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
        (transmittances, angles2pol_deg, np.full_like(angles2pol_deg, 90))
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
                args=(measurements, standard_Ts_1mm),
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
            args=(measurements, standard_Ts_1mm),
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
            args=(measurements, standard_Ts_1mm),
            **kwargs,
        )

        end_time = time.time()
        results["dual_annealing"] = result_anne
        elapsed_time = end_time - start_time

        summarize_results(result_anne, elapsed_time, alg_name="annealing")

    print("DONE")
    print(" ")

    return results


def find_orientation_based_on_multiple_lambdas(
    transmittances: np.ndarray,
    angles2pol_deg: np.ndarray,
    standard_Ts_1mm: np.ndarray,
    upper_bounds: Tuple = (90., 89.99, 180.),
    thickness_bounds: Tuple[float, float] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Determine the crystallographic orientation from sets of polarised
    FTIR measurements taken at several wavelengths, fitted
    simultaneously with the differential evolution algorithm.

    The minimised quantity is the sum of the single-wavelength
    misfits over all wavelengths. This makes the estimate more
    robust than the single-wavelength method, as wavelengths that
    poorly constrain a particular orientation are compensated by
    the others.

    Parameters
    ----------
    transmittances : array-like of shape (n_wavelengths, n_measurements)
        Transmittance measurements, one row per wavelength.
    angles2pol_deg : array-like of shape (n_wavelengths, n_measurements)
        The angle between the polarization direction and the specimen
        reference in degrees at which each measurement in
        ``transmittances`` was taken.
    standard_Ts_1mm : array-like of shape (n_wavelengths, 3)
        The standard transmittance values (1 mm) along a-axis (Ta),
        b-axis (Tb), and c-axis (Tc), one (Ta, Tb, Tc) row per
        wavelength, in the same order as ``transmittances``.
    upper_bounds : tuple, optional
        the upper bounds of Euler angles defining the fundamental
        zone of solutions, by default (90., 89.99, 180.)
    thickness_bounds : Tuple of size 2 or None, optional
        Whether to fit thickness as well, defaults to None

    Returns
    -------
    dict
        a Python dict with the minimization output keyed by
        "differential_evol", plus the misfit of each wavelength at
        the best-fit parameters keyed by "per_wavelength_misfit"
        (array of shape (n_wavelengths,), same order as the inputs).
    """

    # Sanity checks
    transmittances, angles2pol_deg, standard_Ts_1mm = (
        _validate_find_orientation_based_on_multiple_lambdas(
            transmittances, angles2pol_deg, standard_Ts_1mm
        )
    )

    # organize all measurements in a Numpy array of shape (n, 3). The
    # misfit is summed over all points, so all wavelengths are fitted
    # simultaneously with a single orientation.
    measurements = np.column_stack(
        (
            transmittances.ravel(),
            angles2pol_deg.ravel(),
            np.full(transmittances.size, 90.0),
        )
    )

    # repeat each (Ta, Tb, Tc) row to match its measurements so the
    # single-wavelength misfit function can be reused unchanged
    num_measurements = transmittances.shape[1]
    Ta, Tb, Tc = np.repeat(standard_Ts_1mm, num_measurements, axis=0).T

    # Set parameter bounds
    bounds = [
        (0, upper_bounds[0]),  # euler1
        (0, upper_bounds[1]),  # euler2
        (0, upper_bounds[2]),  # euler3
    ]
    if thickness_bounds is None:
        thickness_bounds = (0.9999, 1.0001)  # set thickness to (almost) 1
    bounds.append(thickness_bounds)

    # MINIMIZATION BASED ON differential evolution algorithm
    start_time = time.time()

    result_diff = differential_evolution(
        func=_misfit_function,
        bounds=bounds,
        args=(measurements, (Ta, Tb, Tc)),
        **kwargs,
    )

    elapsed_time = time.time() - start_time
    results = {"differential_evol": result_diff}

    summarize_results(result_diff, elapsed_time, alg_name="diffEvol")

    # per-wavelength misfit breakdown (diagnostic): a wavelength with
    # a misfit far above the others suggests a systematic problem
    # with its data or standard values
    misfit_per_lambda = _per_wavelength_misfits(
        result_diff.x, transmittances, angles2pol_deg, standard_Ts_1mm
    )
    results["per_wavelength_misfit"] = misfit_per_lambda

    total_misfit = misfit_per_lambda.sum()
    print("PER-WAVELENGTH MISFIT BREAKDOWN:")
    for index, misfit in enumerate(misfit_per_lambda):
        share = 100.0 * misfit / total_misfit if total_misfit > 0 else 0.0
        print(f"wavelength {index + 1}: {misfit:.3e} ({share:.1f}% of total)")
    print(" ")

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


def _validate_find_orientation_based_on_multiple_lambdas(
    transmittances: np.ndarray,
    angles2pol_deg: np.ndarray,
    standard_Ts_1mm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate the inputs of `find_orientation_based_on_multiple_lambdas`.

    Returns
    -------
    tuple
        (transmittances, angles2pol_deg, standard_Ts_1mm) converted
        to 2D numpy float arrays.
    """

    transmittances = np.asarray(transmittances, dtype=float)
    angles2pol_deg = np.asarray(angles2pol_deg, dtype=float)
    standard_Ts_1mm = np.asarray(standard_Ts_1mm, dtype=float)

    if transmittances.ndim != 2:
        raise ValueError(
            "`transmittances` must be a 2D array of shape (n_wavelengths, "
            f"n_measurements); got shape {transmittances.shape}."
        )
    if angles2pol_deg.shape != transmittances.shape:
        raise ValueError(
            f"`angles2pol_deg` {angles2pol_deg.shape} and `transmittances` "
            f"{transmittances.shape} must have the same shape."
        )
    if standard_Ts_1mm.shape != (transmittances.shape[0], 3):
        raise ValueError(
            "`standard_Ts_1mm` must have shape (n_wavelengths, 3) = "
            f"({transmittances.shape[0]}, 3); got {standard_Ts_1mm.shape}."
        )

    return transmittances, angles2pol_deg, standard_Ts_1mm


def _per_wavelength_misfits(
    params: np.ndarray,
    transmittances: np.ndarray,
    angles2pol_deg: np.ndarray,
    standard_Ts_1mm: np.ndarray,
) -> np.ndarray:
    """
    Compute the misfit of each wavelength separately at the given
    parameters. The sum over wavelengths equals the combined misfit
    minimized by `find_orientation_based_on_multiple_lambdas`.

    Parameters
    ----------
    params : array-like
        Array of length 4 containing the euler angles (extrinsic,
        Bunge convention, degrees) and the sample thickness.
    transmittances : numpy.ndarray of shape (n_wavelengths, n_measurements)
        Transmittance measurements, one row per wavelength.
    angles2pol_deg : numpy.ndarray of shape (n_wavelengths, n_measurements)
        The angle to the polarizer in degrees for each measurement.
    standard_Ts_1mm : numpy.ndarray of shape (n_wavelengths, 3)
        The standard (Ta, Tb, Tc) values, one row per wavelength.

    Returns
    -------
    numpy.ndarray
        The misfit of each wavelength, shape (n_wavelengths,).
    """

    num_measurements = transmittances.shape[1]
    misfits = np.empty(transmittances.shape[0])

    for index, (T_row, angles_row, Ts) in enumerate(
        zip(transmittances, angles2pol_deg, standard_Ts_1mm)
    ):
        measurements = np.column_stack(
            (T_row, angles_row, np.full(num_measurements, 90.0))
        )
        misfits[index] = _misfit_function(params, measurements, Ts)

    return misfits


def _misfit_function(
    params: np.ndarray,
    measurements: np.ndarray,
    standard_Ts_1mm: np.ndarray,
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
        Measured transmittances and spherical angles respect
        to polarizer in the reference frame.
    standard_Ts_1mm : array-like
        Transmittance reference values along crystal directions
        a, b and c for the wavelength used.

    Returns
    -------
    float
        Sum of squared differences between measured and calculated
    """
    # extract parameters
    e1, e2, e3, d = params
    Ta, Tb, Tc = standard_Ts_1mm
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
        standard_Ts_1mm=(Ta, Tb, Tc), theta_rad=azimuths, phi_rad=polar, d=d
    )

    # calculate the misfit
    return np.sum((T_measured - T_theoretical) ** 2)


# End of function definitions

# End of file
