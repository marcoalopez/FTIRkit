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
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from ..transmittance_model import calc_transmittance
from ..crystallography import canonical_orthorhombic_Asimow_angles
from ..spectrum_tools import smooth_spectrum
from .common import get_initial_guesses, summarize_results

# Function definitions


def find_orientation_based_on_spectrum(
    standard: pd.DataFrame,
    spectrum: np.ndarray | pd.Series,
    algorithm: str | tuple[str, ...] = "all",
    thickness_bound: tuple[float, float] | None = None,
    smooth_window: int = 21,
    num_guesses: int = 15,
    transmittance_range: tuple[float, float] | None = None,
    exclude_wavenumbers: tuple[float, float] | list | None = None,
    canonical_angles: bool = True,
) -> dict[str, Any]:
    """
    Estimate the Asimow angles (θ, φ) (and optionally thickness d)
    by minimizing derivative misfit.

    The orientation is parametrized and reported as Asimow angles in
    radians. If you work with Euler orientations (Bunge convention),
    convert them separately with
    `ftirkit.crystallography.convert_Euler_to_Asimow_angles` for
    comparison with the fitted values.

    Parameters
    ----------
    standard : pd.DataFrame
        A pandas dataframe containing the normalized transmitance
        spectra for principal directions (Ta, Tb, Tc) as a function
        of wavelengths. Is assumes that column names are "Ta", "Tb",
        "Tc" and "wavenumber", respectively.
    spectrum : array-like or pd.Series
        Measured transmittance spectrum (values in (0, 1]) on the
        same wavenumber grid as `standard`.
    algorithm : str or tuple of str, optional
        the minimization algorithm(s) to use, by default "all".
        Can be: "gradient", "diffevol", "all", or a tuple/list with
        different algorithms, e.g. ("gradient", "diffevol").
    thickness_bound : Tuple of size 2 or None, optional
        (lower, upper) bounds of the thickness (relative to the
        standard) to fit it alongside the orientation, or None
        (default) to keep the thickness fixed to 1.
    smooth_window : int, optional
        Number of samples to include in the moving average
        (must be odd), defaults to 21
    num_guesses : int, optional
        Number of initial guesses when using the gradient-based
        algorithm, by default 15
    transmittance_range : Tuple of size 2 or None, optional
        If given, only spectral points whose *measured* transmittance
        lies within (min, max) contribute to the misfit. Useful to
        ignore saturated (T near 0) or noise-dominated regions,
        defaults to None (all points used).
    exclude_wavenumbers : Tuple of size 2, list of such tuples, or None, optional
        One (low, high) pair, or a list of pairs, defining wavenumber
        ranges to ignore in the misfit (e.g. atmospheric CO2/H2O bands
        or resin artefacts), defaults to None. This mirrors the channel
        exclusion ("squash") mechanism of the original Asimow C code.
    canonical_angles : bool, optional
        If True (default), fold the fitted angles into their canonical
        (first octant) representative under orthorhombic mmm symmetry,
        for which all symmetry equivalents are physically identical.
        Set to False to keep the raw optimizer output (e.g. for
        crystals of other symmetries).

    Returns
    -------
    dict
        The `scipy.optimize.OptimizeResult` of each algorithm, keyed
        by "gradient_based" and/or "differential_evol". In each result,
        x = [theta_rad, phi_rad, thickness] contains the best-fit
        parameters and `fun` the corresponding misfit.

    Notes
    -----
    Filtering removes points from the misfit sum only: smoothing and
    differentiation are still computed on the full wavenumber grid, so
    no artificial derivative steps appear at the edges of excluded
    ranges. Because of the smoothing window, values from excluded
    channels can still influence the derivative of nearby included
    channels; widen the excluded ranges by about half the smoothing
    window if the excluded data are strongly corrupted.
    """

    # Sanity checks
    spectrum, algorithm = _validate_find_orientation_based_on_spectrum(
        standard, spectrum, algorithm, thickness_bound, num_guesses
    )

    results = {}  # initialize dict

    # Set parameter bounds [theta, phi, thickness]
    if thickness_bound is not None:
        thickness = thickness_bound
    else:
        thickness = (1, 1)  # set thickness to 1

    theta_ang_rad = (0, 2 * np.pi)
    phi_ang_rad = (0, np.pi)
    bounds = [theta_ang_rad, phi_ang_rad, thickness]

    # Build the inclusion mask for the misfit (None = use all points)
    if transmittance_range is not None or exclude_wavenumbers is not None:
        include_mask = _build_inclusion_mask(
            wavenumbers=standard["wavenumber"].to_numpy(dtype=float),
            T_measured=spectrum,
            transmittance_range=transmittance_range,
            exclude_wavenumbers=exclude_wavenumbers,
            n_free_params=len(bounds),
        )
        print(f"Misfit restricted to {np.count_nonzero(include_mask)} of "
              f"{include_mask.size} spectral points.")
    else:
        include_mask = None

    # Precompute the measured derivative once: it is constant during
    # the optimization, unlike the synthetic one
    wavenumbers = standard["wavenumber"].to_numpy(dtype=float)
    spectrum_smooth = smooth_spectrum(spectrum, window_size=smooth_window)
    dT_measured = np.gradient(spectrum_smooth, wavenumbers)

    # Arguments shared by all minimization algorithms
    misfit_args = (
        wavenumbers,
        dT_measured,
        standard["Ta"].to_numpy(dtype=float),
        standard["Tb"].to_numpy(dtype=float),
        standard["Tc"].to_numpy(dtype=float),
        smooth_window,
        include_mask,
    )

    # MINIMIZATION BASED ON L-BFGS-B algorithm (gradient-based optimization)
    if "gradient" in algorithm:

        best_result = None
        best_error = float("inf")

        # compute initial guesses
        initial_guesses = get_initial_guesses(bounds, num_guesses)

        start_time = time.time()
        # run optimization
        for guess in initial_guesses:
            result_bfgs = minimize(
                fun=_error_function,
                x0=guess,
                args=misfit_args,
                bounds=bounds,
                method="L-BFGS-B",
            )

            # Update result if the current one is better
            if result_bfgs.fun < best_error:
                best_result = result_bfgs
                best_error = result_bfgs.fun

        end_time = time.time()
        if canonical_angles:
            _fold_result_to_canonical(best_result)
        results["gradient_based"] = best_result
        elapsed_time = end_time - start_time

        # print/summarize results
        summarize_results(
            best_result,
            elapsed_time,
            alg_name="L-BFGS-B",
            angle_format="asimow_rad",
            num_guesses=num_guesses,
        )

    # MINIMIZATION BASED ON differential evolution algorithm
    if "diffevol" in algorithm:
        start_time = time.time()

        # run optimization
        result_diff = differential_evolution(
            func=_error_function,
            bounds=bounds,
            args=misfit_args,
        )

        end_time = time.time()
        if canonical_angles:
            _fold_result_to_canonical(result_diff)
        results["differential_evol"] = result_diff
        elapsed_time = end_time - start_time

        summarize_results(
            result_diff, elapsed_time, alg_name="diffEvol", angle_format="asimow_rad"
        )

    print("DONE")
    print(" ")

    return results


# ============================================================================ #
# AUXILIARY FUNCTIONS. Not intended for direct user access.                    #
# ============================================================================ #


def _validate_find_orientation_based_on_spectrum(
    standard: pd.DataFrame,
    spectrum: np.ndarray | pd.Series,
    algorithm: str | tuple[str, ...],
    thickness_bound: tuple[float, float] | None,
    num_guesses: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Validate the inputs of `find_orientation_based_on_spectrum` and
    return the parsed spectrum and the normalized algorithm list.

    Returns
    -------
    tuple
        (spectrum, algorithms) where spectrum is a 1-D float array and
        algorithms is a list containing "gradient" and/or "diffevol".
    """

    required_cols = {"wavenumber", "Ta", "Tb", "Tc"}
    missing = required_cols.difference(standard.columns)
    if missing:
        raise KeyError(f"`standard` is missing required column(s): {sorted(missing)}.")

    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 1 or spectrum.size != len(standard):
        raise ValueError(
            f"`spectrum` must be 1-D with the same length as `standard` "
            f"({len(standard)}); got shape {spectrum.shape}."
        )
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("`spectrum` contains NaN or infinite values.")
    if spectrum.min() < 0.0 or spectrum.max() > 1.0:
        warnings.warn(
            "`spectrum` has values outside [0, 1]. It must be transmittance "
            "on the same wavenumber grid as `standard`, not absorbance."
        )

    if isinstance(algorithm, str):
        algorithms = [algorithm.lower()]
    else:
        algorithms = [alg.lower() for alg in algorithm]
    if "all" in algorithms:
        algorithms = ["gradient", "diffevol"]
    unknown = set(algorithms) - {"gradient", "diffevol"}
    if unknown:
        raise ValueError(
            f"Unsupported algorithm(s): {sorted(unknown)}. "
            'Use "all", "gradient" or "diffevol".'
        )

    if thickness_bound is not None:
        if len(thickness_bound) != 2 or not 0 < thickness_bound[0] <= thickness_bound[1]:
            raise ValueError(
                "`thickness_bound` must be a (lower, upper) pair with "
                "0 < lower <= upper."
            )

    if num_guesses < 1:
        raise ValueError("`num_guesses` must be a positive integer.")

    return spectrum, algorithms


def _build_inclusion_mask(
    wavenumbers: np.ndarray,
    T_measured: np.ndarray,
    transmittance_range: tuple[float, float] | None,
    exclude_wavenumbers: tuple[float, float] | list | None,
    n_free_params: int,
) -> np.ndarray:
    """
    Build the boolean mask of spectral points contributing to the misfit.

    Parameters
    ----------
    wavenumbers : numpy.ndarray
        Wavenumber values of the spectral points.
    T_measured : numpy.ndarray
        Measured transmittance spectrum (same length as wavenumbers).
    transmittance_range : Tuple of size 2 or None
        (min, max) range of measured transmittance to keep, or None.
    exclude_wavenumbers : Tuple of size 2, list of such tuples, or None
        One (low, high) wavenumber pair, or a list of pairs, to exclude,
        or None.
    n_free_params : int
        Number of fitted parameters; at least this many points plus one
        must remain after filtering.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where the point is used in the misfit.
    """

    if T_measured.shape != wavenumbers.shape:
        raise ValueError(
            f"`spectrum` (length {T_measured.size}) and the standard "
            f"wavenumber grid (length {wavenumbers.size}) must have the "
            "same length."
        )

    include_mask = np.ones(wavenumbers.size, dtype=bool)

    if transmittance_range is not None:
        t_min, t_max = transmittance_range
        if not t_min < t_max:
            raise ValueError(
                "`transmittance_range` must be an increasing (min, max) pair."
            )
        include_mask &= (T_measured >= t_min) & (T_measured <= t_max)

    if exclude_wavenumbers is not None:
        ranges = np.atleast_2d(np.asarray(exclude_wavenumbers, dtype=float))
        if ranges.ndim != 2 or ranges.shape[1] != 2:
            raise ValueError(
                "`exclude_wavenumbers` must be a (low, high) pair or a "
                "list of (low, high) pairs."
            )
        for low, high in ranges:
            if not low < high:
                raise ValueError(
                    f"Invalid exclusion range ({low}, {high}): the lower "
                    "bound must be smaller than the upper bound."
                )
            include_mask &= ~((wavenumbers >= low) & (wavenumbers <= high))

    if np.count_nonzero(include_mask) <= n_free_params:
        raise ValueError(
            f"Only {np.count_nonzero(include_mask)} spectral points remain "
            f"after filtering, which cannot constrain {n_free_params} "
            "parameters. Relax `transmittance_range` or `exclude_wavenumbers`."
        )

    return include_mask


def _error_function(
    params: np.ndarray,
    wavelengths: np.ndarray,
    dT_measured: np.ndarray,
    Ta_values: np.ndarray,
    Tb_values: np.ndarray,
    Tc_values: np.ndarray,
    smooth_window: int,
    include_mask: np.ndarray | None = None,
) -> float:
    """
    Compute misfit between measured and synthetic derivatives.

    Parameters
    ----------
    params : numpy.ndarray
        Parameters to optimize: [theta_rad, phi_rad, thickness]
    wavelengths : numpy.ndarray
        λ values for data points.
    dT_measured : numpy.ndarray
        Derivative of the smoothed measured transmittance spectrum
        (precomputed once by the caller, as it does not depend on
        the parameters).
    Ta_values : numpy.ndarray
        Array of Ta values corresponding to each wavelength
    Tb_values : numpy.ndarray
        Array of Tb values corresponding to each wavelength
    Tc_values : numpy.ndarray
        Array of Tc values corresponding to each wavelength
    smooth_window : int
        Window size for moving average smoothing(odd integer).
    include_mask : numpy.ndarray or None, optional
        Boolean array (same length as the spectra) selecting the
        points that contribute to the misfit; None (default) uses
        all points. Smoothing and differentiation are always
        computed on the full grid.

    Returns
    -------
    float
        Sum of squared differences between derivatives of measured
        and calculated spectra
    """
    # Extract parameters
    theta_rad, phi_rad, d = params

    # Calculate synthetic spectrum, smooth and derive
    T_calculated = calc_transmittance(
        standard=(Ta_values, Tb_values, Tc_values),
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        d=d,
    )
    T_calculated_smooth = smooth_spectrum(T_calculated, window_size=smooth_window)
    dT_calculated = np.gradient(T_calculated_smooth, wavelengths)

    # calculate the misfit (restricted to the included points, if any)
    squared_differences = (dT_measured - dT_calculated) ** 2
    if include_mask is not None:
        squared_differences = squared_differences[include_mask]

    return np.sum(squared_differences)


def _fold_result_to_canonical(result: Any) -> None:
    """
    Fold the fitted Asimow angles of an optimization result into their
    canonical (first octant) representative under orthorhombic mmm
    symmetry, in place. All symmetry equivalents give an identical
    transmittance model, so the misfit is unchanged.

    Parameters
    ----------
    result : scipy.optimize.OptimizeResult
        Optimization result with x = [theta_rad, phi_rad, thickness].
    """

    theta_deg = np.degrees(result.x[0]) % 360.0
    phi_deg = np.clip(np.degrees(result.x[1]), 0.0, 180.0)
    # canonical_orthorhombic_Asimow_angles returns [phi, theta]
    phi_canon, theta_canon = canonical_orthorhombic_Asimow_angles(
        theta_deg, phi_deg, show=False
    )
    result.x[0] = np.deg2rad(theta_canon)
    result.x[1] = np.deg2rad(phi_canon)


# End of function definitions

# End of file
