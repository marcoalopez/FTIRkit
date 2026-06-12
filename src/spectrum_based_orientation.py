# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: spectrum_based_orientation.py                                     #
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
from crystallography import convert_Euler_to_Asimow_angles
from spectrum_tools import smooth_spectrum

# Function definitions


def calc_transmittance(
    standard: np.ndarray,
    theta_rad: float,
    phi_rad: float,
    d: float = 1.0,
) -> float:
    """
    Compute transmittance for a given wavelength (λ),
    orientation (φ,θ), and sample thickness (d) from a
    standard (T spectra along principal axis) using the
    equation from Asimow et al. (2006) .

    T(λ,φ,θ,d) = Ta^d cos^2θ sin^2φ + Tb^d sin^2θ sin^2φ + Tc^d cos^2φ.

    Parameters
    ----------
    standard : array-like
        shape (3, N) containing transmittance spectra along the
        principal crystal axes (Ta, Tb, Tc).
    theta_rad : float
        Angle θ (a-axis to the vector E') in radians [0, 2π].
        ranges from 0 to 2*pi
    phi_rad : float
        Angle φ (c-axis to the polarization vector E) in radians [0, π].
    d : float, optional
        the sample thickness exponent, defaults to 1.0

    Returns
    -------
    np.ndarray
        calculated T spectrum (length N).
    """

    # Extract T values
    Ta, Tb, Tc = standard

    term_a = Ta**d * np.cos(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    term_b = Tb**d * np.sin(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    term_c = Tc**d * np.cos(phi_rad) ** 2

    return np.array(term_a + term_b + term_c)


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
    elif angles == "Euler":  # the bounds below assumes an orthorhombic symmetry
        euler1_rad = (0, np.pi / 2)
        euler2_rad = (0, np.pi / 2)
        euler3_rad = (0, np.pi)
        bounds = [euler1_rad, euler2_rad, euler3_rad, thickness]
    else:
        raise ValueError('Unsupported orientation method. Use "Asimow" or "Euler"')

    # MINIMIZATION BASED ON L-BFGS-B algorithm (gradient-based optimization)
    if "gradient" in algorithm:

        best_result = None
        best_error = float("inf")

        # compute initial guesses
        initial_guesses = _get_initial_guesses(bounds, num_guesses)

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
        params_bfgs = best_result.x
        elapsed_time = end_time - start_time

        # print/summarize results
        _summarize_results(result_bfgs, params_bfgs, elapsed_time, num_guesses)

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
        params_diff = result_diff.x
        elapsed_time = end_time - start_time

        _summarize_results(result_diff, params_diff, elapsed_time)

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


def _get_initial_guesses(bounds, num_guesses):
    """
    Auxiliary function that draws samples for a uniform distribution
    and returns the desired number of samples (rows) with the different
    parameters ordered in columns. It automatically handles whether
    the input is in Asimow or Euler angles.

    E.g. For
    bounds = [(0, 2*pi), (0, pi), (0.9, 1.1)] &
    num_guesses = 5
    it returns something like:
    array([[6.07130383, 1.20273549, 0.99748096],
           [5.30940826, 2.33540417, 1.03882422],
           [5.59317016, 2.33982344, 1.01041884],
           [3.60262451, 1.99625608, 0.92224606],
           [2.00669947, 1.34270517, 0.97783142]])
    """
    # create a random generator object
    rgn = np.random.default_rng()

    # set whether output is Asimow or Euler angles
    if len(bounds) == 3:
        columns = 3
    else:
        columns = 4

    return rgn.uniform(
        low=np.asarray(bounds)[:, 0],
        high=np.asarray(bounds)[:, 1],
        size=(num_guesses, columns),
    )


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


def _summarize_results(results, params, elapsed_time, num_guesses=None):
    """
    Auxiliary function that prints and summarizes minimization results
    """

    print("****************************************************")
    print("RESULT:")
    if len(params) == 3:
        print(f"theta: {np.degrees(params[0]):.1f}° ({params[0]:.3f} rad)")
        print(f"phi: {np.degrees(params[1]):.1f}° ({params[1]:.3f} rad)")
    else:
        print(
            f"Euler angles (Bunge) ({np.degrees(params[0]):.1f}, {np.degrees(params[1]):.1f}, {np.degrees(params[2]):.1f}) degrees"
        )
    print(f"thickness: {params[-1]:.3f} mm")
    print(" ")
    print("MINIMIZATION FEATURES:")
    if num_guesses is not None:
        print(f"L-BFGS-B (gradient-based) algorithm with {num_guesses} initial guesses")
    else:
        print("Differential evolution algorithm")
    print(f"message: {results.message}")
    print(f"success: {results.success}")
    print(f"misfit: {results.fun}")
    print(f"elapsed time: {elapsed_time:0.4f} seconds")
    print(" ")


# End of function definitions

if __name__ == "__main__":
    pass
else:
    print("spectrum_based_orientation module imported.")

# End of file
