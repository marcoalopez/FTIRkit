# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: lambda_orientation.py                                             #
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
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict, Any, List

from reference_frames import sph2cart, cart2sph

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
    equation from Asimov et al. (2006) .

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
        rgn = np.random.default_rng()  # create a random generator object
        initial_guesses = rgn.uniform(
            low=np.asarray(bounds)[:, 0],
            high=np.asarray(bounds)[:, 1],
            size=(num_guesses, 4),
        )

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
        params_bfgs = best_result.x
        elapsed_time = end_time - start_time

        # print/summarize results
        _summarize_results(result_bfgs, params_bfgs, elapsed_time, alg_name="L-BFGS-B", num_guesses=num_guesses)

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
        params_diff = result_diff.x
        elapsed_time = end_time - start_time

        _summarize_results(result_diff, params_diff, elapsed_time, alg_name="diff")

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
        params_anne = result_anne.x
        elapsed_time = end_time - start_time

        _summarize_results(result_anne, params_anne, elapsed_time, alg_name="dual")

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

    euler = _explore_Euler_space(step)
    diff = np.empty(euler.shape[0])

    for index, euler_ang in enumerate(euler):
        val = _misfit_function(euler_ang, measurements, params=standard_Ts_1mm)
        diff[index] = val

    print(f"Calculated Orientation: {euler[diff.argmin()]}")
    print(f"diff = {diff.min()}")

    return euler[diff.argmin()]


# ============================================================================ #
# AUXILIARY FUNCTIONS. Not intended for direct user access.                    #
# ============================================================================ #


def _rotate(
    coordinates: np.ndarray,
    euler_deg: np.ndarray,
    invert: bool = False
) -> np.ndarray:
    """
    Rotate points in 3D Cartesian space using Euler angles.

    Applies extrinsic rotations using the Bunge (zxz) convention.
    This function is a wrapper of R.from_euler() Scipy method for
    convenience.

    Parameters
    ----------
    coordinates : array_like
        Cartesian coordinates of the points to be rotated.
        Can be a single point (3,), multiple points (n, 3),
        or a grid of 3D points (n, n, 3).
    euler_deg : array_like of size 3
        Three Euler angles (ψ1, φ, ψ2) in degrees,
        following the Bunge zxz convention
    invert : bool, optional
        If True, inverts the rotation (applies the inverse rotation).
        Defaults to False.

    Returns
    -------
    numpy.ndarray
        Rotated coordinates. The output shape matches the input
        `coordinates` shape:
        - (3,) if input is a single point (3,).
        - (n, 3) if input is multiple points (n, 3).
        - (n, n, 3) if input is a grid of 3D points (n, n, 3).

    Example
    -------
    x, y, z = rotate(coordinates=(x, y, z), euler_ang=(30, 0, 40))
    """

    # Validate the input shape for coordinates
    coordinates = np.asarray(coordinates)
    # Single point (3,)
    if coordinates.ndim == 1 and coordinates.shape == (3,):
        coordinates = coordinates[np.newaxis, :]  # Reshape to (1, 3)
        original_shape = "single"
    elif coordinates[0].ndim == 1:  # Multiple points (n, 3)
        coordinates = np.vstack(coordinates).T
        original_shape = "multiple"
    elif coordinates[0].ndim == 2:  # Grid of points (n, n, 3)
        coordinates = np.dstack(coordinates)
        original_shape = "grid"
    else:
        raise ValueError(
            "Invalid shape for coordinates. Must be (3,), (n, 3), or (n, n, 3)."
        )

    # Validate euler_deg
    euler_deg = np.asarray(euler_deg)
    if euler_deg.shape != (3,):
        raise ValueError("Invalid shape for euler_deg. Must be (3,).")

    # Create rotation object
    rotation = R.from_euler("zxz", euler_deg, degrees=True)

    # Invert the rotation if specified
    if invert:
        rotation = rotation.inv()

    # Apply rotation
    rotated_coordinates = coordinates @ rotation.as_matrix().T

    # Reshape back to original shape if necessary
    if original_shape == "single":
        return rotated_coordinates.flatten()
    elif original_shape == "grid":
        return (
            rotated_coordinates[:, :, 0],
            rotated_coordinates[:, :, 1],
            rotated_coordinates[:, :, 2],
        )
    else:  # For "multiple"
        return (
            rotated_coordinates[:, 0],
            rotated_coordinates[:, 1],
            rotated_coordinates[:, 2],
        )


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
    x2, y2, z2 = _rotate(coordinates=(x, y, z), euler_deg=(-e3, -e2, -e1))

    # convert back to spherical coordinates
    T_measured, azimuths, polar = cart2sph(x2.ravel(), y2.ravel(), z2.ravel())

    # estimate theoretical T values
    T_theoretical = calc_transmittance(
        standard=(Ta, Tb, Tc), theta_rad=azimuths, phi_rad=polar, d=d
    )

    # calculate the misfit
    return np.sum((T_measured - T_theoretical) ** 2)


def _explore_Euler_space(
    step: int = 1,
    lower_bounds: Tuple = (0, 0, 0),
    upper_bounds: Tuple = (90, 90, 180)
) -> np.ndarray:
    """
    Returns a Numpy array with a uniform combination
    of Euler angles in degrees to explore the Euler space
    based on a defined step size.

    By default it assumes a orthorhombic symmetry mmm
    where the the fundamental zone are defined by:
    varphi1: 0-90
        Phi: 0-90
    varphi2: 0-180

    Parameters
    ----------
    step : int, optional
        the resolution, by default 1 degree
    lower_bounds : tuple, optional
        range of Euler angles, by default (0, 0, 0)
    upper_bounds : tuple, optional
        range of Euler angles, by default (90, 90, 180)

    Returns
    -------
    _type_
        _description_
    """
    lang1, lang2, lang3 = lower_bounds
    ang1, ang2, ang3 = upper_bounds

    phi1 = np.arange(lang1, ang1 + 1, step)
    Phi = np.arange(lang2, ang2 + 1, step)
    phi2 = np.arange(lang3, ang3 + 1, step)

    # Create a meshgrid of all possible combinations
    phi1, Phi, phi2 = np.meshgrid(phi1, Phi, phi2, indexing="ij")

    # Stack the angles along the third axis to create the final array
    array = np.stack((phi1, Phi, phi2), axis=-1)

    # Reshape the array to size 3*n
    array = array.reshape(-1, 3)

    return array


def _summarize_results(results, params, elapsed_time, alg_name, num_guesses=None) -> None:
    """
    Auxiliary function that prints and summarizes minimization results
    """

    print("****************************************************")
    print("RESULT:")
    print(f"Orientation (Euler angles in degrees): ({params[0]:.1f}, {params[1]:.1f}, {params[2]:.1f})")
    print(f"thickness: {params[3]:.3f} (with respect to the standard)")
    print(" ")
    print("MINIMIZATION FEATURES:")
    if alg_name == "L-BFGS-B":
        print(f"L-BFGS-B (gradient-based) algorithm with {num_guesses} initial guesses")
    elif alg_name == "diff":
        print("Differential evolution algorithm")
    elif alg_name == "dual":
        print("Dual annealing algorithm")
    print(f"message: {results.message}")
    print(f"success: {results.success}")
    print(f"misfit: {results.fun}")
    print(f"elapsed time: {elapsed_time:0.4f} seconds")
    print(" ")


# End of function definitions

if __name__ == "__main__":
    pass
else:
    print("lambda_orientation module imported.")

# End of file
