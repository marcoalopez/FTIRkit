# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: orientation/common.py                                             #
# Description: Shared scaffolding for the orientation estimation methods      #
# (initial guesses, Euler space exploration, result reporting).               #
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
from typing import Tuple

# Function definitions

def get_initial_guesses(bounds, num_guesses):
    """
    Draws samples from a uniform distribution and returns the
    desired number of samples (rows) with the different
    parameters ordered in columns. One column is generated per
    entry in `bounds`, so it automatically handles whether the
    input is in Asimow or Euler angles (with or without thickness).

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

    bounds = np.asarray(bounds)

    return rgn.uniform(
        low=bounds[:, 0],
        high=bounds[:, 1],
        size=(num_guesses, bounds.shape[0]),
    )


def explore_Euler_space(
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


def summarize_results(
    result,
    elapsed_time: float,
    alg_name: str,
    angle_format: str = "euler_deg",
    num_guesses: int | None = None,
) -> None:
    """
    Auxiliary function that prints and summarizes minimization results.

    Parameters
    ----------
    result : scipy.optimize.OptimizeResult
        The minimization result to report. The fitted parameters
        (result.x) must contain the orientation angles followed by
        the thickness as the last entry.
    elapsed_time : float
        Elapsed wall-clock time of the minimization in seconds.
    alg_name : str
        The algorithm used: "L-BFGS-B", "diffEvol" or "annealing".
    angle_format : str, optional
        Format of the orientation parameters in result.x:
        - "euler_deg": Euler angles (Bunge) in degrees (default)
        - "euler_rad": Euler angles (Bunge) in radians
        - "asimow_rad": Asimow angles (theta, phi) in radians
    num_guesses : int or None, optional
        Number of initial guesses (reported for the gradient-based
        algorithm), by default None
    """

    params = result.x

    print("****************************************************")
    print("RESULT:")
    if angle_format == "euler_deg":
        print(f"Orientation (Euler angles in degrees): ({params[0]:.1f}, {params[1]:.1f}, {params[2]:.1f})")
    elif angle_format == "euler_rad":
        print(
            f"Euler angles (Bunge) ({np.degrees(params[0]):.1f}, {np.degrees(params[1]):.1f}, {np.degrees(params[2]):.1f}) degrees"
        )
    elif angle_format == "asimow_rad":
        print(f"theta: {np.degrees(params[0]):.1f}° ({params[0]:.3f} rad)")
        print(f"phi: {np.degrees(params[1]):.1f}° ({params[1]:.3f} rad)")
    else:
        raise ValueError(
            'angle_format must be "euler_deg", "euler_rad" or "asimow_rad"'
        )
    print(f"thickness: {params[-1]:.3f} (with respect to the standard)")
    print(" ")
    print("MINIMIZATION FEATURES:")
    if alg_name == "L-BFGS-B":
        print(f"L-BFGS-B (gradient-based) algorithm with {num_guesses} initial guesses")
    elif alg_name == "diffEvol":
        print("Differential evolution algorithm")
    elif alg_name == "annealing":
        print("Dual annealing algorithm")
    print(f"message: {result.message}")
    print(f"success: {result.success}")
    print(f"misfit: {result.fun}")
    print(f"elapsed time: {elapsed_time:0.4f} seconds")
    print(" ")


# End of function definitions

# End of file
