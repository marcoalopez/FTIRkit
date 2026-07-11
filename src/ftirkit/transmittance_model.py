# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: transmittance_model.py                                            #
# Description: Implements the transmittance model of Asimow et al. (2006),    #
# the core physical model shared by all FTIRkit modules.                      #
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

# Function definitions

def calc_transmittance(
    standard_Ts_1mm: np.ndarray,
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
    standard_Ts_1mm : array-like
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
    Ta, Tb, Tc = standard_Ts_1mm

    term_a = Ta**d * np.cos(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    term_b = Tb**d * np.sin(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    term_c = Tc**d * np.cos(phi_rad) ** 2

    return np.array(term_a + term_b + term_c)

# End of function definitions

# End of file
