################################################################################
# This file is part of FTIRkit repository                                      #
#                                                                              #
# Filename: principal_axis.py                                                  #
# Description: Implements synthesis of principal axis spectra                  #
#                                                                              #
# SPDX-License-Identifier: Apache-2.0                                          #
# Copyright (c) 2025 Marco A. Lopez-Sanchez. All rights reserved.              #
#                                                                              #
# Licensed under the Apache License, Version 2.0 (the License);                #
# you may not use this file except in compliance with the License.             #
# You may obtain a copy of the License at                                      #
#                                                                              #
#     http://www.apache.org/licenses/LICENSE-2.0                               #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the License is distributed on an AS IS BASIS,              #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
# See the License for the specific language governing permissions and          #
# limitations under the License.                                               #
#                                                                              #
# Version: 2025.06.04-beta                                                     #
# Repository: https://github.com/marcoalopez/FTIR_CrystalOrient                #
################################################################################


# Import statements
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
# from scipy.optimize import nnls
from scipy.signal import savgol_filter

# Function definitions

def calc_transmittance(
    principal_Ts: np.ndarray,
    theta_rad: float,
    phi_rad: float,
    d: float = 1.0,
) -> float | np.ndarray:
    """
    Compute transmittance using the equation from Asimov et al.
    (2006) for a given wavelength (λ), orientation (φ,θ), and
    sample thickness (d).

    T(λ,φ,θ,d) = Ta^d cos^2θ sin^2φ + Tb^d sin^2θ sin^2φ + Tc^d cos^2φ.

    Parameters
    ----------
    principal_Ts : array-like
        shape (3, N) containing Ta, Tb, Tc spectra along the crystal axes.
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
    Ta, Tb, Tc = principal_Ts

    term_a = Ta**d * np.cos(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    term_b = Tb**d * np.sin(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    term_c = Tc**d * np.cos(phi_rad) ** 2

    return np.array(term_a + term_b + term_c)


def estimate_principal_axis_transmittance(
    T_obs: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    evaluate_model=False
):
    """
    Estimate (Ta, Tb, Tc) from measured transmittances
    via least squares.

    Parameters
    ----------
    T_obs : np.ndarray
        Observed transmittance values between 0 and 1.
    theta_deg : np.ndarray
        a-axis to polarization vector projection (E') angles
        in degrees
    phi_deg : np.ndarray
        c-axis to polarization vector (e) angles in degrees
    evaluate_model : bool, optional
        evaluate the model fit if requested, defaults to False
    """

    if not (T_obs.shape == theta_deg.shape == phi_deg.shape):
        raise ValueError(
            "Input arrays must have the same shape."
            f"T_obs shape: {T_obs.shape}; Theta shape: {theta_deg.shape}; Phi shape: {phi_deg.shape}; "
        )

    # convert to radians
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # build coefficient matrix A and solve A x = T_obs
    A = build_coefficient_matrix(theta, phi)

    # solve Ax = b for x = [Ta, Tb, Tc]
    # principal_Ts, *_ = lstsq(A, T_obs, overwrite_a=True, overwrite_b=True)
    principal_Ts, *_ = lstsq(A, T_obs)
    # principal_Ts, *_ = nnls(A, T_obs)

    # Evaluate model fit if requested
    if evaluate_model:
        r_squared, rmse = evaluate_model_fit(principal_Ts, T_obs, theta, phi)
        return np.hstack([principal_Ts, r_squared, rmse])
    else:
        return principal_Ts
 

def apply_principal_spectra_to_dataframe(
    Abs_spectra: pd.DataFrame,
    theta_deg: np.ndarray | pd.Series,
    phi_deg: np.ndarray | pd.Series,
    thickness_mm: np.ndarray | pd.Series,
    evaluate_model: bool = False,
) -> pd.DataFrame:
    """
    Wrapper to call the lstsq solver over the rows of a DataFrame.

    Parameters
    ----------
    Abs_spectra : pd.DataFrame
        Absorbance spectra, columns (wavenumbers + n_spectra).
    theta_deg : np.ndarray | pd.Series
        Angle between a-axis and polarization vector (degrees),
        shape (n_measurements,).
    phi_deg : np.ndarray | pd.Series
        Angle between c-axis and polarization vector (degrees),
        shape (n_measurements,).
    thickness_mm : np.ndarray | pd.Series
        Sample thickness in millimeters, shape (n_measurements,)
    evaluate_model : bool, optional
        If True, returns model fit metrics (R² and RMSE). Defaults to False.

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # estimate scaling factors
    raw_scaling_factor = 100.
    scaling_factor = raw_scaling_factor / np.asarray(thickness_mm)

    # prepare data
    wavenumber = Abs_spectra["wavenumber"].copy()
    A_spectra = Abs_spectra.drop(columns="wavenumber").copy()

    # set tiny absorbance (A) values to zero
    A_spectra[A_spectra < 1e-8] = 0

    # Calc transmittance while renormalize A_spectra for lstsq
    T_spectra = 10 ** -(A_spectra / scaling_factor)

    # apply lstsq solver over rows
    out = T_spectra.apply(
        estimate_principal_axis_transmittance,
        axis=1,
        args=(theta_deg, phi_deg, evaluate_model),
        result_type="expand",
    )

    if evaluate_model:
        out = out.rename(columns={0: "Ta", 1: "Tb", 2: "Tc", 3: "R_squared", 4: "RMSE"})
    else:
        out = out.rename(columns={0: "Ta", 1: "Tb", 2: "Tc"})
    
    # convert to absorbance (A = -log10(T)) and normalize to 1 cm (10 mm)
    out["Aa"] = np.around(-10 * np.log10(out["Ta"]), 5)
    out["Ab"] = np.around(-10 * np.log10(out["Tb"]), 5)
    out["Ac"] = np.around(-10 * np.log10(out["Tc"]), 5)

    # renormalize Absorbance values
    out[["Aa", "Ab", "Ac"]] *= raw_scaling_factor

    # insert the wavenumber and remove Ts
    out.insert(0, "wavenumber", wavenumber)
    out = out.drop(columns=["Ta", "Tb", "Tc"])

    return out


def smooth_spectrum(
    data: np.ndarray,
    window_size: int = 9
)-> np.ndarray:
    """
    Apply a moving average smoothing to a 1D array.

    Parameters
    ----------
    data : array-like
        Input spectrum values.
    window_size : int, optional
        Number of samples to include in the moving average
        (must be odd), defaults to 9

    Returns
    -------
    numpy.ndarray
        Smoothed spectrum of the same length as input.
    """

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    # Apply Savitzky-Golay filter for smoothing
    # (more robust than simple moving average)
    # Fall back to simple moving average if window
    # size is too small for savgol
    if len(data) > window_size and window_size > 2:
        poly_order = min(
            window_size - 2, 3
        )  # Polynomial order must be less than window_size
        smoothed = savgol_filter(data, window_size, poly_order)
    else:
        # Simple moving average as fallback
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode="same")

    return smoothed


# ============================================================================ #
# AUXILIARY FUNCTIONS                                                          #
# ============================================================================ #

def build_coefficient_matrix(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray
) -> np.ndarray:
    """
    Computes the coefficient matrix A for the linear
    system Ax = b.

    The columns correspond to the trigonometric terms
    multiplying Ta, Tb, and Tc.

    Parameters
    ----------
    theta_rad : numpy.ndarray
        Array of a-axis to polarizer angles (theta)
        in radians.
    phi_rad : numpy.ndarray
         Array of c-axis to polarizer angles (phi)
         in radians.

    Returns
    -------
    numpy.ndarray
        The design matrix A with shape (n_measurements, 3).
    """

    # Construct the columns of the design matrix A
    col_a = np.cos(theta_rad) ** 2 * np.sin(phi_rad) ** 2  # Ta coefficient
    col_b = np.sin(theta_rad) ** 2 * np.sin(phi_rad) ** 2  # Tb coefficient
    col_c = np.cos(phi_rad) ** 2                           # Tc coefficient

    # Stack columns horizontally
    A = np.column_stack([col_a, col_b, col_c])

    return A


def evaluate_model_fit(
    principal_Ts,
    T_measured,
    theta_rad,
    phi_rad
):
    """
    Evaluate how well the least-squares model fits.

    Parameters
    ----------
    principal_Ts : numpy.ndarray
        fitted principal axis transmittance values
    T_measured : numpy.ndarray
        the transmision values measured for a especific wavelength
    theta_rad : numpy.ndarray
        Array of a-axis to polarizer angles (theta) in radians.
    phi_rad : numpy.ndarray
         Array of c-axis to polarizer angles (phi) in radians.

    Returns
    -------
    r_squared : float
        R-squared value (coefficient of determination)
    rmse : float
        Root Mean Square Error
    """

    # Calculate predicted T values
    T_predicted = calc_transmittance(principal_Ts, theta_rad, phi_rad)

    # Calculate R-squared
    ss_total = np.sum((T_measured - np.mean(T_measured)) ** 2)
    ss_residual = np.sum((T_measured - T_predicted) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

    # Calculate RMSE
    rmse = np.sqrt(np.mean((T_measured - T_predicted) ** 2))

    return r_squared, rmse


# End of function definitions

if __name__ == "__main__":
    pass
else:
    print("principal_axis module (old) imported.")

# End of file