# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: spectrum_tools.py                                                 #
# Description: Implements tools related to spectra                            #                                               #
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
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# Function definitions

def smooth_spectrum(
    data: np.ndarray,
    window_size: int = 9
) -> np.ndarray:
    """
    Apply a moving average smoothing to a 1D array.
    It uses Savitzky–Golay (preferred, more robust than
    simple moving average) or moving average fallback if
    window size is too small for Savitzky–Golay.

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

    # Apply Savitzky-Golay filter
    if len(data) > window_size and window_size > 2:
        poly_order = min(window_size - 2, 3)
        smoothed = savgol_filter(data, window_size, poly_order)
    else:
        # or fall back to simple moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode="same")

    return smoothed


def smooth_spectra(
    spectra: pd.DataFrame,
    window_size: int = 9
) -> pd.DataFrame:
    """
    This is a wrapper of the function smooth_spectrum
    to apply smoothing over tabular data. It is expected
    that the column containing the wavenumbers (to ignore)
    is named "wavenumbers.

    Parameters
    ----------
    spectra : pd.DataFrame
        a dataframe with spectra
    window_size : int, optional
        Number of samples to include in the moving average
        (must be odd), defaults to 9

    Returns
    -------
    pd.DataFrame
        Smoothed spectra of the same length as input.
    """
    
    # TODO: input checking!
    
    # smooth looping over columns (it's not the fastest, but easy to read).
    for col in spectra.columns:
        if col != "wavenumber":
            spectra[col] = smooth_spectrum(spectra[col], window_size)
    
    return spectra


def interpolate_spectra(
    target_wavenumbers: np.ndarray | pd.Series,
    spectra: pd.DataFrame,
    method: str = "pchip",
) -> pd.DataFrame:
    """
    Interpolate FTIR spectra at target wavenumbers.

    This function performs interpolation of one or more
    FTIR spectra from their original wavenumber grid to
    a new target wavenumber grid. By default, it uses a
    shape-preserving cubic Hermite interpolator (PCHIP),
    which avoids overshoot/undershoot around sharp spectral
    features and typically preserves peak shapes better than
    standard cubic splines.

    Parameters
    ----------
    target_wavenumbers : array-like of shape (M,)
        Wavenumbers (cm^-1) defining the desired target
        grid. Need not be uniformly spaced. Order can be
        increasing or decreasing.
    spectra : pd.DataFrame
        Pandas DataFrame with FTIR spectra to be interpolated.
        It must be contain a column with wavenumbers as "wavenumber",
        others represents different spectrum. Column names will be
        preserved.
    method : {"pchip", "linear"}, optional
        Interpolation method. "pchip" (default) is shape-preserving;
        "linear" uses first-order interpolation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with:
        - column 'wavenumbers' containing the target grid
          (increasing)
        - one column per spectrum containing the interpolated
          values (NaN outside original range).
    """

    # Sanity checks
    target_wavenumbers = np.asarray(target_wavenumbers, dtype=float).ravel()

    if False in np.isfinite(target_wavenumbers) or False in np.isfinite(spectra["wavenumber"]):
        raise ValueError("Wavenumbers must be finite (no NaN/inf).")
    
    method = method.lower()
    if method not in {"pchip", "linear"}:
        raise ValueError('Unsupported method. Use "pchip" or "linear"')
    
    # set the interpolation range to apply
    min_val = max(target_wavenumbers.min(), spectra["wavenumber"].min())
    max_val = min(target_wavenumbers.max(), spectra["wavenumber"].max())
    print(f"Spectra will be interpolated between {min_val:.1f} and {max_val:.1f} /cm")

    spectra = spectra.loc[spectra['wavenumber'].between(min_val, max_val)]
    mask = np.where((target_wavenumbers >= min_val) & (target_wavenumbers <= max_val))
    target_wavenumbers = target_wavenumbers[mask]

    # sort ascending for interpolation
    spectra = spectra.sort_values("wavenumber", kind="mergesort").reset_index(drop=True)
    target_wavenumbers.sort()

    # Initialize dataframe
    result = pd.DataFrame({"wavenumber": target_wavenumbers})

    # Nyquist-minded sampling diagnostics
    # TODO
    
    # interpolate looping over columns (it's not the fastest, but easy to read).
    for col in spectra.columns:
        if col != "wavenumber":
            if method == "pchip":
                f =  PchipInterpolator(spectra['wavenumber'], spectra[col], extrapolate=False)
                result[col] = f(target_wavenumbers)
            else:
                result[col] = np.interp(target_wavenumbers, spectra['wavenumber'], spectra[col])
    
    # TODO: check if nan in spectra and propagate last valid observation
    
    return result


def absorbance_to_transmittance(
    absorbance: np.ndarray,
    clip: bool = True
) -> np.ndarray:
    """
    Convert absorbance spectrum to transmittance (1d array).
    
    Parameters
    ----------
    absorbance : np.ndarray
        Absorbance values
    clip : bool, optional
        If True, clip values to physically meaningful ranges,
        by default True
        
    Returns
    -------
    np.ndarray
        Transmittance values (0 to 1)
        
    Notes
    -----
    Transmittance T = 10^(-A), where A is absorbance
    Physically meaningful ranges: A ∈ [0, ∞), T ∈ (0, 1]
    """
    absorbance = np.asarray(absorbance, dtype=float)
    transmittance = 10 ** (-absorbance)
    
    if clip:
        eps = np.finfo(float).eps
        transmittance = np.clip(transmittance, eps, 1.0)
    
    return transmittance


def abs_to_trans_batch(
    spectra: pd.DataFrame,
    clip: bool = True
) -> pd.DataFrame:
    """
    This is a wrapper of the function absorbance_to_transmittance
    It is expected that the column containing the wavenumbers
    (to ignore) is named "wavenumbers.

    Parameters
    ----------
    spectra : pd.DataFrame
        a dataframe with spectra
    clip : bool, optional
        If True, clip values to physically meaningful ranges,
        by default True

    Returns
    -------
    pd.DataFrame
        spectra in transmittance values (0 to 1)
    """

    t_spectra = pd.DataFrame(spectra["wavenumber"])
    
    # interpolate looping over columns (it's not the fastest, but easy to read).
    for col in spectra.columns:
        if col != "wavenumber":
            t_spectra[col] = absorbance_to_transmittance(
                absorbance=spectra[col],
                clip=clip
            )
    
    return t_spectra


def transmittance_to_absorbance(
    transmittance: np.ndarray,
    clip: bool = True
) -> np.ndarray:
    """
    Convert transmittance spectrum to absorbance (1d array).
    
    Parameters
    ----------
    transmittance : np.ndarray
        Transmittance values (0 to 1 or 0% to 100%)
    clip : bool, optional
        If True, clip values to physically meaningful ranges,
        by default True
        
    Returns
    -------
    np.ndarray
        Absorbance values
    """
    transmittance = np.asarray(transmittance, dtype=float)
    
    # prevent log(0) and ensures T ∈ (0, 1]
    if clip:
        eps = np.finfo(float).eps
        transmittance = np.clip(transmittance, eps, 1.0)
    
    return -np.log10(transmittance)


# End of function definitions

if __name__ == "__main__":
    pass
else:
    print("spectrum_tools module imported.")

# End of file
