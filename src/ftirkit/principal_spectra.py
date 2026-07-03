# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: principal_spectra.py                                             #
# Description: Synthesis of principal-axis spectra from sets of polarized    #
# FTIR measurements following the mixing model of Asimow et al. (2006),      #
# with physically correct handling of thickness variation and a numerically  #
# robust back-transformation from transmittance to absorbance.               #
#                                                                             #
# SPDX-License-Identifier: Apache-2.0                                         #
# Copyright (c) 2026 Marco A. Lopez-Sanchez. All rights reserved.             #
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
"""
Synthesis of principal-axis FTIR spectra from polarized measurements.

Physical model
--------------
At each wavenumber, a measurement i with orientation (θ_i, φ_i) and
thickness d_i is modelled as a geometric mixture of independently
attenuated principal-axis components (Asimow et al., 2006):

    T_i = w_ia 10^(-α_a d_i) + w_ib 10^(-α_b d_i) + w_ic 10^(-α_c d_i)

with weights w_ia = cos²θ sin²φ, w_ib = sin²θ sin²φ, w_ic = cos²φ
(w_ia + w_ib + w_ic = 1) and α_j the principal absorption coefficients
per unit thickness. Only the principal absorbances obey Beer–Lambert
exactly, so mixed (non-principal) spectra cannot be rescaled to a common
thickness; instead, the α_j are fitted directly with each measurement at
its own thickness. When all thicknesses are equal (within a tolerance),
the model is exactly linear in the principal transmittances at that
thickness and a fast bounded linear least-squares path is used.

Numerical robustness
--------------------
The back-transformation A = -log10(T) amplifies noise without bound as
T → 0: where the true principal transmittance falls below the
measurement noise the data only bound the absorbance from below. This
is an information limit of the experiment, not a defect of the solver.
The module therefore (i) constrains solutions to the physical domain
(α ≥ 0, i.e. T ≤ 1, plus a hard transmittance floor), (ii) propagates
per-wavenumber uncertainties to the reported absorbances, and
(iii) flags axis/wavenumber values that are censored (only a lower
bound on absorbance is supported by the data) so they can be masked.

Reference
---------
Asimow, P.D., Stein, L.C., Mosenfelder, J.L., Rossman, G.R. (2006)
Quantitative polarized infrared analysis of trace OH in populations of
randomly oriented mineral grains. American Mineralogist 91, 278-284.
"""

# Import statements
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

# Module-level constants
LN10 = np.log(10.0)
_CONDITION_WARN_THRESHOLD = 1.0e3


# Function definitions

def synthesize_principal_spectra(
    spectra: pd.DataFrame,
    theta_deg: np.ndarray | pd.Series,
    phi_deg: np.ndarray | pd.Series,
    thickness_mm: np.ndarray | pd.Series,
    reference_thickness_mm: float = 10.0,
    method: str = "auto",
    thickness_rtol: float = 0.005,
    transmittance_floor: float = 1e-6,
    noise_std_T: float | None = None,
    censor_factor: float = 2.0,
    robust: bool = False,
    loss_scale: float = 1e-2,
) -> pd.DataFrame:
    """
    Synthesize principal-axis absorbance spectra (Aa, Ab, Ac) from a set
    of polarized FTIR absorbance spectra with known orientations.

    The solver works in transmittance, where the Asimow et al. (2006)
    mixing model is valid, and fits the principal absorption
    coefficients per mm at each wavenumber. Each measurement enters the
    model at its own thickness, so no thickness pre-normalization of
    mixed spectra is performed. Results are reported as absorbance at
    `reference_thickness_mm`, together with propagated 1-sigma
    uncertainties and censoring flags marking wavenumbers where the
    data only support a lower bound on the absorbance.

    Parameters
    ----------
    spectra : pd.DataFrame
        Absorbance spectra with a "wavenumber" column plus one column
        per measurement (at that measurement's thickness). Values must
        be finite; small negative absorbances (noise) are clipped to 0.
    theta_deg : np.ndarray | pd.Series
        Angle between the a-axis and the polarizer projection E' in
        degrees, shape (n_measurements,).
    phi_deg : np.ndarray | pd.Series
        Angle between the c-axis and the polarization vector E in
        degrees, shape (n_measurements,).
    thickness_mm : np.ndarray | pd.Series
        Sample thickness in millimeters, shape (n_measurements,).
    reference_thickness_mm : float, optional
        Thickness at which the principal absorbances are reported,
        defaults to 10.0 (1 cm, the usual basis of water calibrations).
    method : {"auto", "linear", "nonlinear"}, optional
        "linear" solves the bounded linear system in transmittance and
        is exact only for equal thicknesses (all within
        `thickness_rtol`); "nonlinear" fits the absorption coefficients
        with each spectrum at its own thickness; "auto" (default) picks
        "linear" when the relative thickness spread is within
        `thickness_rtol` and "nonlinear" otherwise.
    thickness_rtol : float, optional
        Maximum relative thickness spread, (max - min) / mean, for
        which the linear path is accepted, defaults to 0.005 (0.5%).
    transmittance_floor : float, optional
        Hard lower bound on the principal transmittance of the thinnest
        measurement, defaults to 1e-6. This caps the fitted absorption
        coefficients at -log10(floor) / min(thickness) and is a
        numerical guard, not a reliability limit (see censoring flags).
    noise_std_T : float | None, optional
        Standard deviation of the measurement noise in transmittance
        units. If None (default) it is estimated robustly from the fit
        residuals; with exactly 3 spectra the system is exactly
        determined and the noise cannot be estimated, so censoring
        flags are disabled unless a value is given.
    censor_factor : float, optional
        Multiple of the noise level below which an axis is flagged as
        censored: axis j is censored when its largest contribution to
        any measured transmittance, w_ij 10^(-alpha_j d_i), falls below
        `censor_factor * noise_std_T`, i.e. the axis signal is buried
        in the noise of every measurement. Defaults to 2.0.
    robust : bool, optional
        If True, use a soft-L1 loss in the nonlinear fit to reduce the
        influence of outlier spectra (cracks, inclusions), defaults to
        False (plain least squares, the maximum-likelihood choice for
        Gaussian noise in transmittance).
    loss_scale : float, optional
        Residual scale (transmittance units) at which the soft-L1 loss
        transitions from quadratic to linear; only used when
        `robust=True`. Defaults to 1e-2.

    Returns
    -------
    pd.DataFrame
        Columns: "wavenumber"; "Aa", "Ab", "Ac" (absorbance at the
        reference thickness); "Aa_sigma", "Ab_sigma", "Ac_sigma"
        (propagated 1-sigma uncertainties, NaN when the noise level is
        unknown, inf where the data carry no sensitivity);
        "Aa_censored", "Ab_censored", "Ac_censored" (True where the
        value is only a lower bound and should be masked in
        quantitative use); "rms_residual" (per-wavenumber RMS misfit in
        transmittance). Fit diagnostics are stored in the DataFrame
        `attrs` dictionary: "method_used", "condition_number",
        "noise_std_T", "alpha_max_per_mm", "thickness_spread" and
        "reference_thickness_mm".

    Notes
    -----
    Principal transmittances at the reference thickness can be
    recovered as T = 10**(-A). Uncertainties are derived from a local
    linearization and are not meaningful at censored wavenumbers.
    """

    A_meas, theta_rad, phi_rad, d_mm = _validate_synthesize_principal_spectra(
        spectra,
        theta_deg,
        phi_deg,
        thickness_mm,
        method,
        thickness_rtol,
        transmittance_floor,
        noise_std_T,
        censor_factor,
        reference_thickness_mm,
    )
    wavenumber = spectra["wavenumber"].to_numpy(dtype=float)
    n_meas = d_mm.size

    # geometric weight (design) matrix and non-coplanarity check
    weights = build_coefficient_matrix(theta_rad, phi_rad)
    condition_number = np.linalg.cond(weights)
    if condition_number > _CONDITION_WARN_THRESHOLD:
        warnings.warn(
            f"The orientation set is nearly coplanar (condition number "
            f"{condition_number:.3g}). The principal spectra will be "
            "poorly constrained; add measurements with more varied "
            "orientations if possible."
        )

    # clip small negative absorbances (noise) and convert to transmittance
    n_negative = np.count_nonzero(A_meas < 0)
    if n_negative > 0:
        warnings.warn(
            f"{n_negative} negative absorbance values were clipped to zero."
        )
    T_obs = 10.0 ** (-np.clip(A_meas, 0.0, None))

    # hard cap on the absorption coefficients: even the thinnest sample
    # cannot constrain transmittances below `transmittance_floor`
    d_min = d_mm.min()
    d_mean = d_mm.mean()
    alpha_max = -np.log10(transmittance_floor) / d_min

    # choose the solution path
    thickness_spread = np.ptp(d_mm) / d_mean
    if method == "auto":
        method_used = "linear" if thickness_spread <= thickness_rtol else "nonlinear"
    elif method == "linear":
        if thickness_spread > thickness_rtol:
            raise ValueError(
                f"method='linear' requires a relative thickness spread <= "
                f"{thickness_rtol} but the data spread is "
                f"{thickness_spread:.4f}. The linear transmittance mixing "
                "model is only valid at a common thickness; use "
                "method='nonlinear' (or 'auto') instead."
            )
        method_used = "linear"
    else:
        method_used = "nonlinear"

    # solve for the principal absorption coefficients (per mm)
    if method_used == "linear":
        t_lower = 10.0 ** (-alpha_max * d_mean)
        T_principal = _solve_linear_transmittance(weights, T_obs, t_lower)
        alpha = -np.log10(T_principal) / d_mean
    else:
        alpha_guess = _linearized_alpha_guess(weights, d_mm, A_meas, alpha_max)
        alpha = _solve_nonlinear_alpha(
            weights, T_obs, d_mm, alpha_guess, alpha_max, robust, loss_scale
        )

    # residual diagnostics (per-axis transmittance at each measurement)
    T_axes = 10.0 ** (-d_mm[None, :, None] * alpha[:, None, :])
    T_model = np.sum(weights[None, :, :] * T_axes, axis=2)
    residuals = T_model - T_obs
    rms_residual = np.sqrt(np.mean(residuals**2, axis=1))

    # noise level and per-wavenumber residual scale
    noise_std, sigma_wn = _estimate_noise_level(
        residuals, rms_residual, n_meas, noise_std_T
    )

    # 1-sigma uncertainty of alpha from the local Gauss-Newton covariance
    jacobian = -LN10 * d_mm[None, :, None] * weights[None, :, :] * T_axes
    JtJ = np.einsum("kij,kil->kjl", jacobian, jacobian)
    covariance_unit = np.linalg.pinv(JtJ, hermitian=True)
    sigma_alpha = sigma_wn[:, None] * np.sqrt(
        np.diagonal(covariance_unit, axis1=1, axis2=2)
    )
    # axes with no residual sensitivity (fully saturated) are unconstrained
    no_sensitivity = np.diagonal(JtJ, axis1=1, axis2=2) < 1e-30
    sigma_alpha[no_sensitivity] = np.inf

    # censoring: absorbance is only a lower bound where the axis's largest
    # contribution to any measured transmittance, w_ij 10^(-alpha_j d_i),
    # is within the noise, or where the fit hit the hard cap
    censored = alpha >= 0.999 * alpha_max
    if np.isfinite(noise_std):
        contributions = weights[None, :, :] * T_axes
        censored |= contributions.max(axis=1) < censor_factor * noise_std

    # report absorbance at the reference thickness
    A_reference = alpha * reference_thickness_mm
    sigma_A_reference = sigma_alpha * reference_thickness_mm

    out = pd.DataFrame(
        {
            "wavenumber": wavenumber,
            "Aa": A_reference[:, 0],
            "Ab": A_reference[:, 1],
            "Ac": A_reference[:, 2],
            "Aa_sigma": sigma_A_reference[:, 0],
            "Ab_sigma": sigma_A_reference[:, 1],
            "Ac_sigma": sigma_A_reference[:, 2],
            "Aa_censored": censored[:, 0],
            "Ab_censored": censored[:, 1],
            "Ac_censored": censored[:, 2],
            "rms_residual": rms_residual,
        }
    )
    out.attrs = {
        "method_used": method_used,
        "condition_number": condition_number,
        "noise_std_T": noise_std,
        "alpha_max_per_mm": alpha_max,
        "thickness_spread": thickness_spread,
        "reference_thickness_mm": reference_thickness_mm,
    }

    return out


def build_coefficient_matrix(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray
) -> np.ndarray:
    """
    Compute the geometric weight (design) matrix of the Asimow model.

    The columns are the weights multiplying the principal-axis
    transmittances along a, b and c:

        col_a = cos²(theta) sin²(phi)
        col_b = sin²(theta) sin²(phi)
        col_c = cos²(phi)

    Parameters
    ----------
    theta_rad : np.ndarray
        a-axis to polarizer projection (E') angles in radians,
        shape (n_measurements,).
    phi_rad : np.ndarray
        c-axis to polarization vector (E) angles in radians,
        shape (n_measurements,).

    Returns
    -------
    np.ndarray
        The design matrix with shape (n_measurements, 3); rows sum to 1.
    """

    if theta_rad.shape != phi_rad.shape:
        raise ValueError("theta_rad and phi_rad must have the same shape.")

    col_a = np.cos(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    col_b = np.sin(theta_rad) ** 2 * np.sin(phi_rad) ** 2
    col_c = np.cos(phi_rad) ** 2

    return np.column_stack([col_a, col_b, col_c])


# ============================================================================ #
# AUXILIARY FUNCTIONS                                                          #
# ============================================================================ #

def _validate_synthesize_principal_spectra(
    spectra: pd.DataFrame,
    theta_deg: np.ndarray | pd.Series,
    phi_deg: np.ndarray | pd.Series,
    thickness_mm: np.ndarray | pd.Series,
    method: str,
    thickness_rtol: float,
    transmittance_floor: float,
    noise_std_T: float | None,
    censor_factor: float,
    reference_thickness_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate the inputs of `synthesize_principal_spectra` and return the
    parsed arrays.

    Returns
    -------
    tuple of np.ndarray
        (A_meas, theta_rad, phi_rad, d_mm) where A_meas has shape
        (n_wavenumbers, n_measurements) and the angle/thickness arrays
        have shape (n_measurements,).
    """

    if "wavenumber" not in spectra.columns:
        raise KeyError("`spectra` must contain a 'wavenumber' column.")

    A_meas = spectra.drop(columns="wavenumber").to_numpy(dtype=float)
    n_meas = A_meas.shape[1]
    if n_meas < 3:
        raise ValueError(
            f"At least 3 spectra are required to synthesize principal "
            f"spectra; got {n_meas}."
        )
    if not np.all(np.isfinite(A_meas)):
        raise ValueError(
            "`spectra` contains NaN or infinite values. Drop incomplete "
            "rows (e.g. interpolation edges) before calling this function."
        )

    theta = np.asarray(theta_deg, dtype=float)
    phi = np.asarray(phi_deg, dtype=float)
    d_mm = np.asarray(thickness_mm, dtype=float)
    for name, arr in (("theta_deg", theta), ("phi_deg", phi), ("thickness_mm", d_mm)):
        if arr.ndim != 1 or arr.size != n_meas:
            raise ValueError(
                f"`{name}` must be 1-D with one value per spectrum "
                f"({n_meas}); got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"`{name}` contains NaN or infinite values.")
    if np.any(d_mm <= 0):
        raise ValueError("All thicknesses must be positive.")

    if method not in {"auto", "linear", "nonlinear"}:
        raise ValueError('`method` must be "auto", "linear" or "nonlinear".')
    if thickness_rtol < 0:
        raise ValueError("`thickness_rtol` must be non-negative.")
    if not 0.0 < transmittance_floor < 1.0:
        raise ValueError("`transmittance_floor` must lie in (0, 1).")
    if noise_std_T is not None and not 0.0 < noise_std_T < 1.0:
        raise ValueError("`noise_std_T` must lie in (0, 1) or be None.")
    if censor_factor <= 0:
        raise ValueError("`censor_factor` must be positive.")
    if reference_thickness_mm <= 0:
        raise ValueError("`reference_thickness_mm` must be positive.")

    return A_meas, np.deg2rad(theta), np.deg2rad(phi), d_mm


def _solve_linear_transmittance(
    weights: np.ndarray,
    T_obs: np.ndarray,
    t_lower: float,
) -> np.ndarray:
    """
    Solve the equal-thickness linear mixing model with physical bounds.

    All wavenumbers are first solved at once with an unconstrained
    least-squares solve; only the wavenumbers whose solution violates
    the bounds [t_lower, 1] are re-solved with bounded-variable least
    squares (BVLS).

    Parameters
    ----------
    weights : np.ndarray
        Design matrix, shape (n_measurements, 3).
    T_obs : np.ndarray
        Observed transmittances, shape (n_wavenumbers, n_measurements).
    t_lower : float
        Lower bound on the principal transmittances.

    Returns
    -------
    np.ndarray
        Principal transmittances at the common thickness,
        shape (n_wavenumbers, 3).
    """

    T_principal, *_ = np.linalg.lstsq(weights, T_obs.T, rcond=None)
    T_principal = T_principal.T

    out_of_bounds = np.flatnonzero(
        np.any((T_principal < t_lower) | (T_principal > 1.0), axis=1)
    )
    for k in out_of_bounds:
        result = lsq_linear(
            weights, T_obs[k], bounds=(t_lower, 1.0), method="bvls"
        )
        T_principal[k] = result.x

    return T_principal


def _linearized_alpha_guess(
    weights: np.ndarray,
    d_mm: np.ndarray,
    A_meas: np.ndarray,
    alpha_max: float,
) -> np.ndarray:
    """
    Initial guess for the absorption coefficients from the linear-in-
    absorbance (weak absorption) approximation A_i ≈ d_i Σ_j w_ij α_j.

    Parameters
    ----------
    weights : np.ndarray
        Design matrix, shape (n_measurements, 3).
    d_mm : np.ndarray
        Thicknesses in mm, shape (n_measurements,).
    A_meas : np.ndarray
        Measured absorbances, shape (n_wavenumbers, n_measurements).
    alpha_max : float
        Upper bound used to clip the guess.

    Returns
    -------
    np.ndarray
        Initial guess for alpha, shape (n_wavenumbers, 3), within
        [0, alpha_max].
    """

    weights_d = weights * d_mm[:, None]
    alpha_guess, *_ = np.linalg.lstsq(
        weights_d, np.clip(A_meas, 0.0, None).T, rcond=None
    )

    return np.clip(alpha_guess.T, 0.0, alpha_max)


def _solve_nonlinear_alpha(
    weights: np.ndarray,
    T_obs: np.ndarray,
    d_mm: np.ndarray,
    alpha_guess: np.ndarray,
    alpha_max: float,
    robust: bool,
    loss_scale: float,
) -> np.ndarray:
    """
    Fit the principal absorption coefficients wavenumber by wavenumber
    with bounded nonlinear least squares (each spectrum at its own
    thickness).

    Parameters
    ----------
    weights : np.ndarray
        Design matrix, shape (n_measurements, 3).
    T_obs : np.ndarray
        Observed transmittances, shape (n_wavenumbers, n_measurements).
    d_mm : np.ndarray
        Thicknesses in mm, shape (n_measurements,).
    alpha_guess : np.ndarray
        Initial guesses, shape (n_wavenumbers, 3).
    alpha_max : float
        Upper bound on the absorption coefficients (per mm).
    robust : bool
        Use a soft-L1 loss instead of plain least squares.
    loss_scale : float
        Soft-L1 transition scale in transmittance units.

    Returns
    -------
    np.ndarray
        Fitted absorption coefficients per mm, shape (n_wavenumbers, 3).
    """

    n_wavenumbers = T_obs.shape[0]
    alpha = np.empty((n_wavenumbers, 3))
    loss = "soft_l1" if robust else "linear"
    previous_solution = None

    for k in range(n_wavenumbers):
        x0 = alpha_guess[k]
        # spectra are continuous across wavenumbers, so the previous
        # solution is often a better starting point than the linear guess
        if previous_solution is not None:
            cost_guess = np.sum(
                _model_residuals(x0, weights, d_mm, T_obs[k]) ** 2
            )
            cost_previous = np.sum(
                _model_residuals(previous_solution, weights, d_mm, T_obs[k]) ** 2
            )
            if cost_previous < cost_guess:
                x0 = previous_solution

        result = least_squares(
            _model_residuals,
            x0,
            jac=_model_jacobian,
            bounds=(0.0, alpha_max),
            args=(weights, d_mm, T_obs[k]),
            method="trf",
            loss=loss,
            f_scale=loss_scale,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        alpha[k] = result.x
        previous_solution = result.x

    return alpha


def _model_residuals(
    alpha: np.ndarray,
    weights: np.ndarray,
    d_mm: np.ndarray,
    T_obs_row: np.ndarray,
) -> np.ndarray:
    """
    Transmittance residuals of the mixing model at one wavenumber.

    Parameters
    ----------
    alpha : np.ndarray
        Absorption coefficients per mm, shape (3,).
    weights : np.ndarray
        Design matrix, shape (n_measurements, 3).
    d_mm : np.ndarray
        Thicknesses in mm, shape (n_measurements,).
    T_obs_row : np.ndarray
        Observed transmittances, shape (n_measurements,).

    Returns
    -------
    np.ndarray
        Model minus observed transmittance, shape (n_measurements,).
    """

    T_axes = 10.0 ** (-np.outer(d_mm, alpha))

    return np.sum(weights * T_axes, axis=1) - T_obs_row


def _model_jacobian(
    alpha: np.ndarray,
    weights: np.ndarray,
    d_mm: np.ndarray,
    T_obs_row: np.ndarray,
) -> np.ndarray:
    """
    Analytic Jacobian of `_model_residuals` with respect to alpha.

    Parameters
    ----------
    alpha : np.ndarray
        Absorption coefficients per mm, shape (3,).
    weights : np.ndarray
        Design matrix, shape (n_measurements, 3).
    d_mm : np.ndarray
        Thicknesses in mm, shape (n_measurements,).
    T_obs_row : np.ndarray
        Observed transmittances (unused; required by the solver
        signature), shape (n_measurements,).

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape (n_measurements, 3).
    """

    T_axes = 10.0 ** (-np.outer(d_mm, alpha))

    return -LN10 * d_mm[:, None] * weights * T_axes


def _estimate_noise_level(
    residuals: np.ndarray,
    rms_residual: np.ndarray,
    n_meas: int,
    noise_std_T: float | None,
) -> tuple[float, np.ndarray]:
    """
    Estimate the transmittance noise level and the per-wavenumber
    residual scale used for uncertainty propagation.

    The global noise level is a robust (median absolute deviation)
    estimate over all residuals, corrected for the 3 fitted parameters.
    The per-wavenumber scale is the local RMS residual floored by the
    global level, so misfit regions get honestly inflated uncertainties.

    Parameters
    ----------
    residuals : np.ndarray
        Fit residuals, shape (n_wavenumbers, n_measurements).
    rms_residual : np.ndarray
        Per-wavenumber RMS residual, shape (n_wavenumbers,).
    n_meas : int
        Number of measurements.
    noise_std_T : float | None
        User-supplied noise standard deviation, or None to estimate it.

    Returns
    -------
    noise_std : float
        Global noise level (NaN if it cannot be determined).
    sigma_wn : np.ndarray
        Per-wavenumber residual scale, shape (n_wavenumbers,).
    """

    if n_meas > 3:
        dof_factor = np.sqrt(n_meas / (n_meas - 3))
        estimated = 1.4826 * np.median(np.abs(residuals)) * dof_factor
        noise_std = estimated if noise_std_T is None else noise_std_T
        sigma_wn = np.maximum(rms_residual * dof_factor, noise_std)
    else:
        # exactly determined system: residuals are ~0 by construction
        if noise_std_T is None:
            warnings.warn(
                "With exactly 3 spectra the model interpolates the data and "
                "the noise level cannot be estimated from residuals; "
                "uncertainties are reported as NaN and censoring flags are "
                "disabled. Pass `noise_std_T` to enable them."
            )
            noise_std = np.nan
        else:
            noise_std = noise_std_T
        sigma_wn = np.full(rms_residual.size, noise_std)

    return noise_std, sigma_wn


# End of function definitions

# End of file
