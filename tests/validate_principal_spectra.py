"""Validation of ftirkit.principal_spectra against synthetic ground truth.

Builds synthetic principal absorption spectra, mixes them with the Asimow
et al. (2006) transmittance model at known orientations/thicknesses, adds
detector-like Gaussian noise in transmittance, and checks that
`synthesize_principal_spectra`:

1. recovers the principal absorbances within the propagated uncertainties,
2. produces no unflagged absorbance spikes at strong peaks (the failure
   mode of the naive lstsq + -log10 pipeline, reproduced here for
   comparison),
3. handles variable thickness correctly (nonlinear path),
4. behaves sensibly in the edge cases (3 spectra, noiseless data,
   linear method requested with unequal thicknesses).

Run from the repository root:
    python tests/validate_principal_spectra.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from ftirkit.principal_spectra import (
    build_coefficient_matrix,
    synthesize_principal_spectra,
)

rng = np.random.default_rng(42)

REFERENCE_MM = 10.0  # report at 1 cm


# --------------------------------------------------------------------------- #
# Synthetic ground truth
# --------------------------------------------------------------------------- #

def gaussian(wn, center, amplitude, width):
    return amplitude * np.exp(-((wn - center) / width) ** 2)


wn = np.linspace(3200.0, 3700.0, 800)

# principal absorption coefficients (per mm). The alpha_a peak at 3480 is
# strong enough that principal transmittance drops below the noise floor
# at the measurement thicknesses -> exercises the censoring logic.
alpha_true = np.column_stack([
    0.02 + gaussian(wn, 3480, 6.0, 12) + gaussian(wn, 3560, 1.5, 18),
    0.015 + gaussian(wn, 3400, 0.8, 20) + gaussian(wn, 3480, 0.5, 14),
    0.01 + gaussian(wn, 3330, 0.3, 16) + gaussian(wn, 3610, 0.4, 22),
])
A_true_ref = alpha_true * REFERENCE_MM

# orientations (well spread, checked below via the condition number)
n_meas = 8
theta_deg = rng.uniform(0.0, 360.0, n_meas)
phi_deg = np.degrees(np.arccos(rng.uniform(-1.0, 1.0, n_meas)))
W = build_coefficient_matrix(np.deg2rad(theta_deg), np.deg2rad(phi_deg))
cond_W = np.linalg.cond(W)
assert cond_W < 20, f"orientation draw is poorly spread (cond={cond_W:.1f})"
print(f"Orientation set: {n_meas} spectra, cond(W) = {cond_W:.2f}")

NOISE_STD_T = 2e-4  # detector-like noise, constant in transmittance


def make_measurements(thickness_mm, noise_std):
    """Mix the true principal spectra and return an absorbance DataFrame."""
    d = np.asarray(thickness_mm, dtype=float)
    T_axes = 10.0 ** (-alpha_true[:, None, :] * d[None, :, None])
    T_mix = np.sum(W[None, :, :] * T_axes, axis=2)
    if noise_std > 0:
        T_mix = T_mix + rng.normal(0.0, noise_std, T_mix.shape)
    T_mix = np.clip(T_mix, 1e-12, None)  # detector cannot report T <= 0
    A_mix = -np.log10(T_mix)
    data = {"wavenumber": wn}
    for i in range(d.size):
        data[f"spec{i + 1}"] = A_mix[:, i]
    return pd.DataFrame(data)


def naive_linear_synthesis(spectra, thickness_mm):
    """The failure-prone pipeline: unconstrained lstsq in T + -log10."""
    A = spectra.drop(columns="wavenumber").to_numpy()
    T_obs = 10.0 ** (-np.clip(A, 0.0, None))
    T_hat, *_ = np.linalg.lstsq(W, T_obs.T, rcond=None)
    T_hat = np.clip(T_hat.T, np.finfo(float).eps, 1.0)
    return -np.log10(T_hat) * REFERENCE_MM / np.mean(thickness_mm)


def report_errors(label, out, expect_censored_axis=None):
    """Compare recovered vs true absorbance; assert on unflagged points."""
    A_rec = out[["Aa", "Ab", "Ac"]].to_numpy()
    sigma = out[["Aa_sigma", "Ab_sigma", "Ac_sigma"]].to_numpy()
    censored = out[["Aa_censored", "Ab_censored", "Ac_censored"]].to_numpy()
    err = A_rec - A_true_ref

    ok = ~censored & np.isfinite(sigma)
    z = np.abs(err[ok]) / np.maximum(sigma[ok], 1e-12)
    frac_within_4sigma = np.mean(z < 4.0)
    moderate = ok & (A_true_ref < 20.0)
    max_err_moderate = np.max(np.abs(err[moderate]))
    print(f"[{label}] method={out.attrs['method_used']}, "
          f"noise_hat={out.attrs['noise_std_T']:.2e}, "
          f"censored={censored.sum()} pts, "
          f"max|err| (A<20, unflagged)={max_err_moderate:.3f}, "
          f"P(|z|<4)={frac_within_4sigma:.3f}")

    assert frac_within_4sigma > 0.97, f"{label}: too many outliers"
    assert max_err_moderate < 2.0, f"{label}: large error on reliable points"
    if expect_censored_axis is not None:
        # the saturated strong peak must be flagged, and only axis a
        peak = np.abs(wn - 3480) < 5
        assert censored[peak, expect_censored_axis].any(), (
            f"{label}: saturated peak not flagged as censored"
        )
    return err, censored


# --------------------------------------------------------------------------- #
# Case A: equal thickness, realistic noise (linear path)
# --------------------------------------------------------------------------- #
d_equal = np.full(n_meas, 0.7)
spectra_A = make_measurements(d_equal, NOISE_STD_T)

out_A = synthesize_principal_spectra(
    spectra_A, theta_deg, phi_deg, d_equal, reference_thickness_mm=REFERENCE_MM
)
assert out_A.attrs["method_used"] == "linear"
err_A, cens_A = report_errors("A: equal d, noisy", out_A, expect_censored_axis=0)

# naive pipeline comparison: absorbance spikes at the strong peak
A_naive = naive_linear_synthesis(spectra_A, d_equal)
naive_err = np.abs(A_naive - A_true_ref)
unflagged = ~cens_A
print(f"    naive lstsq+log10 max|err| = {np.nanmax(naive_err):.1f} "
      f"(new method, unflagged pts: {np.max(np.abs(err_A[unflagged])):.3f})")
assert np.nanmax(naive_err) > 10 * np.max(np.abs(err_A[unflagged])), (
    "expected the naive pipeline to show much larger spikes"
)

# --------------------------------------------------------------------------- #
# Case B: variable thickness (+-20%), realistic noise (nonlinear path).
# Thicknesses are kept >= 0.8 mm so the strong alpha_a peak stays saturated
# (contribution < noise) and must be flagged as censored.
# --------------------------------------------------------------------------- #
d_var = rng.uniform(0.8, 1.2, n_meas)
spectra_B = make_measurements(d_var, NOISE_STD_T)

out_B = synthesize_principal_spectra(
    spectra_B, theta_deg, phi_deg, d_var, reference_thickness_mm=REFERENCE_MM
)
assert out_B.attrs["method_used"] == "nonlinear"
report_errors("B: variable d, noisy", out_B, expect_censored_axis=0)

# --------------------------------------------------------------------------- #
# Case C: noiseless data -> near-exact recovery
# --------------------------------------------------------------------------- #
spectra_C = make_measurements(d_equal, 0.0)
out_C = synthesize_principal_spectra(
    spectra_C, theta_deg, phi_deg, d_equal, reference_thickness_mm=REFERENCE_MM
)
cens_C = out_C[["Aa_censored", "Ab_censored", "Ac_censored"]].to_numpy()
err_C = out_C[["Aa", "Ab", "Ac"]].to_numpy() - A_true_ref
max_err_C = np.max(np.abs(err_C[~cens_C]))
print(f"[C: equal d, noiseless] max|err| (unflagged) = {max_err_C:.2e}")
assert max_err_C < 1e-6, "noiseless linear recovery should be near-exact"

spectra_C2 = make_measurements(d_var, 0.0)
out_C2 = synthesize_principal_spectra(
    spectra_C2, theta_deg, phi_deg, d_var, reference_thickness_mm=REFERENCE_MM
)
cens_C2 = out_C2[["Aa_censored", "Ab_censored", "Ac_censored"]].to_numpy()
err_C2 = out_C2[["Aa", "Ab", "Ac"]].to_numpy() - A_true_ref
max_err_C2 = np.max(np.abs(err_C2[~cens_C2]))
print(f"[C2: variable d, noiseless] max|err| (unflagged) = {max_err_C2:.2e}")
assert max_err_C2 < 1e-3, "noiseless nonlinear recovery should be accurate"

# --------------------------------------------------------------------------- #
# Case D: exactly 3 spectra -> works, warns about unknown noise level
# --------------------------------------------------------------------------- #
import warnings as _warnings
with _warnings.catch_warnings(record=True) as caught:
    _warnings.simplefilter("always")
    out_D = synthesize_principal_spectra(
        spectra_A[["wavenumber", "spec1", "spec2", "spec3"]],
        theta_deg[:3], phi_deg[:3], d_equal[:3],
        reference_thickness_mm=REFERENCE_MM,
    )
assert any("noise level cannot be estimated" in str(w.message) for w in caught)
assert np.isnan(out_D.attrs["noise_std_T"])
assert out_D[["Aa_sigma", "Ab_sigma", "Ac_sigma"]].isna().all().all()
print("[D: 3 spectra] runs, warns, sigmas are NaN as documented")

# with a user-supplied noise level the flags come back
out_D2 = synthesize_principal_spectra(
    spectra_A[["wavenumber", "spec1", "spec2", "spec3"]],
    theta_deg[:3], phi_deg[:3], d_equal[:3],
    reference_thickness_mm=REFERENCE_MM, noise_std_T=NOISE_STD_T,
)
assert out_D2[["Aa_censored", "Ab_censored", "Ac_censored"]].to_numpy().any()
print("[D2: 3 spectra + noise_std_T] censoring flags enabled")

# --------------------------------------------------------------------------- #
# Case E: method='linear' with unequal thickness must be rejected
# --------------------------------------------------------------------------- #
try:
    synthesize_principal_spectra(
        spectra_B, theta_deg, phi_deg, d_var, method="linear"
    )
except ValueError as exc:
    assert "thickness spread" in str(exc)
    print("[E: linear + unequal d] correctly rejected:", str(exc)[:60], "...")
else:
    raise AssertionError("method='linear' with unequal d should raise")

print("\nALL CHECKS PASSED")
