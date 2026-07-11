"""Smoke test for the ftirkit package: exercises the cross-module import
paths and the main entry points with synthetic data.

Run from the repository root:
    python tests/smoke_test.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

# crystallography → geometry path
from ftirkit.crystallography import (
    convert_Euler_to_Asimow_angles,
    calc_disorientation,
    symmetrise_orthorhombic_Euler,
)
asimow = convert_Euler_to_Asimow_angles(np.array([30.0, 40.0, 50.0]))
assert asimow.shape == (2,), asimow
assert symmetrise_orthorhombic_Euler(np.array([30.0, 40.0, 50.0])).shape == (8, 3)
assert np.isscalar(calc_disorientation([10, 20, 30], [11, 21, 31]).item())
print("crystallography OK:", asimow)

# synthetic → transmittance_model / geometry / spectrum_tools path
from ftirkit.synthetic import generate_spectra, generate_section
wn = np.linspace(1900, 2100, 201)
standard = pd.DataFrame({
    "wavenumber": wn,
    "Ta": 0.4 + 0.2 * np.exp(-((wn - 2000) / 20) ** 2),
    "Tb": 0.5 + 0.1 * np.exp(-((wn - 1980) / 15) ** 2),
    "Tc": 0.6 + 0.15 * np.exp(-((wn - 2020) / 25) ** 2),
})
spectra, angles = generate_spectra([30.0, 60.0], [45.0, 80.0], standard)
assert list(spectra.columns) == ["wavenumber", "spec1", "spec2"]
assert (spectra[["spec1", "spec2"]] >= 0).all().all()  # absorbance
spectra_T, _ = generate_spectra(30.0, 45.0, standard, output="transmittance")
assert spectra_T["spec1"].between(0, 1).all()
section = generate_section(
    euler_deg=np.array([20.0, 30.0, 40.0]),
    standard_Ts_1mm=(0.5, 0.6, 0.7),
)
assert {"T_values", "angle_deg"} <= set(section.columns)
print("synthetic OK:", spectra.shape, section.shape)

# orientation (lambda method) full solver path
from ftirkit.orientation.lambda_method import find_orientation_based_on_lambda
results = find_orientation_based_on_lambda(
    transmittances=section["T_values"].to_numpy(),
    angles2pol_deg=section["angle_deg"].to_numpy(),
    standard_Ts_1mm=(0.5, 0.6, 0.7),
    algorithm=["gradient"],
    num_guesses=3,
)
assert "gradient_based" in results
print("lambda_method OK")

# orientation (spectrum method) full solver path
from ftirkit.orientation.spectrum_method import find_orientation_based_on_spectrum
results2 = find_orientation_based_on_spectrum(
    standard=standard,
    spectrum=spectra_T["spec1"].to_numpy(),
    algorithm="gradient",
    num_guesses=3,
)
assert "gradient_based" in results2
# canonical folding (default) maps the result to the first octant
theta_fit, phi_fit = np.degrees(results2["gradient_based"].x[:2])
assert 0.0 <= theta_fit <= 90.0 and 0.0 <= phi_fit <= 90.0, (theta_fit, phi_fit)
print("spectrum_method OK")

# plots (fixed call sites: project_vector_onto_plane and counterclockwise_angle)
from ftirkit.plots import plot_crystal_axes
fig, ax = plot_crystal_axes([30, 40, 50])
print("plots OK")

print("ALL SMOKE TESTS PASSED")
