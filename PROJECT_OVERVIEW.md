# FTIRkit — Project Overview

> Snapshot of the repository structure as of 2026-07-03, after adding the
> principal-axis spectra synthesis module. Scope: package layout,
> module/function inventory, and internal dependencies. Function correctness
> is **not** assessed here.

FTIRkit estimates crystal orientation and synthesises principal-axis spectra
from polarized μ-FTIR data. The code is organised as an installable Python
package (`ftirkit`) using a `src` layout.

```
FTIRkit/
├── pyproject.toml                 # package metadata (pip-installable)
├── src/
│   ├── ftirkit/
│   │   ├── __init__.py            # version + key re-exports
│   │   ├── transmittance_model.py # core physical model (Asimow et al. 2006)
│   │   ├── geometry.py            # coordinate transforms, grids, vector ops
│   │   ├── crystallography.py     # orientations, symmetry, misorientation
│   │   ├── spectrum_tools.py      # spectrum processing utilities
│   │   ├── principal_spectra.py   # principal-axis spectra synthesis
│   │   ├── synthetic.py           # synthetic FTIR data generation
│   │   ├── plots.py               # 3D visualisation
│   │   └── orientation/
│   │       ├── __init__.py
│   │       ├── common.py          # shared optimisation scaffolding
│   │       ├── lambda_method.py   # orientation from single-λ sections
│   │       └── spectrum_method.py # orientation from a full spectrum
│   └── deprecated/
│       ├── principal_spectra.py           # old linear synthesis (Asimow trick)
│       └── principal_spectra_nonlinear.py # old thickness-aware prototype
└── tests/
    ├── smoke_test.py                  # end-to-end runnable smoke test
    └── validate_principal_spectra.py  # ground-truth validation of synthesis
```

## Internal dependencies

The package has two foundation layers that everything else builds on:

```
transmittance_model ─┐
geometry ────────────┼──> crystallography ──> orientation/spectrum_method
                     ├──> synthetic           orientation/lambda_method
spectrum_tools ──────┤    plots               (both also use orientation/common)
                     └──> orientation/*
```

- `transmittance_model`, `geometry`, and `spectrum_tools` import nothing from
  the package (only NumPy/SciPy/Pandas).
- `principal_spectra` is self-contained (only NumPy/SciPy/Pandas); it carries
  its own copy of `build_coefficient_matrix`.
- `crystallography` imports vector helpers from `geometry`.
- `synthetic` imports the model, geometry helpers, and a spectrum conversion.
- `orientation/*` imports the model, geometry, crystallography,
  `spectrum_tools`, and the shared scaffolding in `orientation/common`.
- `plots` imports vector helpers from `geometry`.

Top-level re-exports (`from ftirkit import ...`): `calc_transmittance`,
`find_orientation_based_on_lambda`, `find_orientation_based_on_spectrum`,
`synthesize_principal_spectra`.
Heavier modules (`plots`, `synthetic`) are imported on demand.

---

## Module-by-module function inventory

### `transmittance_model.py` — the core physical model

| Function | Description |
|---|---|
| `calc_transmittance(standard, theta_rad, phi_rad, d)` | Asimow et al. (2006) transmittance model: T(λ,φ,θ,d) = Ta^d cos²θ sin²φ + Tb^d sin²θ sin²φ + Tc^d cos²φ, computed from principal-axis spectra (Ta, Tb, Tc). The single shared implementation used package-wide. |

### `geometry.py` — coordinate systems, spherical grids, vector/rotation ops

| Function | Description |
|---|---|
| `sph2cart(r, azimuth_rad, polar_rad)` | Convert spherical/polar coordinates (ISO physics convention) to Cartesian x, y, z. Polar angle defaults to 90° (XY plane). |
| `cart2sph(x, y, z)` | Convert Cartesian coordinates to spherical (r, azimuth φ ∈ [0, 2π), polar θ ∈ [0, π]); sets azimuth to 0 at the poles. |
| `regular_S2_grid(n_squared)` | Generate a regular meshgrid on the unit sphere (uniform azimuths × equal-area-spaced colatitudes). |
| `gauss_legendre_S2_meshgrid(n_polar, n_azimuth, degrees)` | Tensor-product spherical meshgrid using Gauss–Legendre nodes in cos(polar) for better point packing than a regular grid. |
| `rotate(coordinates, euler_deg, invert)` | Rotate points/grids in Cartesian space by Bunge (zxz) Euler angles; handles (3,), (n, 3) and grid inputs. |
| `project_vector_onto_plane(vector, plane_normal)` | Orthogonal projection of a 3D vector onto a plane defined by its normal. |
| `counterclockwise_angle(vector_1, vector_2, normal)` | Signed (counter-clockwise, right-hand rule) angle from one vector to another about a given axis, in [0, 2π). |
| `_axis_angle_rotation(axis, angle)` *(private)* | 3×3 rotation matrix for a rotation about an arbitrary axis. Currently unused (kept as scaffolding). |

### `crystallography.py` — orientations, symmetry, misorientation

| Function | Description |
|---|---|
| `convert_Euler_to_Asimow_angles(euler_angles_deg, degrees)` | Convert Bunge (zxz) Euler angles to Asimow angles (φ: c-axis↔polarizer E; θ: a-axis↔E′ projection). Handles single or multiple orientations. |
| `calc_misorientation(euler1_deg, euler2_deg, precision)` | Misorientation angle between two orientations (no symmetry considered; intended for small angles). |
| `calc_disorientation(euler1_deg, euler2_deg, all)` | Disorientation (minimum symmetry-equivalent misorientation) between two orthorhombic crystals; optionally returns all 8 misorientations. |
| `symmetrise_orthorhombic_Euler(Euler_angles_deg, output_unit, canonical)` | All 8 symmetry-equivalent Bunge Euler triplets for point group mmm; optionally only the canonical (fundamental-zone) one. |
| `symmetrise_orthorhombic_Asimow_angles(theta_deg, phi_deg)` | All symmetry-equivalent (θ, φ) Asimow angle pairs under mmm symmetry. |
| `canonical_orthorhombic_Asimow_angles(theta_deg, phi_deg, show)` | Map an Asimow angle pair to its unique canonical representative (first octant). |
| `minimum_angular_distance(direction1_rad, direction2_rad)` | Great-circle (haversine) angular distance between two directions on the unit sphere. |
| `_euler_Asimow_misfit(euler_angles_deg, target_Asimow_deg)` *(private)* | Squared spherical angular separation between the Asimow direction of a Euler triplet and a target. Currently unused (scaffolding for a planned Euler-from-Asimow inversion). |

### `spectrum_tools.py` — spectrum processing

| Function | Description |
|---|---|
| `smooth_spectrum(data, window_size)` | Smooth a 1D spectrum with a Savitzky–Golay filter (moving-average fallback for short data/windows). |
| `smooth_spectra(spectra, window_size)` | DataFrame wrapper: apply `smooth_spectrum` to every column except `wavenumber`. |
| `interpolate_spectra(target_wavenumbers, spectra, method)` | Interpolate spectra onto a target wavenumber grid using PCHIP (shape-preserving) or linear interpolation. |
| `absorbance_to_transmittance(absorbance, clip)` | Convert absorbance to transmittance, T = 10^(−A), with optional clipping to (0, 1]. |
| `abs_to_trans_batch(spectra, clip)` | DataFrame wrapper for `absorbance_to_transmittance` over all spectrum columns. |
| `transmittance_to_absorbance(transmittance, clip)` | Convert transmittance to absorbance, A = −log10(T), with optional clipping. |

### `principal_spectra.py` — principal-axis spectra synthesis

(Method of Asimow et al., 2006, with physically correct thickness handling
and a numerically robust transmittance→absorbance back-transformation)

| Function | Description |
|---|---|
| `synthesize_principal_spectra(spectra, theta_deg, phi_deg, thickness_mm, reference_thickness_mm, method, thickness_rtol, transmittance_floor, noise_std_T, censor_factor, robust, loss_scale)` | Synthesise principal-axis absorbances (Aa, Ab, Ac) at a reference thickness from a set of polarized absorbance spectra. Solves in transmittance (where the Asimow mixing model is valid), fitting principal absorption coefficients per mm at each wavenumber. `method="auto"` uses a fast bounded **linear** solve when all thicknesses agree within `thickness_rtol`, else a bounded **nonlinear** fit with each spectrum at its own thickness (no invalid pre-normalisation of mixed spectra). Returns propagated 1σ uncertainties (`A*_sigma`), censoring flags (`A*_censored`, True = lower bound only), per-wavenumber `rms_residual`, and fit diagnostics in `DataFrame.attrs`. |
| `build_coefficient_matrix(theta_rad, phi_rad)` | Geometric weight (design) matrix of the Asimow model; columns are the weights on Ta, Tb, Tc (cos²θ sin²φ, sin²θ sin²φ, cos²φ); rows sum to 1. |
| `_validate_synthesize_principal_spectra(...)` *(private)* | Input validation helper for `synthesize_principal_spectra` (shapes, finiteness, positivity, option domains). |
| `_solve_linear_transmittance(weights, T_obs, t_lower)` *(private)* | Equal-thickness path: vectorised `lstsq` in transmittance, with BVLS re-solves only where the physical bounds [t_lower, 1] are violated. |
| `_linearized_alpha_guess(weights, d_mm, A_meas, alpha_max)` *(private)* | Initial guess for the absorption coefficients from the weak-absorption (linear-in-absorbance) approximation. |
| `_solve_nonlinear_alpha(weights, T_obs, d_mm, alpha_guess, alpha_max, robust, loss_scale)` *(private)* | Per-wavenumber bounded nonlinear least squares (analytic Jacobian, warm-started across wavenumbers); optional soft-L1 robust loss. |
| `_model_residuals(alpha, weights, d_mm, T_obs_row)` *(private)* | Transmittance residuals of the mixing model at one wavenumber. |
| `_model_jacobian(alpha, weights, d_mm, T_obs_row)` *(private)* | Analytic Jacobian of `_model_residuals` w.r.t. the absorption coefficients. |
| `_estimate_noise_level(residuals, rms_residual, n_meas, noise_std_T)` *(private)* | Robust (MAD) transmittance noise estimate and per-wavenumber residual scale used for uncertainty propagation and censoring. |

### `synthetic.py` — synthetic FTIR data generation

| Function | Description |
|---|---|
| `generate_spectra(theta_deg, phi_deg, standard, thickness, output, wavelength_colname)` | Generate synthetic FTIR spectra (absorbance or transmittance) for one or more orientations from a principal-axis standard. |
| `generate_section(euler_deg, standard_Ts_1mm, thickness, sample_size, azimuth_range_deg, noise, grid_resolution)` | Generate a synthetic XY-section dataset (T vs. polarizer angle) for a crystal orientation, optionally with Gaussian noise. *(Docstring is a placeholder.)* |
| `extract_XY_section_ang(x, y, z)` | Use ContourPy to extract the z=0 section of a transmittance envelope → DataFrame (x, y, T, in-plane angle in degrees [0, 360)). |
| `_find_nearest(df, values)` *(private)* | Indices of the nearest values in a pandas Series for each query value. |

### `orientation/common.py` — shared optimisation scaffolding

| Function | Description |
|---|---|
| `get_initial_guesses(bounds, num_guesses)` | Draw uniform random starting points within the parameter bounds (one column per bound, so it handles Asimow or Euler parameterisations). |
| `explore_Euler_space(step, lower_bounds, upper_bounds)` | Uniform grid of Euler triplets over a (default orthorhombic mmm) fundamental zone. |
| `summarize_results(result, elapsed_time, alg_name, angle_format, num_guesses)` | Print a summary of a minimisation result. `angle_format` selects how the parameters are reported: `"euler_deg"`, `"euler_rad"`, or `"asimow_rad"`. |

### `orientation/lambda_method.py` — orientation from single-wavelength sections

(Method of Lopez-Sanchez & Padrón-Navarta, 2026)

| Function | Description |
|---|---|
| `extract_section_from_spectra(spectra, angles2pol_deg, wavenumber)` | Extract transmittance values at one wavenumber from a set of spectra → DataFrame (T, angle-to-polarizer) ready for section-based analysis. *(TODO noted in code: auto-select the nearest wavenumber instead of requiring an exact match.)* |
| `find_orientation_based_on_lambda(transmitances, angles2pol_deg, principal_Ts, algorithm, num_guesses, upper_bounds, thickness_bounds, **kwargs)` | Estimate crystal orientation (Euler angles + optional thickness) from polarized single-wavelength measurements via L-BFGS-B (multi-start), differential evolution, and/or dual annealing. |
| `bruteforce_algorithm(measurements, standard_Ts_1mm, step)` | Grid search over Euler space for the best-fit orientation (testing/validation only). **Known issue:** it feeds 3-element Euler triplets to `_misfit_function`, which unpacks 4 parameters (angles + thickness), so it currently fails. |
| `_misfit_function(params, measurements, principal_Ts)` *(private)* | Objective: sum of squared differences between measured and theoretical T after rotating measurements into the crystal frame. |

### `orientation/spectrum_method.py` — orientation from a full spectrum

(Modified method of Asimow et al., 2006)

| Function | Description |
|---|---|
| `find_orientation_based_on_spectrum(standard, spectrum, algorithm, thickness_bound, smooth_window, num_guesses, transmittance_range, exclude_wavenumbers, canonical_angles)` | Estimate Asimow angles (θ, φ) + optional thickness from a single measured spectrum by minimising derivative misfit (L-BFGS-B multi-start and/or differential evolution). Euler users convert separately via `convert_Euler_to_Asimow_angles`. Optional filters restrict the misfit to measured transmittances within a (min, max) range and/or exclude wavenumber ranges (restores the "squash" channel-exclusion of the original Asimow C code). By default the fitted angles are folded to their canonical first-octant representative (mmm symmetry); disable with `canonical_angles=False`. |
| `_validate_find_orientation_based_on_spectrum(standard, spectrum, algorithm, thickness_bound, num_guesses)` *(private)* | Input validation (standard columns, spectrum grid/finiteness/range, algorithm normalisation incl. tuple input, thickness bounds). |
| `_build_inclusion_mask(wavenumbers, T_measured, transmittance_range, exclude_wavenumbers, n_free_params)` *(private)* | Boolean mask of spectral points contributing to the misfit; validates the filter arguments and that enough points remain. |
| `_error_function(params, wavelengths, dT_measured, Ta/Tb/Tc_values, smooth_window, include_mask)` *(private)* | Objective: sum of squared differences between the (precomputed) measured derivative and the smoothed synthetic derivative, optionally restricted to masked-in points (smoothing/derivatives always on the full grid). |
| `_fold_result_to_canonical(result)` *(private)* | Fold the fitted Asimow angles of an OptimizeResult into the canonical first-octant representative (mmm), in place. |

### `plots.py` — visualisation

| Function | Description |
|---|---|
| `plot_crystal_axes(euler_angles_deg)` | 3D plot of crystal axes (a, b, c) at a given orientation, with the polarizer vector E, its projection E′, angle arcs, and the a–b plane. |
| `plot_transmitance_envelope(coordinates_xyz, T_values)` | 3D surface plot of a transmittance envelope, coloured by T value. |
| `plot_XY_section(coordonates_xy)` | **Stub — not implemented (`pass`).** |
| `_generate_arc_points(vector_1, vector_2, radius, num_points)` *(private)* | Points along an arc between two 3D vectors (for drawing angle arcs). |
| `_generate_plane_points(vector1, vector2, num_points)` *(private)* | Meshgrid of points spanning the plane defined by two vectors. *(Docstring is a placeholder.)* |

### `deprecated/principal_spectra.py` — principal-axis spectra synthesis (old)

Kept outside the package, untouched, for reference. **Superseded by
`ftirkit.principal_spectra`.** Self-contained (it carries its own copies of
`calc_transmittance` and `smooth_spectrum`). Uses Asimow's thickness pre-
normalisation and rescaling trick, which is only approximate for mixed
spectra and can produce absorbance spikes on noisy data.

| Function | Description |
|---|---|
| `calc_transmittance(principal_Ts, theta_rad, phi_rad, d)` | Same Asimow transmittance model (local copy). |
| `estimate_principal_axis_transmittance(T_obs, theta_deg, phi_deg, evaluate_model)` | Estimate (Ta, Tb, Tc) from measured transmittances by linear least squares. |
| `apply_principal_spectra_to_dataframe(Abs_spectra, theta_deg, phi_deg, thickness_mm, evaluate_model)` | Apply the least-squares solver row-wise over a DataFrame of absorbance spectra; returns principal-axis absorbances (Aa, Ab, Ac). |
| `smooth_spectrum(data, window_size)` | Savitzky–Golay smoothing (local copy). |
| `build_coefficient_matrix(theta_rad, phi_rad)` | Design matrix A for the least-squares system A·[Ta, Tb, Tc] = T_obs. |
| `evaluate_model_fit(principal_Ts, T_measured, theta_rad, phi_rad)` | R² and RMSE of the fitted least-squares model. |

### `deprecated/principal_spectra_nonlinear.py` — thickness-aware prototype (old)

Kept outside the package for reference. **Superseded by
`ftirkit.principal_spectra`**, which folds in the same thickness-aware idea
(fitting absorption per mm) plus the linear fast path, uncertainty
propagation, and censoring flags. Self-contained.

| Function | Description |
|---|---|
| `build_coefficient_matrix(theta_rad, phi_rad)` | Geometric weight matrix (local copy). |
| `estimate_principal_axis_transmittance(T_obs, theta_deg, phi_deg, thickness_mm, robust, ...)` | Per-wavenumber nonlinear fit of absorption per mm; returns 1-mm transmittances. |
| `apply_principal_spectra_to_dataframe(Abs_spectra, theta_deg, phi_deg, thickness_mm, evaluate_model, reference_mm, robust)` | Driver returning principal T/A at a reference thickness. |
| `calc_transmittance(principal_Ts, theta_rad, phi_rad, d)` | Mixed transmittance at thickness d from 1-mm principal transmittances. |
| `_model_transmittance_from_alpha`, `_initial_linear_guess` *(private)* | Forward model and linearized initial guess. |

### `tests/smoke_test.py`

Runnable end-to-end check (`python tests/smoke_test.py` from the repository
root): exercises the cross-module import paths, Euler→Asimow conversion,
synthetic spectra/section generation, both orientation solvers on synthetic
data, and the crystal-axes plot.

### `tests/validate_principal_spectra.py`

Runnable ground-truth validation of `ftirkit.principal_spectra`
(`python tests/validate_principal_spectra.py` from the repository root):
mixes synthetic principal spectra with the Asimow model at known
orientations/thicknesses, adds detector-like transmittance noise, and checks
recovery within propagated uncertainties, absence of unflagged absorbance
spikes (versus the naive lstsq+log₁₀ pipeline), correct linear/nonlinear path
selection, censoring of saturated peaks, and the edge cases (3 spectra,
noiseless data, `method="linear"` rejected for unequal thicknesses).

---

## Known open items

- `plots.plot_XY_section` is an unimplemented stub.
- `bruteforce_algorithm` (lambda method) passes 3 parameters to a 4-parameter
  objective (see table above).
- Placeholder docstrings (`_summary_` / `_description_`) remain in
  `generate_section`, `extract_XY_section_ang`, `extract_section_from_spectra`,
  `bruteforce_algorithm`, `_find_nearest`, and `_generate_plane_points`.
- Unused scaffolding kept on purpose: `geometry._axis_angle_rotation` and
  `crystallography._euler_Asimow_misfit` (planned Euler-from-Asimow inversion).
- TODOs in code: input checking in `smooth_spectra`, `extract_section_from_spectra`;
  nearest-wavenumber selection in `extract_section_from_spectra`; NaN
  propagation and sampling diagnostics in `interpolate_spectra`; configurable
  colormap and axis lengths in `plots`.
- Naming consistency (deferred): `standard` vs. `principal_Ts` vs.
  `standard_Ts_1mm` all denote the (Ta, Tb, Tc) concept; typos `transmitances`
  and `coordonates_xy` remain in public signatures.
