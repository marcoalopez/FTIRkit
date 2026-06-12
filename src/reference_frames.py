# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: reference_frames.py                                               #
# Description: Implements tools for dealing with reference frames             #
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

def sph2cart(r, azimuth_rad, polar_rad=np.deg2rad(90)):
    """
    Convert from spherical/polar (magnitude, azimuth, polar) to
    cartesian coordinates. Azimuth and polar angles are as used in
    physics (ISO 80000-2:2019) and in radians. If the polar angle is
    not given, the coordinate is assumed to lie on the XY plane.

    Parameters
    ----------
    r : int, float or array
        radial distance (magnitud of the vector)
    azimuth_rad : int, float or array with values between 0 and 2*pi
        azimuth angle respect to the x-axis direction in radians
    polar_rad : int, float or array with values between 0 and pi,
        polar angle respect to the zenith (z) direction in radians
        optional. Optional, defaults to np.deg2rad(90)

    Returns
    -------
    numpy ndarrays (1d)
        three numpy 1d arrays with the cartesian x, y, and z coordinates
    """

    x = r * np.sin(polar_rad) * np.cos(azimuth_rad)
    y = r * np.sin(polar_rad) * np.sin(azimuth_rad)
    z = r * np.cos(polar_rad)

    return x, y, z


def cart2sph(x, y, z):
    """
    Converts 3D rectangular cartesian coordinates to spherical
    coordinates.

    Parameters
    ----------
    x, y, z : float or array_like
        Cartesian coordinates.

    Returns
    -------
    r, theta, phi : float or array_like
        Spherical coordinates:
        - r: radial distance,
        - theta: inclination or polar angle (range from 0 to π),
        - phi: azimuthal angle (range from 0 to 2π).

    Notes
    -----
    This function follows the ISO 80000-2:2019 norm (physics convention).
    The input coordinates (x, y, z) are assumed to be in a right-handed
    Cartesian system. The spherical coordinates are returned in the order
    (r, phi, theta). The angles theta and phi are in radians.
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    # calculate the inclination - polar angle
    theta = np.arccos(z / r)

    # Calculate the azimuthal angle ensuring that phi is within [0, 2π)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    # if inclination is 0 or 180 set azimuth to 0
    phi[np.isclose(theta, 0)] = 0
    phi[np.isclose(theta, np.deg2rad(180))] = 0

    return r, phi, theta


def regular_S2_grid(n_squared: int = 144) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a regular grid on the unit sphere S2.

    The grid consists of evenly spaced azimuthal angles
    (longitude) and polar angles (colatitude). This is useful
    for visualizing or sampling the surface of a sphere.

    Parameters
    ----------
    n_squared : int, optional
        The number of grid points for both azimuthal and
        polar angles. Must be a positive integer.
        Default is 144.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two 2D arrays:
        - The first array contains the polar angles (colatitude).
        - The second array contains the azimuthal angles (longitude).

    Raises
    ------
    ValueError
        If "n_squared" is not a positive integer.
    """

    # Input validation
    if not isinstance(n_squared, int) or n_squared <= 0:
        raise ValueError("`n_squared` must be a positive integer.")

    azimuths = np.linspace(
        0, 2 * np.pi, n_squared, endpoint=True
    )  # Longitude (0 to 2*pi)
    polar = np.arccos(1 - 2 * np.linspace(0, 1, n_squared))  # Colatitude (0 to pi)

    return np.meshgrid(polar, azimuths)


def gauss_legendre_S2_meshgrid(
    n_polar: int = 144,
    n_azimuth: int | None = None,
    *,
    degrees: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tensor-product spherical grid with improved packing over a
    regular (phi, theta) grid by using Gauss–Legendre nodes in
    cos(polar) and uniform azimuth spacing.

    This produces a true tensor-product meshgrid:
    polar_grid, azimuth_grid = np.meshgrid(polar_1d, azimuth_1d, indexing="xy")

    Parameters
    ----------
    n_polar : int, optional
        Number of polar (colatitude) samples in [0, pi]. Default is 144.
    n_azimuth : int or None, optional
        Number of azimuth samples in [0, 2*pi). If None, uses `n_polar` to
        emulate the square output of `regular_S2_grid`. Default is None.
    degrees : bool, optional
        If True, return angles in degrees. Default is False.

    Returns
    -------
    polar_grid : np.ndarray
        2D array of polar (colatitude) angles, shape (n_azimuth, n_polar).
    azimuth_grid : np.ndarray
        2D array of azimuth angles, shape (n_azimuth, n_polar).

    Raises
    ------
    ValueError
        If `n_polar` or `n_azimuth` are not positive integers.

    Notes
    -----
    - Polar angle is colatitude measured from +z: 0 at north pole, pi at south pole.
    - Gauss–Legendre nodes are in μ = cos(polar) and exclude the poles. This avoids
      pathological clustering right at the poles and often improves packing for
      rectangular grids.
    """

    if not isinstance(n_polar, int) or n_polar <= 0:
        raise ValueError("`n_polar` must be a positive integer.")

    if n_azimuth is None:
        n_azimuth = n_polar
    if not isinstance(n_azimuth, int) or n_azimuth <= 0:
        raise ValueError("`n_azimuth` must be a positive integer.")

    # Gauss–Legendre nodes in μ ∈ (-1, 1). (Roots of Legendre polynomial.)
    mu, _w = np.polynomial.legendre.leggauss(n_polar)  # mu sorted ascending (-1..1)
    # Map to polar colatitude in [0, pi]. Flip so polar increases 0..pi.
    polar_1d = np.arccos(mu[::-1])

    azimuth_1d = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)

    polar_grid, azimuth_grid = np.meshgrid(polar_1d, azimuth_1d, indexing="xy")

    if degrees:
        polar_grid = np.rad2deg(polar_grid)
        azimuth_grid = np.rad2deg(azimuth_grid) % 360.0

    return polar_grid, azimuth_grid

# End of function definitions

if __name__ == "__main__":
    pass
else:
    print("reference_frame module imported.")

# End of file
