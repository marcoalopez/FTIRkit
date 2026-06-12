# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: geometry.py                                                       #
# Description: Implements tools for dealing with reference frames, coordinate #
# conversions, spherical grids, and shared 3D vector/rotation operations.     #
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
from scipy.spatial.transform import Rotation as R

# Function definitions

# ============================================================================ #
# COORDINATE CONVERSIONS                                                       #
# ============================================================================ #

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


# ============================================================================ #
# SPHERICAL GRIDS                                                              #
# ============================================================================ #

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


# ============================================================================ #
# VECTOR AND ROTATION OPERATIONS                                               #
# ============================================================================ #

def rotate(
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
    x, y, z = rotate(coordinates=(x, y, z), euler_deg=(30, 0, 40))
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


def project_vector_onto_plane(
    vector: np.ndarray,
    plane_normal: np.ndarray
) -> np.ndarray:
    """
    Calculates the orthogonal projection of a vector onto a plane
    defined by its normal vector.

    Parameters
    ----------
    vector : np.ndarray, shape (3,)
        The vector to project.
    plane_normal : np.ndarray, shape (3,)
        The normal vector to the plane. Does not need to be normalized.

    Returns
    -------
    np.ndarray, shape (3,)
        The projection of the vector onto the plane. Returns a zero
        vector if the plane_normal is a zero vector.

    """

    # Ensure vectors are Numpy arrays (should be but belt-and-braces)
    vector = np.asarray(vector)
    plane_normal = np.asarray(plane_normal)

    # Check if the input vectors are valid.
    if vector.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError("Input vectors must be 3D.")

    # check for degenerate planes
    norm_sq = np.dot(plane_normal, plane_normal)
    if norm_sq < 1e-12:
        raise ValueError(
            "Plane normals are collinear or zero. Projection is ill-defined."
        )

    # Calculate the projection of the vector onto the normal vector
    proj_onto_normal = (np.dot(vector, plane_normal) / norm_sq) * plane_normal

    # Subtract the normal component to get the projection onto the plane
    proj_onto_plane = vector - proj_onto_normal

    return proj_onto_plane


def counterclockwise_angle(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
    normal: np.ndarray,
) -> float:
    """
    Calculates a signed angle from vector_1 to vector_2 in
    the plane perpendicular to "normal".

    Measured counterclockwise about 'normal' using
    right-hand rule.

    Parameters
    ----------
    vector_1, vector_2 : np.ndarray
        Vectors in 3D, represented as a NumPy array of shape (3,).
    normal : array_like, shape (3,)
        Unit or non-unit normal defining the rotation axis.

    Returns
    -------
    angle : float
        Angle in radians in [0,2π)

    Raises
    ------
    ValueError
        If either of the input vectors is not a 3D vector.
    """

    # Ensure vectors are Numpy arrays (should be but belt-and-braces)
    vector_1 = np.asarray(vector_1, dtype=float)
    vector_2 = np.asarray(vector_2, dtype=float)
    normal = np.asarray(normal, dtype=float)

    # Check if the input vectors are valid.
    if vector_1.shape != (3,) or vector_2.shape != (3,) or normal.shape != (3,):
        raise ValueError("All inputs must be must be 3D vectors.")

    normal = normal / np.linalg.norm(normal)

    # project vector1 and vector2 onto plane ⟂n
    vector1 = vector_1 - np.dot(vector_1, normal) * normal
    vector2 = vector_2 - np.dot(vector_2, normal) * normal
    vector1_norm = np.linalg.norm(vector1)
    vector2_norm = np.linalg.norm(vector2)

    # check
    if vector1_norm < 1e-12 or vector2_norm < 1e-12:
        raise ValueError("Input vectors must not be parallel to normal.")

    # Normalize for angle calculation
    vector1 = vector1 / vector1_norm
    vector2 = vector2 / vector2_norm

    # Calculate the cosine of the angle ensuring a valid range
    cos_angle = np.clip(np.dot(vector1, vector2), -1.0, 1.0)

    # Calculate the sine of the angle
    sin_angle = np.dot(normal, np.cross(vector1, vector2))

   # Use arctan2 to return angles in [0,2π)
    angle = np.arctan2(sin_angle, cos_angle)

    # Determine the direction of the angle
    if angle < 0:
        angle += 2 * np.pi

    return angle


def _axis_angle_rotation(
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    Return the 3x3 rotation matrix for a right-hand
    rotation of 'angle' radians about the (unit) 'axis'.
    """
    axis = axis / np.linalg.norm(axis)
    return R.from_rotvec(axis * angle).as_matrix()

# End of function definitions

# End of file
