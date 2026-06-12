# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: crystallography.py                                                #
# Description: Implements crystallographic calculations for FTIR analysis.    #
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
from itertools import product
from typing import Tuple, List
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, differential_evolution

# Function definitions

def convert_Euler_to_Asimow_angles(
    euler_angles_deg: np.ndarray,
    degrees: bool = True
) -> np.ndarray:
    """
    Computes the Asimow angles (phi, theta) from Euler angles.

    Calculates:
    1. phi: The angle between the crystal c-axis and the polarizer vector E
       (E=[1,0,0] in the reference frame). Range: [0, 180] degrees.
    2. theta: The angle from the crystal a-axis to E' (the projection of E
       onto the crystal a-b plane), measured counter-clockwise when
       viewing the a-b plane from the positive c-axis direction.
       Range: [0, 360) degrees, i.e. [0, 2*pi) radians.

    Parameters
    ----------
    euler_angles : array_like, shape (3,) or (n,3)
        Euler angles (phi1, Phi, phi2) in Bunge zxz convention,
        in degrees.
    degrees : bool, default True
        If True, return Asimow angles are in degrees,
        otherwise radians.

    Returns
    -------
    np.ndarray, shape (2,) or (n,2)
        Array of Asimow angles [phi, theta].

    Notes
    -----
    - It calculates phi as the standard angle between vectors [0, pi].
    - Theta is calculated using atan2 relative to the crystal c-axis, giving a
      directed angle [0, 2pi).
    - Assumes crystal axes a, b, c align with x, y, z in the zero-rotation state.
    - Assumes E vector (reference for angles) is fixed to microscope X-axis [1,0,0].
    - Handles the degenerate case where E || c (E' is zero) by setting theta = 0.
    """
    
    # Ensure input is a Numpy array (should be but belt-and-braces)
    euler_angles = np.asarray(euler_angles_deg)

    # Input validation
    if euler_angles.ndim == 1:
        euler_angles = euler_angles[np.newaxis, :]
    if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
        raise ValueError("Input must be of shape (3,) or (n, 3).")

    # Initialize output array
    n_orientations = euler_angles.shape[0]
    Asimow_angles = np.zeros((n_orientations, 2))
    eps = 1e-9 # Tolerance for floating point checks

    # set reference crystal axes and E vector in Cartesian frame (unit vectors)
    a_axis = np.array([1., 0., 0.])
    c_axis = np.array([0., 0., 1.])
    E_vector = np.array([1., 0., 0.])

    # Iterate over orientations and compute Asimow's angles
    for i, orientation_deg in enumerate(euler_angles):

        # Create rotation object for the i-th orientation
        rotation = R.from_euler('zxz', orientation_deg, degrees=True)

        # estimate orientation of crystal axes after rotation in ref. frame 
        a_axis = np.array([1, 0, 0]) @ rotation.as_matrix().T
        c_axis = np.array([0, 0, 1]) @ rotation.as_matrix().T

        # Phi: unsigned angle between vector E and c
        E_norm = np.linalg.norm(E_vector)
        c_axis_norm = np.linalg.norm(c_axis)
        cos_E2c = np.dot(E_vector, c_axis) / (E_norm * c_axis_norm)
        cos_E2c = np.clip(cos_E2c, -1.0, 1.0)  # Ensure valid range for arccos
        phi = np.arccos(cos_E2c)               # Angle in [0, pi)
        
        # Theta: signed angle from a to Eprime around c-axis
        # Project E onto the a-b plane (plane normal to c-axis)
        Eprime = _project_vector_onto_plane(E_vector, c_axis)
        Eprime_norm = np.linalg.norm(Eprime)

        if Eprime_norm < eps:  # Handle edge case: E parallel to c (E' is zero vector)
            theta = 0.0
        else:
            Eprime = Eprime / Eprime_norm
            theta = _calculate_counterclockwise_angle(a_axis, Eprime, normal=c_axis)
        
        if degrees:
            theta = np.degrees(theta)
            phi   = np.degrees(phi)
        Asimow_angles[i] = [np.around(phi, 3), np.around(theta, 3)]

    return Asimow_angles[0] if Asimow_angles.shape[0]==1 else Asimow_angles


def calc_misorientation(
    euler1_deg: np.ndarray,
    euler2_deg: np.ndarray,
    precision: int = 3,
) -> float:
    """
    Calculate the misorientation angle between two crystals.

    Parameters
    ----------
    euler1_deg : array_like (3,)
        Euler angles representing the orientation of the first
        crystal in degrees (Bunge xzx convention).
    euler2_deg : array_like (3,)
        Euler angles representing the orientation of the second
        crystal in degrees (Bunge xzx convention).
    precision : int, optional
        the precision of the angle, defaults to 3

    Note
    ----
    THE METHOD IS SAFE ONLY WHEN IT DEALS WITH SMALL MISORIENTATION
    ANGLES! This method assumes that the calculated rotation matrix
    represent a proper rotation (i.e. a rotation with no reflection
    or inversion). This general method should work for any crystal
    symmetry but for large misorientation angles it couldn't give
    the lowest angle, i.e. the disorientation, but any other
    misorientation angle as it does not take into account the
    particular symmetry of the crystal.

    Returns
    -------
    float
        The misorientation angle between the two crystalS
        in degrees.
    """

    # Convert Euler angles to rotation matrices
    R1 = R.from_euler("zxz", euler1_deg, degrees=True)
    R2 = R.from_euler("zxz", euler2_deg, degrees=True)

    # calc rotation
    rotation = R1.inv() * R2

    # Get the magnitude(s) of the rotation(s): angle(s) in radians
    misorientation_rad = rotation.magnitude()
    # misorientation_rad = np.arccos((np.trace(rotation.as_matrix()) - 1) / 2)

    return np.around(np.rad2deg(misorientation_rad), precision)


def calc_disorientation(
    euler1_deg: np.ndarray,
    euler2_deg: np.ndarray,
    all=False
) -> float | np.ndarray:
    """
    Calculate the disorientation angle, i.e. minimum
    misorientation angle between two orthorhombic crystals
    or all misorientation angles.

    Parameters
    ----------
    euler1_deg : array_like (3,)
        Euler angles representing the orientation of the
        first crystal in degrees (Bunge convention).
    euler2_deg : array_like (3,)
        Euler angles representing the orientation of the
        second crystal in degrees (Bunge convention).
    all : bool, defaults to False
        if True, it returs all the misorientation angles

    Returns
    -------
    float or ndarray
        The disorientation or misorientation(s) angle(s)
        between the two orthorhombic crystals in degrees.
    """

    # covert euler 1 to rotation matrix
    R1 = R.from_euler("zxz", euler1_deg, degrees=True)

    # symmetrize 2nd orientation
    equivalent_orientations = symmetrise_orthorhombic_Euler(
        euler2_deg, input_unit="degrees", output_unit="degrees"
    )

    misorientations = np.empty(len(equivalent_orientations))

    for index, orientation in enumerate(equivalent_orientations):
        R2 = R.from_euler("zxz", orientation, degrees=True)
        rotation = R1.inv() * R2
        misorientation_rad = rotation.magnitude()
        misorientations[index] = misorientation_rad

    if all:
        return np.around(np.rad2deg(misorientations), 3)
    else:
        return np.around(np.rad2deg(misorientations.min()), 3)

# ============================================================================ #
# SYMMETRISE                                                                   #
# ============================================================================ #


def symmetrise_orthorhombic_Euler(
    Euler_angles_deg: np.ndarray,
    output_unit: str = "degrees",
    canonical: bool = False,
) -> np.ndarray:
    """
    Apply symmetry operations for an orthorhombic crystal
    (point group mmm) to compute all 8 equivalent orientations
    in Bunge Euler angles.

    The 8 operations come from combining 180° rotations
    about X, Y, Z:
        E,   C2z,   C2y,   C2x,
        i,   C2z·i, C2y·i, C2x·i   (≡ σxy, σxz, σyz planes)

    In Bunge convention (ZXZ extrinsic), acting on (φ1, Φ, φ2):
        E          → ( φ1,     Φ,      φ2    )
        C2z        → ( φ1+π,   Φ,      φ2    )
        C2y        → ( φ1+π,   π−Φ,    π-φ2  )
        C2x        → ( φ1,     π−Φ,    π-φ2  )
        i·E        → ( φ1+π,   π−Φ,    φ2+π  )   [= inversion]
        i·C2z      → ( φ1,     π−Φ,    φ2+π  )
        i·C2y      → ( φ1,     Φ,      φ2+π  )
        i·C2x      → ( φ1+π,   Φ,      φ2+π  )

    Parameters
    ----------
    Euler_angles_deg : array_like, shape (3,)
        Bunge Euler angles (φ1, Φ, φ2) in degrees.
    output_unit : str, optional
        'degrees' (default) or 'radians'.
    canonical : bool, optional
        If True, return only the orientation satisfying
        φ1 ∈ [0,360), Φ ∈ [0,90], φ2 ∈ [0,180].
        If False (default), return all 8 equivalent orientations.

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        Equivalent orientations wrapped to [0, 2π) (or [0°, 360°)).
        In canonical mode, shape is (1, 3) when a match is found.

    Raises
    ------
    ValueError
        If output_unit is not 'degrees' or 'radians'.
    """

    if output_unit not in ("degrees", "radians"):
        raise ValueError(
            f"output_unit must be 'degrees' or 'radians', got {output_unit!r}"
        )

    phi1, Phi, phi2 = np.radians(Euler_angles_deg)
    pi = np.pi

    # All 8 symmetry operations of mmm in Bunge Euler angles
    equivalent_orientations = np.array(
        [
            (phi1, Phi, phi2),                 # E
            (phi1 + pi, Phi, phi2),            # C2z
            (phi1 + pi, pi - Phi, pi - phi2),  # C2y
            (phi1, pi - Phi, pi - phi2),       # C2x
            (phi1 + pi, pi - Phi, phi2 + pi),  # i
            (phi1, pi - Phi, phi2 + pi),       # i·C2z
            (phi1, Phi, phi2 + pi),            # i·C2y
            (phi1 + pi, Phi, phi2 + pi),       # i·C2x
        ]
    )

    # Wrap all angles to [0, 2π)
    equivalent_orientations = equivalent_orientations % (2 * pi)

    if output_unit == "degrees":
        equivalent_orientations = np.degrees(equivalent_orientations)

    if canonical:
        limits = (
            np.array([360.0, 90.0, 180.0])
            if output_unit == "degrees"
            else np.array([2 * pi, pi / 2, pi])
        )
        tol = 1e-9
        mask = np.all(equivalent_orientations <= limits + tol, axis=1)
        matches = equivalent_orientations[mask]
        if matches.shape[0] == 0:
            raise RuntimeError(
                "No canonical orientation found — check input or symmetry operations."
            )
        return matches

    return equivalent_orientations


def symmetrise_orthorhombic_Asimow_angles(
    theta_deg: float,
    phi_deg: float,
) -> np.ndarray:
    """
    Apply symmetry operations for an orthorhombic crystal
    to estimate all the equivalent Asimow angles (theta, phi).

    The orthorhombic point group (mmm) has 8 symmetry operations.
    These operations result in up to 8 unique equivalent directions.

    Parameters:
    -----------
    theta_deg : float between 0 and 360
        theta angle in degrees
    phi_deg : float between 0 and 180
        phi angle in degrees

    Returns:
    --------
    np.ndarray
        An array of (theta_deg, phi_deg) representing all
        symmetrically equivalent directions. The array is sorted
        for consistent output.
    """
    if not (0 <= phi_deg <= 180):
        raise ValueError("phi_deg must be between 0 and 180 degrees.")
    if not (0 <= theta_deg <= 360):
        raise ValueError("theta_deg must be between 0 and 360 degrees.")

    # Transformations for theta angle
    if theta_deg > 180.0:
        theta_deg2 = 360.0 - theta_deg
        theta_deg3 = theta_deg - 180.0
        theta_deg4 = 360.0 - theta_deg3
    else:
        theta_deg2 = 360.0 - theta_deg
        theta_deg3 = 180.0 - theta_deg
        theta_deg4 = 360.0 - theta_deg3

    # Transformations for phi angle
    phi_deg2 = 180.0 - phi_deg

    # Use set to handle potential duplicates
    thetas = set([theta_deg, theta_deg2, theta_deg3, theta_deg4])
    phis = set([phi_deg, phi_deg2])

    # return all combinations
    return np.array(list(product(sorted(thetas), sorted(phis))))


def canonical_orthorhombic_Asimow_angles(
    theta_deg: float, 
    phi_deg: float, 
    show: bool = True
) -> np.ndarray:
    """
    Map an Asimow angle pair (theta, phi) to a unique representative
    under orthorhombic crystal symmetry (point group mmm).

    This function uses the symmetry-equivalent directions generated by
    `symmetrise_orthorhombic_Asimow_angles` and selects a canonical
    representative according to the following rule:

    1. Generate all symmetry-equivalent (theta, phi) pairs.
    2. Normalize theta into the range [0, 360) degrees.
    3. Select the pair with the smallest phi (polar angle).
    4. If several pairs share the same phi, select the one with the
       smallest theta.

    With the current definition of orthorhombic symmetry in
    `symmetrise_orthorhombic_Asimow_angles`, the canonical angles
    always lie in the first octant of the sphere:

    - 0° ≤ theta_canon ≤ 90°
    - 0° ≤ phi_canon ≤ 90°

    Parameters
    ----------
    theta_deg : float
        Azimuthal angle theta in degrees. Expected range: 0° ≤ theta ≤ 360°.
    phi_deg : float
        Polar angle phi in degrees. Expected range: 0° ≤ phi ≤ 180°.

    Returns
    -------
    numpy.ndarray
        A 1D array of shape (2,) containing
        [theta_canon_deg, phi_canon_deg].

    Raises
    ------
    ValueError
        If the input angles are outside their expected ranges.
    """
    # Symmetrise
    equivalents = symmetrise_orthorhombic_Asimow_angles(theta_deg, phi_deg).astype(
        float
    )

    # Reorder columns: now col 0 = phi, col 1 = theta
    equivalents = equivalents[:, [1, 0]]

    # Normalize theta to [0, 360) to avoid 360° vs 0° inconsistencies
    equivalents[:, 0] = np.mod(equivalents[:, 0], 360.0)

    # Sort primarily by phi (column 0), secondarily by theta (column 1)
    # np.lexsort uses the last key as the primary key
    sort_indices = np.lexsort((equivalents[:, 1], equivalents[:, 0]))

    # First entry after sorting is the canonical representative
    canonical_pair = equivalents[sort_indices[0]]

    if show:
        print(f"Canoninal angles:\n  phi = {canonical_pair[0]:.2f}\ntheta = {canonical_pair[1]:.2f}")

    return canonical_pair


def minimum_angular_distance(
    direction1_rad: np.ndarray,
    direction2_rad: np.ndarray,
) -> float:
    """
    Computes the minimum angular distance between
    two directions on a sphere.

    This function calculates the great-circle distance
    between a single pair of points on a unit sphere
    using the haversine formula which is numerically
    stable for small distances.

    Parameters
    ----------
    direction1 : numpy.ndarray
        A NumPy array of shape (2,) containing the
        [phi, theta] angles in radians for the first
        direction.
        - phi: polar angle from the z-axis, 0 to pi.
        - theta: azimuthal angle in the xy-plane, 0 to 2*pi.
    direction2 : numpy.ndarray
        A NumPy array of shape (2,) with the [phi, theta]
        angles in radians for the second direction.

    Returns
    -------
    float
        The angular distance in radians.
    """
    # Unpack the polar (phi) and azimuthal (theta) angles
    phi1, theta1 = direction1_rad
    phi2, theta2 = direction2_rad

    # Differences in angles
    delta_phi = phi2 - phi1
    delta_theta = theta2 - theta1

    # Haversine formula implementation
    a = (np.sin(delta_phi / 2.0)**2 +
         np.sin(phi1) * np.sin(phi2) * np.sin(delta_theta / 2.0)**2)
    a = np.clip(a, 0.0, 1.0)
    
    distance = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return distance


# ============================================================================ #
# AUXILIARY FUNCTIONS. Not intended for direct user access.                    #
# ============================================================================ #

def _project_vector_onto_plane(
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


def _calculate_counterclockwise_angle(
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


def _euler_Asimow_misfit(euler_angles_deg, target_Asimow_deg):
    """
    Convert Euler to Asimow angles and then compute misfit
    (squared spherical angular separation) between the unit
    vectors represented by the estimated and target angle
    pairs.

    Parameters
    ----------
    euler_angles_deg : ndarray of shape (3,)
        Euler angles [phi1, Phi, phi2] in degrees.
    target_Asimow : ndarray of shape (2,)
        Target Asimow angles [phi, theta] in degrees.

    Returns
    -------
    float
        Sum of squared differences between computed and target Asimow angles.
    """

    # convert target to radians
    target_Asimow = np.radians(target_Asimow_deg)

    # Convert Euler to Asimow (output in radians!)
    computed_Asimow = convert_Euler_to_Asimow_angles(euler_angles_deg, degrees=False)

    # Ensure shape (2,) for consistency
    computed_Asimow = np.asarray(computed_Asimow).ravel()

    # compute misfit: minimum angular distance on a sphere
    # between two directions 
    distance = minimum_angular_distance(computed_Asimow, target_Asimow)
    
    return distance ** 2

# End of function definitions

if __name__ == "__main__":
    pass
else:
    print("crystallography module imported.")

# End of file
