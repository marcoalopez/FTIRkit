# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# Filename: plots.py                                                          #
# Description: Implements custom plots for the FTIRkit                        #
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
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
from scipy.spatial.transform import Rotation as R

from .geometry import project_vector_onto_plane, counterclockwise_angle

# Function definitions

def plot_crystal_axes(euler_angles_deg):
    # Define the original Cartesian coordinate axes (x, y, z)
    origin = np.array([0, 0, 0])
    x_axis = np.array([1.5, 0, 0])  # 1.5 the length
    y_axis = np.array([0, 1.5, 0])
    z_axis = np.array([0, 0, 1.5])
    x_axis_neg = np.array([-1.5, 0, 0])
    y_axis_neg = np.array([0, -1.5, 0])
    z_axis_neg = np.array([0, 0, -1.5])
    E_vector = np.array([1.5, 0, 0])

    # Rotate crystal axes to orientation (euler angles, Bunge)
    rotation = R.from_euler("zxz", euler_angles_deg, degrees=True)
    a_axis = np.array([1, 0, 0]) @ rotation.as_matrix().T
    b_axis = np.array([0, 1, 0]) @ rotation.as_matrix().T
    c_axis = np.array([0, 0, 1]) @ rotation.as_matrix().T

    # Estimate the position of vector E prime (projection of E onto the
    # a-b plane, whose normal is the c-axis)
    Eprime_vector = project_vector_onto_plane(E_vector, c_axis)

    # initialize plot
    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": "3d"}, constrained_layout=True
    )

    # Plot the Cartesian coordinate system
    ax.quiver(*origin, *x_axis, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *y_axis, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *z_axis, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *x_axis_neg, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *y_axis_neg, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *z_axis_neg, color="grey", alpha=0.5, arrow_length_ratio=0.0)

    # Plot crystal axis
    ax.quiver(
        *origin,
        *a_axis,
        color="#FF6666",
        zorder=0,
        linewidths=2,
        arrow_length_ratio=0.1,
    )
    ax.quiver(
        *origin,
        *b_axis,
        color="#005533",
        zorder=0,
        linewidths=2,
        arrow_length_ratio=0.1,
    )
    ax.quiver(
        *origin,
        *c_axis,
        color="#1199EE",
        zorder=0,
        linewidths=2,
        arrow_length_ratio=0.1,
    )

    # Plot E vector and E' projection
    ax.quiver(
        *origin,
        *E_vector,
        color="black",
        zorder=0,
        length=0.5,
        arrow_length_ratio=0.1
    )
    ax.quiver(
        *origin,
        *Eprime_vector,
        color="dimgrey",
        zorder=0,
        length=0.5,
        arrow_length_ratio=0.1,
        linestyle="--",
    )

    # Plot arcs between c-axis to E-vector and a-axis to E'-vector
    arc_points_E2c = _generate_arc_points(E_vector, c_axis)
    arc_points_a2Ep = _generate_arc_points(a_axis, Eprime_vector)
    ax.plot(
        arc_points_E2c[:, 0],
        arc_points_E2c[:, 1],
        arc_points_E2c[:, 2],
        color="black",
        linestyle="-",
    )
    ax.plot(
        arc_points_a2Ep[:, 0],
        arc_points_a2Ep[:, 1],
        arc_points_a2Ep[:, 2],
        color="dimgray",
        linestyle="-",
    )

    # Project a plane containing the axes a and b
    plane = _generate_plane_points(a_axis, b_axis)
    ax.plot_surface(*plane, color="paleturquoise", alpha=0.4)

    # Adding labels to the original axes
    ax.text(
        *x_axis,
        "x",
        color="k",
        alpha=0.5,
        fontsize=16,
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.text(*y_axis, "y", color="k", alpha=0.5, fontsize=16)
    ax.text(*z_axis, "z", color="k", alpha=0.5, fontsize=16, verticalalignment="bottom")

    # Adding labels to the rotated axes
    ax.text(
        *a_axis,
        "a",
        color="#FF6666",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    ax.text(
        *b_axis,
        "b",
        color="#005533",
        fontsize=14,
        horizontalalignment="right",
        verticalalignment="bottom",
    )
    ax.text(
        *c_axis,
        "c",
        color="#1199EE",
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    ax.text(
        0.85,
        0,
        0,
        "E",
        color="black",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.text(
        0.85,
        0.25,
        0,
        "E'",
        color="dimgray",
        fontsize=14,
        horizontalalignment="left",
        verticalalignment="top",
    )

    # Remove the axes panes and grid
    ax.grid(False)  # Turn off the grid
    ax.axis("off")  # Turn off the axis lines

    # Set the plot limits and labels
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_title("Crystal axes")
    ax.view_init(elev=20, azim=30)

    return fig, ax


def plot_transmitance_envelope(coordinates_xyz, T_values):

    # Define the original Cartesian coordinate axes (x, y, z)
    # TODO: calc optimal length based on range of T_values
    origin = np.array([0, 0, 0])
    x_axis = np.array([1.5, 0, 0])  # 1.5 the length
    y_axis = np.array([0, 1.5, 0])  # 1.5 the length
    z_axis = np.array([0, 0, 1.5])  # 1.5 the length
    x_axis_neg = np.array([-1.5, 0, 0])  # 1.5 the length
    y_axis_neg = np.array([0, -1.5, 0])  # 1.5 the length
    z_axis_neg = np.array([0, 0, -1.5])  # 1.5 the length

    # initialize plot
    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={"projection": "3d"},
        constrained_layout=True
    )

    # Plot the Cartesian coordinate system
    ax.quiver(*origin, *x_axis, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *y_axis, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *z_axis, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *x_axis_neg, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *y_axis_neg, color="grey", alpha=0.5, arrow_length_ratio=0.0)
    ax.quiver(*origin, *z_axis_neg, color="grey", alpha=0.5, arrow_length_ratio=0.0)

    # Adding labels to the original axes
    ax.text(
        *x_axis,
        "x",
        color="k",
        alpha=0.5,
        fontsize=16,
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.text(*y_axis, "y", color="k", alpha=0.5, fontsize=16)
    ax.text(*z_axis, "z", color="k", alpha=0.5, fontsize=16, verticalalignment="bottom")

    # normalize colors for T values to max and min values
    Tmax, Tmin = T_values.max(), T_values.min()
    Tcolors = (T_values - Tmin) / (Tmax - Tmin)

    # plot the envelope
    # TODO: add option to define the colormap in funtion parameters
    ls = LightSource(30, 20)
    im = ax.plot_surface(
        coordinates_xyz[0],
        coordinates_xyz[1],
        coordinates_xyz[2],
        rstride=1,
        cstride=1,
        facecolors=cm.Spectral_r(Tcolors),
        lightsource=ls,
        alpha=0.4,
    )

    # Remove the axes panes and grid
    ax.grid(False)  # Turn off the grid
    ax.axis("off")  # Turn off the axis lines

    # Set the plot limits and labels
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_title("transmittance envelope")
    ax.view_init(elev=20, azim=30)

    return fig, ax


def plot_XY_section(coordonates_xy):
    pass


# ============================================================================ #
# AUXILIARY FUNCTIONS                                                          #
# ============================================================================ #

def _generate_arc_points(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
    radius: float = 0.4,
    num_points: int = 50,
) -> np.ndarray:
    """
    Generates points along a counterclockwise arc between two
    vectors in 3D space. Uses SciPy's Rotation class for simpler
    rotation calculations.

    Parameters
    ----------
    vector_1 : np.ndarray
        The starting 3D vector, represented as a NumPy array of shape (3,).
    vector_2 : np.ndarray
        The ending 3D vector, represented as a NumPy array of shape (3,).
    radius : float, optional
        The radius of the arc, by default 0.4.
    num_points : int, optional
        The number of points used to define the arc, by default 50.

    Returns
    -------
    np.ndarray
        The points along the arc, represented as a NumPy array of shape (num_points, 3).

    Raises
    ------
    ValueError
        If either of the input vectors is not a 3D vector.
        If either of the input vectors is a zero vector.
    """

    # Check if the input vectors are valid.
    if vector_1.shape != (3,) or vector_2.shape != (3,):
        raise ValueError("Input vectors must be 3D vectors.")
    if np.all(vector_1 == 0) or np.all(vector_2 == 0):
        raise ValueError("Input vectors cannot be zero vectors.")

    # Normalize the vectors
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    # Create a rotation axis perpendicular to both vectors
    rotation_axis = np.cross(vector_1, vector_2)
    if np.linalg.norm(rotation_axis) < 1e-8:
        print("Some vectors are parallel or anti-parallel!")
        # Handle the case where vectors are collinear by choosing an arbitrary perpendicular axis
        rotation_axis = np.cross(
            vector_1, np.array([1, 0, 0])
        )  # Try cross product with [1, 0, 0]
        if np.linalg.norm(rotation_axis) < 1e-8:  # If still zero, try [0, 1, 0]
            rotation_axis = np.cross(vector_1, np.array([0, 1, 0]))
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Determine the direction of the angle under the right-hand rule convention
    # If the z-component of the cross product is negative, the angle is clockwise.
    if rotation_axis[2] < 0:
        rotation_axis = -rotation_axis

    # Calculate the angle between the vectors, measured counterclockwise
    # about the rotation axis
    angle = counterclockwise_angle(vector_1, vector_2, normal=rotation_axis)
    # print(np.rad2deg(angle))

    # Generate rotation angles from 0 to the calculated angle
    rotation_angles = np.linspace(0, angle, num_points)

    # Use SciPy's Rotation class to create rotations
    # np.newaxis adds an extra dimension to rotation_angles, making it a column vector
    # This is done because from_rotvec expects an array of shape (N, 3)
    rotations = R.from_rotvec(rotation_angles[:, np.newaxis] * rotation_axis)

    # Apply the rotations to vector_1
    arc_points = rotations.apply(vector_1)

    # Scale the points to the desired radius and append the origin
    arc_points = arc_points * radius
    # arc_points = np.vstack((arc_points, [0, 0, 0]))

    return arc_points


def _generate_plane_points(vector1, vector2, num_points=10):
    """_summary_

    Parameters
    ----------
    vector1 : _type_
        _description_
    vector2 : _type_
        _description_
    num_points : int, optional
        _description_, by default 10

    Returns
    -------
    _type_
        _description_
    """

    xx, yy = np.meshgrid(np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points))

    return (
        vector1[:, np.newaxis, np.newaxis] * xx
        + vector2[:, np.newaxis, np.newaxis] * yy
    )


# End of function definitions

# End of file
