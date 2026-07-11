# =========================================================================== #
# FTIRkit: A Python package to estimate crystal orientation and synthesis of  #
# principal axis spectra using polarized μ-FTIR data.                         #
#                                                                             #
# SPDX-License-Identifier: Apache-2.0                                         #
# Copyright (c) 2025 Marco A. Lopez-Sanchez. All rights reserved.             #
#                                                                             #
# Author: Marco A. Lopez-Sanchez                                              #
# ORCID: http://orcid.org/0000-0002-0261-9267                                 #
# Website: https://marcoalopez.github.io/FTIRkit /                            #
# Repository: https://github.com/marcoalopez/FTIRkit                          #
# =========================================================================== #
"""
FTIRkit: A Python package to estimate crystal orientation and synthesis
of principal axis spectra using polarized μ-FTIR data.
"""

__version__ = "0.1.0"

# Re-export the most commonly used functions at the package level.
# Heavier modules (plots, synthetic) are imported on demand, e.g.
# `from ftirkit import plots`.
from .transmittance_model import calc_transmittance
from .orientation.lambda_method import (
    find_orientation_based_on_lambda,
    find_orientation_based_on_multiple_lambdas,
)
from .orientation.spectrum_method import find_orientation_based_on_spectrum
from .principal_spectra import synthesize_principal_spectra

__all__ = [
    "calc_transmittance",
    "find_orientation_based_on_lambda",
    "find_orientation_based_on_multiple_lambdas",
    "find_orientation_based_on_spectrum",
    "synthesize_principal_spectra",
    "__version__",
]
