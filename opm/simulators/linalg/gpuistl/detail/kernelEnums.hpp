/*
  Copyright 2024 SINTEF AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_GPUISTL_PRECONDITIONER_STORAGE_OPTION_HPP
#define OPM_GPUISTL_PRECONDITIONER_STORAGE_OPTION_HPP

#include <opm/common/ErrorMacros.hpp>

/*
    This file is here to organize a growing amount of different mixed precision options for the preconditioners.
    TODO: if this file should be merged then rename to PrecisionEnums maybe?
*/

namespace Opm::gpuistl {
    enum MixedPrecisionScheme {
        DEFAULT,
        STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_FLOAT,
        STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_DOUBLE,
        STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_FLOAT,
        STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_DOUBLE
    };

    enum Precision {
        HALF,
        FLOAT,
        DOUBLE
    };

/**
 * @brief Checks if the given scheme is a valid MixedPrecisionScheme.
 *
 * @param scheme The integer representation of the MixedPrecisionScheme.
 * @return true if the scheme is valid, false otherwise.
 */
    inline bool isValidMixedPrecisionScheme(int scheme) {
        switch (scheme) {
            case DEFAULT:
            case STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_FLOAT:
            case STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_DOUBLE:
            case STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_FLOAT:
            case STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_DOUBLE:
                return true;
            default:
                return false;
        }
    }

/**
 * @brief Gets the precision type for the diagonal elements based on the given MixedPrecisionScheme.
 *
 * @param scheme The MixedPrecisionScheme to evaluate.
 * @return The Precision type for the diagonal elements.
 */
    inline Precision getDiagonalPrecision(MixedPrecisionScheme scheme) {
        switch (scheme) {
            case DEFAULT:
            case STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_DOUBLE:
            case STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_DOUBLE:
                return DOUBLE;
            case STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_FLOAT:
            case STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_FLOAT:
                return FLOAT;
            default:
                OPM_THROW(std::runtime_error, "Invalid mixed precision scheme provided.");
        }
    }

/**
 * @brief Gets the precision type for the off-diagonal elements based on the given MixedPrecisionScheme.
 *
 * @param scheme The MixedPrecisionScheme to evaluate.
 * @return The Precision type for the off-diagonal elements.
 */
    inline Precision getOffDiagonalPrecision(MixedPrecisionScheme scheme) {
        switch (scheme) {
            case DEFAULT:
                return DOUBLE;
            case STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_FLOAT:
            case STORE_OFF_DIAGS_IN_FLOAT_DIAGONAL_AS_DOUBLE:
                return FLOAT;
            case STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_DOUBLE:
            case STORE_OFF_DIAGS_IN_HALF_DIAGONAL_AS_FLOAT:
                return HALF;
            default:
                OPM_THROW(std::runtime_error, "Invalid mixed precision scheme provided.");
        }
    }
}

#endif // OPM_GPUISTL_PRECONDITIONER_STORAGE_OPTION_HPP
