#ifndef PFAFFIAN_H
#define PFAFFIAN_H

#include "matrix.hpp"

/**
 * Calculates the pfaffian of the input matrix.
 *
 * @param matrix The input matrix.
 * @returns The pfaffian.
 */
template <typename TScalar>
extern TScalar pfaffian_cpp(Matrix<TScalar> &matrix_in);

#endif