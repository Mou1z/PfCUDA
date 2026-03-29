#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

using namespace Eigen;

/**
 * Calculates the Pfaffian of a skew-symmetric matrix A.
 * * @param A A reference to an Eigen matrix. Note: This algorithm modifies `A` in-place.
 * If you need to preserve the original matrix, pass a copy instead.
 * @return The Pfaffian (double).
 */
double pfaffian_optimized(Ref<MatrixXd> A) {
    const Index n = A.rows();
    
    // The Pfaffian of an odd-dimensional anti-symmetric matrix is always 0
    if (n % 2 != 0) return 0.0; 
    if (n == 0) return 1.0;
    
    double result = 1.0;

    for (Index i = 0, k = 0; k < n - 1; ++i, k += 2) {
        Index rem_cols = n - k - 1;
        
        // --- ROW UPDATE ---
        if (i > 0) {
            // Eigen::seq enables slicing without copying data
            auto seq_even = seq(0, k - 2, 2);
            auto seq_odd  = seq(1, k - 1, 2);
            auto seq_rem  = seq(k + 1, n - 1);
            
            // Lazy evaluation: these do not allocate new memory, they just map to A
            auto v_ik  = A(seq_even, k);
            auto w_ik  = A(seq_odd,  k);
            auto W_rem = A(seq_odd,  seq_rem);
            auto V_rem = A(seq_even, seq_rem);
            
            // .noalias() explicitly tells Eigen no memory overlaps occur, 
            // bypassing the creation of temporary objects for the result.
            A.block(k, k + 1, 1, rem_cols).noalias() += 
                (v_ik.transpose() * W_rem) - (w_ik.transpose() * V_rem);
        }

        // --- PIVOTING ---
        Index km;
        A.block(k, k + 1, 1, rem_cols).cwiseAbs().maxCoeff(&km);
        Index kp = k + 1 + km;

        if (kp != k + 1) {
            A.row(k + 1).swap(A.row(kp));
            A.col(k + 1).swap(A.col(kp));
            result = -result;
        }

        // --- EARLY EXIT OPTIMIZATION ---
        double ak0 = A(k, k + 1);
        if (std::abs(ak0) < 1e-15) {
            // If the max pivot is effectively zero, the matrix is singular.
            // Bailing out here avoids expensive division-by-zero (NaNs) and saves time.
            return 0.0; 
        }

        // --- COL UPDATE ---
        Index rem_cols_next = n - k - 2;
        if (rem_cols_next > 0) {
            if (i > 0) {
                auto seq_even = seq(0, k - 2, 2);
                auto seq_odd  = seq(1, k - 1, 2);
                auto seq_rem2 = seq(k + 2, n - 1);
                
                auto v_ik1  = A(seq_even, k + 1);
                auto w_ik1  = A(seq_odd,  k + 1);
                auto W_rem2 = A(seq_odd,  seq_rem2);
                auto V_rem2 = A(seq_even, seq_rem2);

                A.block(k + 1, k + 2, 1, rem_cols_next) = -A.block(k + 1, k + 2, 1, rem_cols_next);
                A.block(k + 1, k + 2, 1, rem_cols_next).noalias() -= 
                    (v_ik1.transpose() * W_rem2) - (w_ik1.transpose() * V_rem2);
            } else {
                A.block(k + 1, k + 2, 1, rem_cols_next) = -A.block(k + 1, k + 2, 1, rem_cols_next);
            }
            
            // Normalize remaining elements
            A.block(k, k + 2, 1, rem_cols_next) /= ak0;
        }
        
        result *= ak0;
    }

    return result;
}