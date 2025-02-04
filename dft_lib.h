#ifndef DFT_LIB_H
#define DFT_LIB_H

#include <petsc.h>

/*
 * compute_dft: Computes the forward DFT of the vector x.
 *
 * Inputs:
 *   x          - PETSc vector containing the input signal (global length N)
 *   N          - Global length of the signal
 *   comm       - MPI communicator (typically PETSC_COMM_WORLD)
 *
 * Outputs (allocated on rank 0):
 *   dft_real   - Pointer to an array of length N with the real parts of the DFT
 *   dft_imag   - Pointer to an array of length N with the imaginary parts of the DFT
 *
 * The caller on rank 0 is responsible for freeing the returned arrays.
 */
PetscErrorCode compute_dft(Vec x, PetscInt N, double **dft_real, double **dft_imag, MPI_Comm comm);

/*
 * compute_idft: Computes the inverse DFT given the frequency-domain data.
 *
 * Inputs:
 *   dft_real   - Array of length N containing the real parts of the DFT (on rank 0)
 *   dft_imag   - Array of length N containing the imaginary parts of the DFT (on rank 0)
 *   N          - Global length of the signal
 *   comm       - MPI communicator
 *
 * Outputs (allocated on rank 0):
 *   idft_real  - Pointer to an array of length N with the real parts of the inverse DFT
 *   idft_imag  - Pointer to an array of length N with the imaginary parts of the inverse DFT
 *
 * The inverse is computed as:
 *   x[n] = (1/N) * sum_{k=0}^{N-1} [X[k] * exp(2*pi*i*k*n/N)]
 *
 * The caller on rank 0 is responsible for freeing the returned arrays.
 */
PetscErrorCode compute_idft(double *dft_real, double *dft_imag, PetscInt N,
                           double **idft_real, double **idft_imag, MPI_Comm comm);

#endif