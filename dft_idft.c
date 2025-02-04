#include <petsc.h>
#include "dft_lib.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    MPI_Comm comm = PETSC_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Get N from options (default: N = 1000)
    PetscInt N = 1000;
    ierr = PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL); CHKERRQ(ierr);

    // Create the PETSc vector x and fill it with random values
    Vec x;
    ierr = VecCreate(comm, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, N); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    PetscRandom rand;
    ierr = PetscRandomCreate(comm, &rand); CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand); CHKERRQ(ierr);
    ierr = VecSetRandom(x, rand); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    // On rank 0, print the first 10 samples of the input signal
    if (rank == 0) {
        PetscScalar *in_array;
        ierr = VecGetArray(x, &in_array); CHKERRQ(ierr);
        PetscPrintf(comm, "Input signal (first 10 samples):\n");
        for (PetscInt i = 0; i < 10 && i < N; i++) {
            PetscPrintf(comm, "x[%ld] = %f\n", (long)i, (double)in_array[i]);
        }
        ierr = VecRestoreArray(x, &in_array); CHKERRQ(ierr);
    }

    // Compute the forward DFT
    double *dft_real = NULL, *dft_imag = NULL;
    ierr = compute_dft(x, N, &dft_real, &dft_imag, comm); CHKERRQ(ierr);

    // Compute the inverse DFT
    double *idft_real = NULL, *idft_imag = NULL;
    ierr = compute_idft(dft_real, dft_imag, N, &idft_real, &idft_imag, comm); CHKERRQ(ierr);

    // On rank 0, print the first 10 samples of the reconstructed signal
    if (rank == 0) {
        PetscPrintf(comm, "Reconstructed signal (first 10 samples):\n");
        for (PetscInt i = 0; i < 10 && i < N; i++) {
            PetscPrintf(comm, "x_reconstructed[%ld] = %f + i%f\n", (long)i, idft_real[i], idft_imag[i]);
        }
        free(dft_real);
        free(dft_imag);
        free(idft_real);
        free(idft_imag);
    }

    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}