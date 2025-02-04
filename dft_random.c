#include <petsc.h>
#include "dft_lib.h"
#include <stdlib.h>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    MPI_Comm comm = PETSC_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Default n; can be changed via -N
    PetscInt n = 1000;
    ierr = PetscOptionsGetInt(NULL, NULL, "-N", &n, NULL); CHKERRQ(ierr);

    // Create the input vector x
    Vec x;
    ierr = VecCreate(comm, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    // Fill the vector with random values
    PetscRandom rand_obj;
    ierr = PetscRandomCreate(comm, &rand_obj); CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand_obj); CHKERRQ(ierr);
    ierr = VecSetRandom(x, rand_obj); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand_obj); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    // Check for the "-print_results" flag
    PetscBool print_results = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-print_results", &print_results, NULL); CHKERRQ(ierr);

    // Compute the DFT
    double *dft_real = NULL, *dft_imag = NULL;
    double t_start = MPI_Wtime();
    ierr = compute_dft(x, n, &dft_real, &dft_imag, comm); CHKERRQ(ierr);
    double t_end = MPI_Wtime();

    // On rank 0, print the info and results if flag is provided
    if (rank == 0) {
        PetscPrintf(comm, "DFT computed in %f seconds\n", t_end - t_start);
        if (print_results) {
            PetscPrintf(comm, "DFT Results (Random/Synthetic Input):\n");
            for (PetscInt k = 0; k < n; k++) {
                PetscPrintf(comm, "k=%ld: %f + i%f\n", (long)k, dft_real[k], dft_imag[k]);
            }
        }
        free(dft_real);
        free(dft_imag);
    }

    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}