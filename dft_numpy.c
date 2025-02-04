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

    // Get n from options (default n = 100 if not specified)
    PetscInt n = 100;
    ierr = PetscOptionsGetInt(NULL, NULL, "-N", &n, NULL); CHKERRQ(ierr);

    // Create the PETSc vector 'x'
    Vec x;
    ierr = VecCreate(comm, &x); CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    // Rank 0 reads the input signal from file
    if (rank == 0) {
        FILE *fp = fopen("input_signal.txt", "r");
        if (!fp) {
            SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open input_signal.txt");
        }
        for (PetscInt i = 0; i < n; i++) {
            double val;
            if (fscanf(fp, "%lf", &val) != 1) {
                SETERRQ(comm, PETSC_ERR_FILE_READ, "Error reading input_signal.txt");
            }
            ierr = VecSetValue(x, i, val, INSERT_VALUES); CHKERRQ(ierr);
        }
        fclose(fp);
    }
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    // Compute the DFT
    double *dft_real = NULL, *dft_imag = NULL;
    ierr = compute_dft(x, n, &dft_real, &dft_imag, comm); CHKERRQ(ierr);

    // Rank 0 writes the DFT results to file
    if (rank == 0) {
        FILE *fp_out = fopen("dft_results.txt", "w");
        if (!fp_out) {
            SETERRQ(comm, PETSC_ERR_FILE_OPEN, "Unable to open dft_results.txt for writing");
        }
        for (PetscInt k = 0; k < n; k++) {
            fprintf(fp_out, "%ld %f %f\n", (long)k, dft_real[k], dft_imag[k]);
        }
        fclose(fp_out);
        free(dft_real);
        free(dft_imag);
    }

    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}