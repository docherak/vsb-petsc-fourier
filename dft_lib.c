#include "dft_lib.h"
#include <math.h>
#include <stdlib.h>

// Define M_PI, just because C stuff :')
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

PetscErrorCode compute_dft(Vec x, PetscInt N, double **dft_real, double **dft_imag, MPI_Comm comm) {
    PetscErrorCode ierr;
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Get the local portion of the vector x
    PetscInt n_start, n_end, local_size;
    ierr = VecGetOwnershipRange(x, &n_start, &n_end); CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &local_size); CHKERRQ(ierr);
    PetscScalar *x_local;
    ierr = VecGetArray(x, &x_local); CHKERRQ(ierr);

    // Allocate output arrays on rank 0
    if (rank == 0) {
        *dft_real = (double *)calloc(N, sizeof(double));
        *dft_imag = (double *)calloc(N, sizeof(double));
    }

    // Loop over each frequency bin k
    for (PetscInt k = 0; k < N; k++) {
        double local_sum_re = 0.0, local_sum_im = 0.0;
        // Each process sums over its local portion
        for (PetscInt i = 0; i < local_size; i++) {
            PetscInt n = n_start + i;
            double val = (double)x_local[i];
            double angle = -2.0 * M_PI * k * n / (double)N;
            local_sum_re += val * cos(angle);
            local_sum_im += val * sin(angle);
        }
        // Reduce the local sums across all processes
        double global_sum_re, global_sum_im;
        MPI_Allreduce(&local_sum_re, &global_sum_re, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_sum_im, &global_sum_im, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (rank == 0) {
            (*dft_real)[k] = global_sum_re;
            (*dft_imag)[k] = global_sum_im;
        }
    }

    ierr = VecRestoreArray(x, &x_local); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode compute_idft(double *dft_real, double *dft_imag, PetscInt N,
                           double **idft_real, double **idft_imag, MPI_Comm comm) {
    PetscErrorCode ierr;
    int rank;
    MPI_Comm_rank(comm, &rank);

    double *Xreal = NULL, *Ximag = NULL;
    if (rank == 0) {
        Xreal = dft_real;
        Ximag = dft_imag;
    } else {
        Xreal = (double *)malloc(N * sizeof(double));
        Ximag = (double *)malloc(N * sizeof(double));
    }
    MPI_Bcast(Xreal, N, MPI_DOUBLE, 0, comm);
    MPI_Bcast(Ximag, N, MPI_DOUBLE, 0, comm);

    Vec y;
    ierr = VecCreate(comm, &y); CHKERRQ(ierr);
    ierr = VecSetSizes(y, PETSC_DECIDE, N); CHKERRQ(ierr);
    ierr = VecSetFromOptions(y); CHKERRQ(ierr);

    PetscInt n_start, n_end, local_size;
    ierr = VecGetOwnershipRange(y, &n_start, &n_end); CHKERRQ(ierr);
    ierr = VecGetLocalSize(y, &local_size); CHKERRQ(ierr);

    for (PetscInt n = n_start; n < n_end; n++) {
        double sum_re = 0.0, sum_im = 0.0;
        for (PetscInt k = 0; k < N; k++) {
            double angle = 2.0 * M_PI * k * n / (double)N;
            sum_re += Xreal[k] * cos(angle) - Ximag[k] * sin(angle);
            sum_im += Xreal[k] * sin(angle) + Ximag[k] * cos(angle);
        }
        sum_re /= N;
        sum_im /= N;
        ierr = VecSetValue(y, n, sum_re, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = VecAssemblyBegin(y); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(y); CHKERRQ(ierr);

    if (rank == 0) {
        PetscScalar *array;
        ierr = VecGetArray(y, &array); CHKERRQ(ierr);
        *idft_real = (double *)malloc(N * sizeof(double));
        *idft_imag = (double *)calloc(N, sizeof(double));
        for (PetscInt i = 0; i < N; i++) {
            (*idft_real)[i] = (double)array[i];
        }
        ierr = VecRestoreArray(y, &array); CHKERRQ(ierr);
    }

    ierr = VecDestroy(&y); CHKERRQ(ierr);

    if (rank != 0) {
        free(Xreal);
        free(Ximag);
    }
    return 0;
}