# Parallel DFT and IDFT in PETSc

## Overview
This repository provides a parallel implementation of the Discrete Fourier Transform (DFT) and Inverse Discrete Fourier Transform (IDFT) using the PETSc library. This project is a final assignment for the Numerical Linear Algebra class and is designed to perform parallel computations, leveraging the capabilities of PETSc for efficient and scalable scientific computations.

## Features
- Parallel computation of DFT and IDFT.
- Utilizes PETSc for efficient handling of large-scale scientific computations.
- Suitable for high-performance computing environments.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- **PETSc**: Install PETSc by following the [PETSc installation guide](https://petsc.org/release/install/) or follow instructions in the Jupyter notebook.
- **MPI**: Ensure you have an MPI implementation installed, such as MPICH or OpenMPI.
- **Python**: Ensure you have Python version 3.12 or higher.
- **Nix** (Optional): For easier setup, install Nix by following the [Nix installation guide](https://nixos.org/download.html).
- **uv** (Optional): For easier setup (if not using Nix, redundant if you do) install uv by following the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Installation
To set up the environment using Nix and install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/docherak/vsb-petsc-fourier.git
    cd vsb-petsc-fourier
    ```

2. Use Nix to get `uv`, set up the environment, and start Jupyter Lab:
    ```bash
    nix-shell -p uv --run "uv run --with jupyter jupyter lab"
    ```

## Usage
To run the DFT and IDFT computations, follow the instructions provided in the Jupyter notebook. It contains a guide on setting up PETSc/MPI, a description of the source files, along with a few test and comparison runs.
