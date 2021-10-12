#!/bin/bash

# JobName #
#PBS -N myJob

#Which Queue to use #
#PBS -q N10C80

# Max VM size #
#PBS -l pvmem=4G

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:10:00

# How many nodes and tasks per node
#PBS -l select=2:ncpus=2:mpiprocs=2:mem=16400000kb

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun Parallel_Jacobi_MPI.x < input

