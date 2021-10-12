#!/bin/bash

# JobName #
#PBS -N myJob

#Which Queue to use #
#PBS -q N10C80

# Max VM size #
#PBS -l pvmem=2G

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:01:00

# How many nodes and tasks per node
# Example 4 nodes, 8 cores/node, 8 mpiprocs/node => 32 procs on 32 cores
#PBS -l select=2:ncpus=2:mpiprocs=1:ompthreads=2:mem=16400000kb

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun --bind-to none Parallel_Jacobi_Hybrid.x<input
