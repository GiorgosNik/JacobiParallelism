all: Parallel_Jacobi_MPI.c
	mpicc -O3 -g  -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -lm -fopenmp Parallel_Jacobi_MPI.c -o Parallel_Jacobi_MPI.x