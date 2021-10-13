# JacobiParallelism
### Introduction
This project aims to explore the performance and scaling of parallel computation using MPI on its own, and in combination with OpenMP. This project was completed as part of the
Concurrent Systems (ΘΠ04), with prof. Ioannis Cotronis during the academic year 2020-21. Given an initial sequential program that calculated that solved the Poisson Equation using 
the Jacobi Over-relaxation method, we were to improve its efficiency, parallelize it using at first MPI and then a combination of MPI/OpenMP. Finally, we were to compare our solution
to a given challenge.

## Contents:
In this repository you will find:
* The Initial Sequential code we were given
* Our Modified Version of the Sequential code
* Our Implementation of MPI parallelism
* Our Implementation of a Hybrid MPI/OpenMP Solution
* The Challenge Program
* The Assignment descripion and guidelines (sadly only in Greek for now :( )
* (In the future) The results we obtained


## Methodology:
#### Sequential Optimization:
In order to optimize the initial sequential program, we:
* Declared most functions as static inline.
* Removed excessive recalculation of various variables in the method by calculating them a single time outside the main loop.
* Removed various variables that only stored information temporarily by creating bigger, more complex statements.

## MPI Design and Optimization:
In order to exploit a high number of process, we had to split the problem into smaller chunks. In the sequential version, the program performed computations on a matrix of size N*M
these being given as input. We decided to create a Cartesian grid, and split this matrix into smaller sub-matrixes, one for each process. Since the calculations for the value of
each element is performed using a stencil-like pattern involving the surrounding elements, the calculation of a process's outer elements involved elements of its neighboring process.
In order to perform data-sharing, each process was allocated two extra rows and columns, in which it would store the elements it received from its neighbors.
This resulted in a structure like shown below with the internal elements being white, outer in green, and the elements received from the neighboring processes shown in yellow.

IMAGE HERE

Using custom data types, each process transmitted and received the needed elements to and from the neighboring processes. While the data transfer was taking place, the internal
elements were calculated to save time, followed by the external elements after the transfer was completed. After the calculation of each iteration, the error was calculated for
all processes using Allreduce, and then calculation started again. Using this method, we were able to complete the calculation in almost half the time the Challenge program needed.

## Hybrid MPI/OpenMP Design and Optimization:
We opted to parallelize the for loop of the Jacobi Iteration using threads. After testing various of Process Number/Thread Number combinations, we decided on using two Threads/Process.
The results using this method were similar to the pure MPI implementation for small numbers of Threads, lacking in the case of small data-size and high in cases of big matrices and
high numbers of threads.

