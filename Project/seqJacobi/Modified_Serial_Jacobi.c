#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

/*************************************************************
 * Performs one iteration of the Jacobi method and computes
 * the residual value.
 *
 * NOTE: u(0,*), u(maxXCount-1,*), u(*,0) and u(*,maxYCount-1)
 * are BOUNDARIES and therefore not part of the solution.
 *************************************************************/
 static inline double one_jacobi_iteration(double xStart, double yStart,
                            int maxXCount, int maxYCount,
                            double *src, double *dst,
                            double deltaX, double deltaY,
                            double alpha, double omega)
{
#define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
#define DST(XX,YY) dst[(YY)*maxXCount+(XX)]
    int x, y;
    double fY;
    double error = 0.0;
    double updateVal;
    
    // Initial Calculation of delatX/deltaY squares to avoid recalculation every loo
    double deltaXSQR=deltaX*deltaX;
    double deltaYSQR=deltaY*deltaY;

    // Coefficients
    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*(1.0/(deltaXSQR)) +
                			(SRC(x,y-1) + SRC(x,y+1))*(1.0/(deltaYSQR)) +
                			SRC(x,y)* (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha) - (-alpha*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX))*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)) - 2.0*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX)) - 2.0*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)))
						)/ (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha);
            DST(x,y) = SRC(x,y) - omega*updateVal;
            error += updateVal*updateVal;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}


/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
static inline double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}


int main(int argc, char **argv)
{
    int n, m, maxIterationCount;
    double alpha, maxAcceptableError, relax;
    double error;
    double *u, *u_old, *tmp;
    int allocCount;
    int iterationCount;
    double t1, t2;

    scanf("%d,%d", &n, &m);
    scanf("%lf", &alpha);
    scanf("%lf", &relax);
    scanf("%lf", &maxAcceptableError);
    scanf("%d", &maxIterationCount);


    printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, maxAcceptableError, maxIterationCount);

    allocCount = (n+2)*(m+2);
    // Those two calls also zero the boundary elements
    u = 	(double*)calloc(allocCount, sizeof(double)); //reverse order
    u_old = (double*)calloc(allocCount, sizeof(double));
    
    // Check if two matrixes were created    
    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n", n+2, m+2);
        exit(1);
    }

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;
    
    MPI_Init(NULL,NULL);
    t1 = MPI_Wtime();

    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    {    	
        error = one_jacobi_iteration(xLeft, yBottom, n+2, m+2, u_old, u, deltaX, deltaY,alpha, relax);
        iterationCount++;

        // Swap the matrixes
        tmp = u_old;
        u_old = u;
        u = tmp;
    }

    t2 = MPI_Wtime();
    printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 ); 
    MPI_Finalize();
    
    
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n",error);

    // u_old holds the solution after the most recent buffers swap
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n+2, m+2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha);
    printf("The error of the iterative solution is %g\n", absoluteError);

    return 0;
}
