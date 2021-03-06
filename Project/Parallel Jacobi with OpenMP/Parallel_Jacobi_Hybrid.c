#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#define send_data_tag 2001

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
                            double alpha, double omega){
#define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
#define DST(XX,YY) dst[(YY)*maxXCount+(XX)]
    int x, y;
    double error = 0.0;
    double updateVal;
    double f;
    double deltaXSQR=deltaX*deltaX;
    double deltaYSQR=deltaY*deltaY;

    // Parallelization Directive for the Loops
    #pragma omp parallel for
    for (y = 2; y < (maxYCount-3); y++)
    {
        
        for (x = 2; x < (maxXCount-3); x++)
        {   
            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*(1.0/(deltaXSQR)) +
                			(SRC(x,y-1) + SRC(x,y+1))*(1.0/(deltaYSQR)) +
                			SRC(x,y)* (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha) - (-alpha*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX))*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)) - 2.0*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX)) - 2.0*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)))
						)/ (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha);
            DST(x,y) = SRC(x,y) - omega*updateVal;
            error += updateVal*updateVal;
        }
    }
    return error;
}
static inline double one_jacobi_iterationOut(double xStart, double yStart,
                            int maxXCount, int maxYCount,
                            double *src, double *dst,
                            double deltaX, double deltaY,
                            double alpha, double omega){
#define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
#define DST(XX,YY) dst[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double error = 0.0;
    double updateVal;
    double f;
    double deltaXSQR=deltaX*deltaX;
    double deltaYSQR=deltaY*deltaY;

    // Coefficients

    #pragma omp parallel for
    for (y = maxYCount-2; y < (maxYCount-1); y++)
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
    #pragma omp parallel for
    for (y = 1; y < (2); y++)
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
    #pragma omp parallel for
    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (2); x++)
        {   
            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*(1.0/(deltaXSQR)) +
                			(SRC(x,y-1) + SRC(x,y+1))*(1.0/(deltaYSQR)) +
                			SRC(x,y)* (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha) - (-alpha*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX))*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)) - 2.0*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX)) - 2.0*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)))
						)/ (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha);
            DST(x,y) = SRC(x,y) - omega*updateVal;
            error += updateVal*updateVal;
        }
    }
    #pragma omp parallel for
    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = maxXCount-2; x < (maxXCount-1); x++)
        {   
            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*(1.0/(deltaXSQR)) +
                			(SRC(x,y-1) + SRC(x,y+1))*(1.0/(deltaYSQR)) +
                			SRC(x,y)* (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha) - (-alpha*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX))*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)) - 2.0*(1.0-(xStart + (x-1)*deltaX)*(xStart + (x-1)*deltaX)) - 2.0*(1.0-(yStart + (y-1)*deltaY)*(yStart + (y-1)*deltaY)))
						)/ (-2.0*(1.0/(deltaXSQR))-2.0*(1.0/(deltaYSQR))-alpha);
            DST(x,y) = SRC(x,y) - omega*updateVal;
            error += updateVal*updateVal;
        }
    }
    return error;
}

}

static inline double checkSolution(double xStart, double yStart, int maxXCount, int maxYCount, double *u, double deltaX, double deltaY, double alpha){
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
    return error;
}

static inline int* getNeighbors(MPI_Comm cart_comm,int myRank,int sizeX, int sizeY){
    int *Neighbors;
    int cords[2];
    int neighborCords[2];
    int upNeighbor,downNeighbor,leftNeighbor,rightNeighbor;
    Neighbors=malloc(sizeof(int)*4);
    //Get my coordinates in the grid
    MPI_Cart_coords(cart_comm,myRank,2,cords);

    //Get my Neighbors in the grid
    if(cords[0]>0){
            neighborCords[0] = cords[0]-1;
            neighborCords[1] = cords[1];
            MPI_Cart_rank(cart_comm, neighborCords, &upNeighbor);
            Neighbors[0] = upNeighbor;
    }else{
        Neighbors[0] =- 1;
    }
        
    if(cords[0]<sizeX-1){
        neighborCords[0] = cords[0]+1;
        neighborCords[1] = cords[1];
        MPI_Cart_rank(cart_comm, neighborCords, &downNeighbor);
         Neighbors[1] = downNeighbor;
    }else{
        Neighbors[1]=-1;
    }
    if(cords[1]<sizeY-1){
        neighborCords[0] = cords[0];
        neighborCords[1] = cords[1]+1;
        MPI_Cart_rank(cart_comm, neighborCords, &rightNeighbor);
        Neighbors[2] = rightNeighbor;
    }else{
        Neighbors[2] =- 1;
    }
    if(cords[1]>0){
        neighborCords[0] = cords[0];
        neighborCords[1] = cords[1]-1;
        MPI_Cart_rank(cart_comm, neighborCords, &leftNeighbor);
        Neighbors[3] = leftNeighbor;
    }else{
        Neighbors[3] =- 1;
    }
    return Neighbors;
}

static inline int setup(double *u, double * u_old,int n,int m,int allocCount){
        allocCount = (n+2)*(m+2);
        u = (double*)calloc(allocCount, sizeof(double)); //reverse order
        u_old = (double*)calloc(allocCount, sizeof(double));
        if (u == NULL || u_old == NULL)
        {
            printf("Not enough memory for two %ix%i matrices\n", n+2, m+2);
            return(1);
        }
        return(0);
}

static inline int getSizes(int n, int m, int procs, int sizeX, int sizeY, int* rowPoints, int* columnPoints){
    *rowPoints = (int)(n/ sizeX);
    *columnPoints = (int)(m/ sizeY);
    return 0;
}

static inline int calculateDims(int n, int m, int sizeX, int sizeY, int rowPoints, int columnPoints ,int* cords, double* xLeft, double * xRight, double* yDown, double* yUp, double* deltaX, double* deltaY){
    *deltaX = (double)2/(double)(n-1);
    *deltaY =(double)2/(double)(m-1);
    *xLeft = -1 + (*deltaX)*(cords[0]*rowPoints);
    *yDown = -1 +(*deltaY)* (cords[1]*columnPoints);

    *xRight = *xLeft+(*deltaX)*rowPoints;
    *yUp = *yDown+(*deltaY)*columnPoints;
    
    return 0;
}

int main(int argc, char **argv){

    // Setup for MPI
    int source, prov; 
    int control = 0;
    int receiver_id;
    MPI_Comm cart_comm;
    MPI_Status status;
    MPI_Datatype column_type;
    MPI_Datatype row_type;
    MPI_Request requestUpSend,requestUpGet;
    MPI_Request requestDownSend,requestDownGet;
    MPI_Request requestLeftSend,requestLeftGet;
    MPI_Request requestRightSend,requestRightGet;
    
    int period[] = {0, 0};
    int reorder = 1;
    int *myNeighbors;
    clock_t start = clock(), diff;
    int cords[2];
    int neighborCords[2];
    int upNeighbor,downNeighbor,leftNeighour,rightNeighbor;
    int sizeX;
    int sizeY;
    int rowPoints;
    int columnPoints;
    double* receivedLeft, *receivedRight, *receivedUp, *receivedDown;
    double absoluteError;

    // General Setup
    int n, m, maxIterationCount,allocCount,iterationCount;
    double alpha, maxAcceptableError, relax,error;
    double *u, *u_old, *tmp;
    double t1, t2;
    int my_rank, comm_sz,local_n;   
    double a = 0.0, b = 3.0, h, local_a, local_b;
    double local_int, total_int;
    
    
    // Setup for Jacobi
    int buffInt[3];
    double buffDouble[3];
    double xLeft, xRight;
    double yBottom, yUp;
    double deltaX;
    double deltaY;

    iterationCount = 0;
    error = HUGE_VAL;
    
    MPI_Pcontrol(1);
    //######### START OF MPI #########
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    if(sqrt(comm_sz) == ceil(sqrt(comm_sz))){
        sizeX = (int)sqrt(comm_sz);
        sizeY = sizeX;
    }else if(comm_sz==2){
        sizeX = 2;
        sizeY = 1;
    }else{
        sizeX = 4;
        sizeY = 10;
    }
    int dim[] = {sizeX,sizeY};

    // Create the Cartesian Grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder,&cart_comm);
    
    // Get my Rank and the number of other procs
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Cart_coords(cart_comm,my_rank,2,cords);
    
    if(my_rank == 0){
        // If I am the master proc, get and transmit the input to the others
        scanf("%d,%d", &n, &m);
        scanf("%lf", &alpha);
        scanf("%lf", &relax);
        scanf("%lf", &maxAcceptableError);
        scanf("%d", &maxIterationCount);
        printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, maxAcceptableError, maxIterationCount);
        buffInt[0] = n;
        buffInt[1] = m;
        buffInt[2]=maxIterationCount;
        buffDouble[0] = alpha;
        buffDouble[1] = relax;
        buffDouble[2] = maxAcceptableError;

        if (setup(u,u_old,n,m,allocCount)==1){
            exit(1);
        }
    }

    // Broadcast Input (Integers)
    MPI_Bcast(buffInt, 3, MPI_INT, 0, MPI_COMM_WORLD);
    n = buffInt[0];
    m = buffInt[1];
    maxIterationCount = buffInt[2];

    // Broadcast Input (Doubles)
    MPI_Bcast(buffDouble, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    alpha = buffDouble[0];
    relax = buffDouble[1];
    maxAcceptableError=buffDouble[2];

    // Get Dimensions of Grid 
    getSizes(n, m, comm_sz, sizeX, sizeY, &rowPoints, &columnPoints);

    //Custom Column Datatype
    MPI_Type_vector(columnPoints+2, 1, columnPoints+2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    //Custom Row Datatype
    MPI_Type_contiguous(rowPoints+2, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    // Calculate Neighbors
    myNeighbors = getNeighbors(cart_comm,my_rank, sizeX, sizeY);

    // Calculate Deltas and X/Y coordinates of submatrix
    calculateDims(n ,m, sizeX, sizeY,rowPoints, columnPoints, cords, &xLeft, &xRight, &yBottom, &yUp, &deltaX, &deltaY);
     
    // Create the two sub-matrixes
    allocCount = (rowPoints+2)*(columnPoints+2);

    // Those two calls also zero the boundary elements
    u = 	(double*)calloc(allocCount, sizeof(double)); //reverse order
    u_old = (double*)calloc(allocCount, sizeof(double));

    // Check if the two matrixes were created correctly   
    if (u == NULL || u_old == NULL)
    {
        printf("Not enough memory for two %ix%i matrices\n", rowPoints+2, columnPoints+2);
        exit(1);
    }

    // Barrier to sync all procs
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    // Start Jacobi Calculations
    while (iterationCount < maxIterationCount && error > maxAcceptableError)
    {   
        error=0;

        // Receive Operations
        if (myNeighbors[0] != -1){
            MPI_Irecv(&(u_old[0]), rowPoints+2, row_type, myNeighbors[0], 0, MPI_COMM_WORLD, &requestUpGet);
        }
        if (myNeighbors[1] != -1){
            MPI_Irecv(&(u_old[columnPoints+1]), rowPoints+2, row_type, myNeighbors[1], 0, MPI_COMM_WORLD, &requestDownGet);
        }
        if (myNeighbors[2] != -1){
            MPI_Irecv(&(u_old[(rowPoints+1)]), columnPoints+2, column_type, myNeighbors[2], 0, MPI_COMM_WORLD, &requestRightGet);
        }
        if (myNeighbors[3] != -1){
            MPI_Irecv(&(u_old[0]), columnPoints+2, column_type, myNeighbors[3], 0, MPI_COMM_WORLD, &requestLeftGet);
        }

        // Sending Operations
        if (myNeighbors[0] != -1){
            MPI_Isend(&u_old[rowPoints], 1, row_type,myNeighbors[0] , 0, MPI_COMM_WORLD, &requestUpSend);
        }
        if (myNeighbors[1] != -1){
            MPI_Isend(&u_old[columnPoints*rowPoints], 1, row_type, myNeighbors[1], 0, MPI_COMM_WORLD, &requestDownSend);
        }
        if (myNeighbors[2] != -1){
            MPI_Isend(&u_old[rowPoints], 1, column_type, myNeighbors[2], 0, MPI_COMM_WORLD, &requestRightSend);
        }
        if (myNeighbors[3] != -1){
            MPI_Isend(&u_old[1], 1, column_type, myNeighbors[3], 0, MPI_COMM_WORLD, &requestLeftSend);
        }


        // Calculate Internal Jacobi
        error = one_jacobi_iteration(xLeft, yBottom, rowPoints+2, columnPoints+2, u_old, u,deltaX, deltaY, alpha, relax);

        // Wait for Sending and Receiving to Complete
        if (myNeighbors[0] != -1){
            MPI_Wait(&requestUpGet, MPI_STATUS_IGNORE);
        }
        if (myNeighbors[1] != -1){
            MPI_Wait(&requestDownGet, MPI_STATUS_IGNORE);
        }
        if (myNeighbors[2] != -1){
            MPI_Wait(&requestRightGet, MPI_STATUS_IGNORE);
        }
        if (myNeighbors[3] != -1){
            MPI_Wait(&requestLeftGet, MPI_STATUS_IGNORE);
        }
        
        
        // Calculate External Jacobi
        error += one_jacobi_iterationOut(xLeft, yBottom, rowPoints+2, columnPoints+2, u_old, u,deltaX, deltaY, alpha, relax);
        MPI_Allreduce(&error, &error, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);

        
        if (myNeighbors[0] != -1){
            MPI_Wait(&requestUpSend, MPI_STATUS_IGNORE);
        }
        if (myNeighbors[1] != -1){
            MPI_Wait(&requestDownSend, MPI_STATUS_IGNORE);
        }
        if (myNeighbors[2] != -1){
            MPI_Wait(&requestRightSend, MPI_STATUS_IGNORE);
        }
        if (myNeighbors[3] != -1){
            MPI_Wait(&requestLeftSend, MPI_STATUS_IGNORE);
        }
        
        
        error=sqrt(error)/(n*m);
        iterationCount++;

        // Swap the Arrays
        tmp = u_old;
        u_old = u;
        u = tmp;
    }
    t2 = MPI_Wtime();
    MPI_Pcontrol(0);
    // Check the solution
    absoluteError= checkSolution(xLeft, yBottom, rowPoints+2, columnPoints+2, u_old, deltaX, deltaY, alpha);
    MPI_Allreduce(&absoluteError, &absoluteError, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
    absoluteError=sqrt(absoluteError)/(n*m);

   
    MPI_Finalize();
    //######### END OF MPI #########
    if(my_rank==0){
        // Final Measurements
        printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 );
        printf("The error of the iterative solution is %g\n", absoluteError);
        diff = clock() - start;
        int msec = diff * 1000 / CLOCKS_PER_SEC;
        printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
        printf("Residual %g\n",error);
    }

    return 0;
}