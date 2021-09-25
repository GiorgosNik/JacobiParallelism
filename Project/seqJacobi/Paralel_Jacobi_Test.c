#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
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
                            double alpha, double omega)
{
#define SRC(XX,YY) src[(YY)*maxXCount+(XX)]
#define DST(XX,YY) dst[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double error = 0.0;
    double updateVal;
    double f;
    // Coefficients
    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            f = -alpha*(1.0-fX*fX)*(1.0-fY*fY) - 2.0*(1.0-fX*fX) - 2.0*(1.0-fY*fY);
            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                			(SRC(x,y-1) + SRC(x,y+1))*cy +
                			SRC(x,y)*cc - f
						)/cc;
            DST(x,y) = SRC(x,y) - omega*updateVal;
            error += updateVal*updateVal;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}


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
    int my_rank, comm_sz,local_n;   
    double a = 0.0, b = 3.0, h, local_a, local_b;
    double local_int, total_int;
    int source, prov; 
    int control = 0;
    int message;
    int receiver_id;
    MPI_Comm cart_comm;
    int dim[] = {2,2};
    int period[] = {0, 0};
    int reorder = 1;
    MPI_Status status;


    
    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    iterationCount = 0;
    error = HUGE_VAL;
    clock_t start = clock(), diff;
    int cords[2];
    int neighbourCords[2];
    int neighbourRank;

    //######### START OF MPI #########
    MPI_Init(&argc, &argv);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder,&cart_comm);
    t1 = MPI_Wtime();
    MPI_Pcontrol(0);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Cart_coords(cart_comm,my_rank,2,cords);
    printf("Cords =:%d %d\n",cords[0],cords[1]);
    if(my_rank==0){
        printf("I AM MASTER\n");
        scanf("%d,%d", &n, &m);
        scanf("%lf", &alpha);
        scanf("%lf", &relax);
        scanf("%lf", &maxAcceptableError);
        scanf("%d", &maxIterationCount);
        printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, maxAcceptableError, maxIterationCount);
        allocCount = (n+2)*(m+2);
        u = 	(double*)calloc(allocCount, sizeof(double)); //reverse order
        u_old = (double*)calloc(allocCount, sizeof(double));
        if (u == NULL || u_old == NULL)
        {
            printf("Not enough memory for two %ix%i matrices\n", n+2, m+2);
            exit(1);
        }
        for(receiver_id=1;receiver_id<comm_sz;receiver_id++){
            message=receiver_id*2;
            printf("Master Cords =:%d %d\n",cords[0],cords[1]);
            printf("Sending %d to %d\n",message,receiver_id);
            MPI_Send( &message, 1 , MPI_INT,receiver_id, send_data_tag, MPI_COMM_WORLD);
        }
    }
    if(my_rank!=0){
        if(cords[0]>0){
            neighbourCords[0]=cords[0]-1;
            neighbourCords[1]=cords[1];
            MPI_Cart_rank(cart_comm, neighbourCords, &neighbourRank);
            printf("Above Neighbour: %d\n",neighbourRank);
        }
        MPI_Recv( &message, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("My rank is: %d,Cords :[%d:%d], My message is: %d\n",my_rank,cords[0],cords[1],message);
    }



    t2 = MPI_Wtime();
    /*printf( "Iterations=%3d Elapsed MPI Wall time is %f\n", iterationCount, t2 - t1 ); */
    MPI_Finalize();
    //######### END OF MPI #########
    

    
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    /*printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    printf("Residual %g\n",error);*/

    // u_old holds the solution after the most recent buffers swap
    /*double absoluteError = checkSolution(xLeft, yBottom,n+2, m+2,u_old,deltaX, deltaY,alpha);*/
    /*printf("The error of the iterative solution is %g\n", absoluteError);*/

    return 0;
}
