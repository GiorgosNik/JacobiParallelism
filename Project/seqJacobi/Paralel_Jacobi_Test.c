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
                            double alpha, double omega){
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
                     double alpha){
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

static inline int * getNeighbours(MPI_Comm cart_comm,int myRank,int side){
    int *Neighbours;
    int cords[2];
    int neighbourCords[2];
    int upNeighbour,downNeighbour,leftNeighbour,rightNeighbour;
    Neighbours=malloc(sizeof(int)*4);
    //Get my coordinates in the grid
    MPI_Cart_coords(cart_comm,myRank,2,cords);

    //Get my Neighbours in the grid
    if(cords[0]>0){
            neighbourCords[0]=cords[0]-1;
            neighbourCords[1]=cords[1];
            MPI_Cart_rank(cart_comm, neighbourCords, &upNeighbour);
            Neighbours[0]=upNeighbour;
    }else{
        Neighbours[0]=-1;
    }
        
    if(cords[0]<side-1){
        neighbourCords[0]=cords[0]+1;
        neighbourCords[1]=cords[1];
        MPI_Cart_rank(cart_comm, neighbourCords, &downNeighbour);
         Neighbours[1]=downNeighbour;
    }else{
        Neighbours[1]=-1;
    }
    if(cords[1]<side-1){
        neighbourCords[0]=cords[0];
        neighbourCords[1]=cords[1]+1;
        MPI_Cart_rank(cart_comm, neighbourCords, &rightNeighbour);
        Neighbours[2]=rightNeighbour;
    }else{
        Neighbours[2]=-1;
    }
    if(cords[1]>0){
        neighbourCords[0]=cords[0];
        neighbourCords[1]=cords[1]-1;
        MPI_Cart_rank(cart_comm, neighbourCords, &leftNeighbour);
        Neighbours[3]=leftNeighbour;
    }else{
        Neighbours[3]=-1;
    }
    return Neighbours;
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

static inline int getSizes(int n, int m, int procs, int* sizeX, int* sizeY, int* rowSize, int* collumnSize){
    if(sqrt(procs)%1!=0){
        *sizeX=sqrt(procs)%1;
        *sizeY=*sizeX;
    }else{
        *sizeX=8;
        *sizeY=10;
    }
    *rowSize=(n/ *sizeX+2)%1;
    *collumnSize=(m/ *sizeY+2)%1;
    return 0;
}

int main(int argc, char **argv)
{
    
    

    // Setup for MPI
    int source, prov; 
    int control = 0;
    int receiver_id,message;
    MPI_Comm cart_comm;
    MPI_Status status;
    int dim[] = {2,2};
    int period[] = {0, 0};
    int reorder = 1;
    int side;
    int *myNeighbours;
    clock_t start = clock(), diff;
    int cords[2];
    int neighbourCords[2];
    int upNeighbour,downNeighbour,leftNeighour,rightNeighbour;

    // General Setup
    int n, m, maxIterationCount,allocCount,iterationCount;
    double alpha, maxAcceptableError, relax,error;
    double *u, *u_old, *tmp;
    double t1, t2;
    int my_rank, comm_sz,local_n;   
    double a = 0.0, b = 3.0, h, local_a, local_b;
    double local_int, total_int;
    
    // Setup for Jacobi
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    iterationCount = 0;
    error = HUGE_VAL;


    //######### START OF MPI #########
    MPI_Init(&argc, &argv);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder,&cart_comm);
    t1 = MPI_Wtime();
    MPI_Pcontrol(control);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Cart_coords(cart_comm,my_rank,2,cords);

    //Get the size of the square
    side=sqrt(comm_sz);
    MPI_Cart_coords(cart_comm,my_rank,2,cords);
    if(my_rank==0){
        scanf("%d,%d", &n, &m);
        scanf("%lf", &alpha);
        scanf("%lf", &relax);
        scanf("%lf", &maxAcceptableError);
        scanf("%d", &maxIterationCount);
        printf("-> %d, %d, %g, %g, %g, %d\n", n, m, alpha, relax, maxAcceptableError, maxIterationCount);
        if (setup(u,u_old,n,m,allocCount)==1){
            exit(1);
        }
        for(receiver_id=1;receiver_id<comm_sz;receiver_id++){
            message=receiver_id*2;
            MPI_Send( &message, 1 , MPI_INT,receiver_id, send_data_tag, MPI_COMM_WORLD);
        }
        message = 0;
    }else if(my_rank!=0){
        MPI_Recv( &message, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
    myNeighbours=getNeighbours(cart_comm,my_rank,side);
    printf("My rank is: %d  My message is: %d My Up is %d My Down is %d My Left is %d My Right is %d\n",my_rank,message,myNeighbours[0],myNeighbours[1],myNeighbours[3],myNeighbours[2]);


    t2 = MPI_Wtime();
    MPI_Finalize();
    //######### END OF MPI #########
    

    
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    return 0;
}
