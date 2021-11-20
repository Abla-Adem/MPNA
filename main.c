#include <stdio.h>
#include <mpi.h>
#include<omp.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//alloc fonction
double *alloc_matrix_p(int n,int m)
{
    double *matrix= malloc(sizeof (double)*n*m);
    return matrix;
}
double *write_matrix_p(int n,int m,int file,char *name_file,int first_line_value,int sparse)
{

    double *matrix= malloc(sizeof (double)*n*m);


    if(file==0)
    {

        if(sparse==0)
        {

            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < m; ++j) {
                    //matrix[i][j]=rand()%INT_MAX;
                    matrix[i*m+j]= (rand()%10);

                }
            }
        }
        else
        {
            double random_sparce;
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < m; ++j) {
                    random_sparce=(double )(rand())/RAND_MAX;
                    //if(random_sparce>0.7)
                    //{

                    matrix[i*m+j]=j+1;


                    //}
                    //else
                    //{
                    //    matrix[i][j]=0;
                    //}
                    //matrix[i][j]=rand()%INT_MAX;

                }
            }
            double *vecteur= malloc(sizeof(double )*m );
            double *result= malloc(sizeof(double )*m );
            for (int i = 0; i < m-3; ++i)
            {
                result[i]=0;
                vecteur[i]=i;
                //matrix[i][3+i]=0;
            }

            int diagonal=1,diag,indice_lapack_ligne=0;
            double *lapack_matrix= malloc(sizeof (double )*n*(m+1));
            for (int i = 0; i < m; ++i)
            {
                lapack_matrix[i]=0;
            }
            for (int j = m-1; j > -1; --j) {
                diag=0;
                for (int i = 0; i < diagonal; ++i)
                {
                    if(matrix[i*m+j+i]>0.0)
                    {
                        printf("%d %lf",j,matrix[i*m+j+i]);
                        diag=1;
                    }
                }
                if(diag==1)
                {
                    printf("%d \n",diagonal);
                    for (int w = 0; w < j; ++w) {
                        lapack_matrix[m*(indice_lapack_ligne+1)+w]=0;
                    }
                    for (int z = j; z < m; ++z) {
                        lapack_matrix[m*(indice_lapack_ligne+1)+z]=matrix[(z-j)*m+z];
                    }
                    indice_lapack_ligne=indice_lapack_ligne+1;
                }
                diagonal=diagonal+1;
            }
//            cblas_dgbmv(CblasRowMajor,CblasNoTrans,10,10,0,9,1,lapack_matrix,m,vecteur,1,1,result,1);
            printf("MATRIX %d*%d:\n", n, m);
            for (int i = 0; i < n+1; ++i) {
                printf("[");
                for (int j = 0; j < m; ++j) {
                    printf(" %lf ,", lapack_matrix[i*m+j]);
                }
                printf("] \n");
            }

            for (int j = 0; j < m; ++j) {
                printf("%lf ,",result[j]);
            }

        }
    }
    else
    {
        FILE* mscd = fopen(name_file,"r");
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        int car=0,i,j;
        double val;
        for (int ii = 0; ii < n; ++ii)
        {
            for (int jj = 0; jj < m; ++jj) {
                matrix[ii*m+jj]=0;
            }
        }
        int faire=0;
        do {
            car = fgetc(mscd);
            //printf("%d \n",car);
            if(car!=37)
            {

                fseek(mscd, -1, SEEK_CUR);
                fscanf(mscd, "%d %d %lf", &i,&j,&val);
                if(first_line_value==0 && faire==0)
                {
                    faire=1;
                }
                else
                {
                    matrix[i*m+j]=val;
                }

            }
            else
            {
                getline(&line, &len, mscd);

            }
            printf("%d \n",car);

        } while (car != EOF);
    }
    return matrix;
}
double *matrice_test_p()
{
    int a=11111111,b = 9090909,c = 10891089,d = 8910891,e = 11108889,f = 9089091,g = 10888911,h = 8909109;
    double *matrice= malloc(sizeof(double *)*8*8);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            matrice[i*8+j]=0;
        }
    }
    for (int i = 0; i < 8; ++i) {
        matrice[i*8+i]=a;
    }

    int pile=0,c_merde=0;
    for (int i = 0; i < 8; ++i) {
        matrice[i*8+8-i-1]=-h;

        if(i==3)
        {
            c_merde=1;
        }
        if(i%2==0)
        {
            matrice[i*8+8-i-2]=g;
            matrice[(i+1)*8+8-i-1]=g;
            matrice[i*8+i+1]=-b;
            matrice[(i+1)*8+i]=-b;
            if(pile==0)
            {
                matrice[i*8+i+2]=-c;
                matrice[(i+1)-8+i+3]=-c;

                matrice[i*8+i+3]=d;
                matrice[(i+1)*8+i+2]=d;
                if(c_merde==0)
                {
                    matrice[i*8+i+4]=-e;
                    matrice[(i+1)*8+i+5]=-e;

                    matrice[i*8+i+5]=f;
                    matrice[(i+1)*8+i+4]=f;
                }
                else
                {
                    matrice[i*8+i-4]=-e;
                    matrice[(i+1)*8+i-3]=-e;

                    matrice[i*8+i-3]=f;
                    matrice[(i+1)*8+i-4]=f;
                }

            }
            else
            {
                matrice[i*8+i-2]=-c;
                matrice[(i+1)*8+i-1]=-c;

                matrice[i*8+i-1]=d;
                matrice[(i+1)*8+i-2]=d;
                if(c_merde==0)
                {
                    matrice[i*8+i+4]=-e;
                    matrice[(i+1)*8+i+5]=-e;

                    matrice[i*8+i+5]=f;
                    matrice[(i+1)*8+i+4]=f;
                }
                else
                {
                    matrice[i*8+i-4]=-e;
                    matrice[(i+1)*8+i-3]=-e;

                    matrice[i*8+i-3]=f;
                    matrice[(i+1)+i-4]=f;
                }

            }





            if(pile==0)
            {
                pile=1;
            } else
            {
                pile=0;
            }
        }

    }
    return matrice;

}
void print_matrix(double ** matrix,int n,int m,char *s) {
    printf("    %s %d*%d:\n", s,n, m);
    for (int i = 0; i < n; ++i) {
        printf("    [");
        for (int j = 0; j < m; ++j) {
            printf(" %lf ,", matrix[i][j]);
        }

        printf("]\n");
    }
}

//print Function
void print_matrix_p(double * matrix,int n,int m,char *s) {
    printf("    %s %d*%d:\n", s,n, m);
    for (int i = 0; i < n; ++i) {
        printf("    [");
        for (int j = 0; j < m; ++j) {
            printf(" %lf ,", matrix[i*m+j]);
        }

        printf("]\n");
    }
}
void print_vector(double *vect,int n,char *s)
{
    printf( "%s=[",s);
    for (int i = 0; i < n; ++i) {
        printf("%lf ",vect[i]);
    }
    printf("]\n ");
}
void print_value_test(double ** matrix,double *matrix_p,int n,int m,double *vect_test,double *vect_test_2,int print,int world_size)
{
    if(print!=0)
    {
        printf("Valeur de test: \n");
        if(print==1)
        {
            print_vector(vect_test,n,"    vect_test");
            print_vector(vect_test_2,n,"   vect_test_2");
        }
        else if(print==2)
        {
            if(world_size>0)
            {
                print_matrix_p(matrix_p,n,m,"matrix");
            }
            else
            {
                print_matrix(matrix,n,m,"matrix");
            }

            print_vector(vect_test,n,"    vect_test");
        }
        else if (print==3)
        {
            if(world_size>0)
            {
                print_matrix(matrix,n,m,"matrix");
            }
            else
            {
                print_matrix_p(matrix_p,n,m,"matrix");
            }
            print_vector(vect_test,n,"    x");
        }
        else
        {
            if(world_size>0)
            {
                print_matrix(matrix,n,m,"matrix");
            }
            else
            {
                print_matrix_p(matrix_p,n,m,"matrix");
            }
        }
    }
}

//operation vector ajouter limite et nombre de pas

double *add_vector(double *a,double coef_a,double *b,double coef_b,int debut,int fin)
{

    double * y = calloc(fin-debut,sizeof(double ) );
    for (int i = debut; i < fin; ++i) {
        y[i]=coef_a*a[i]+coef_b*b[i];
    }
    return y;
}
/*
double *mul_vector(double *a,double coef_a,double *b,double coef_b,int debut,int fin)
{
    double * y = calloc(n,sizeof(double ) );
    for (int i = 0; i < n; ++i) {
        y[i]=a[i]*b[i];
    }
    return y;
}
double *sub_vector(double *a,double coef_a,double *b,double coef_b,int debut,int fin)
{
    double * y = calloc(n,sizeof(double ) );
    for (int i = 0; i < n; ++i) {
        y[i]=a[i]-b[i];
    }
    return y;
}
double *div_vector(double *a,double *b,int n)
{
    double * y = calloc(n,sizeof(double ) );
    for (int i = 0; i < n; ++i) {
        y[i]=a[i]/b[i];
    }
    return y;
}
*/
//operation matrix
double blas1(double *vecteur2,double *vecteur1,int n)
{
    double resultat=0;
    for (int i = 0; i < n; ++i) {
        resultat=vecteur2[i]*vecteur1[i]+resultat;
    }
    return resultat;
}
double *blas2(double * matrix,unsigned long long n,unsigned long long m,double *v)
{



    //calcul du produit local
    double *result_p=(double*) malloc(sizeof(double)*n),sum;
    for (int i = 0; i < n; ++i) {
        sum=0;
        for (int j = 0; j < m; ++j) {
            sum=sum+matrix[i*m+j]*v[j];
        }

        result_p[i]=sum;
    }

    return result_p;

}
double *blas3(double * matrix,unsigned long long n,unsigned long long m,double * matrix_1,unsigned long long n1,unsigned long long m1)
{
    double *result= alloc_matrix_p(n,n1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n1; ++j) {
            result[i*n1+j]=0;
            for (int k = 0; k < m1; ++k) {
                result[i*n1+j]=result[i*n1+j]+matrix[i*m+k]*matrix_1[j*m1+k];
            }
        }
    }
    return result;
}
/*
double blas1_p(double *vecteur2,double *vecteur1,double *vect_temp_1,double vect_temp_2,int n)
{

    MPI_Scatter(vecteur1,n,MPI_DOUBLE,vect_temp_1,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatter(vecteur2,n,MPI_DOUBLE,vect_temp_2,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

    double resultat=0;
    double resultat_final;
    for (int i = 0; i < n; ++i) {
        resultat=vecteur2[i]*vecteur1[i]+resultat;
    }
    MPI_Allreduce(&resultat,&resultat_final,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    return resultat;
}
void blas2_p(double * matrix,double *sub_matrix,double *result_p,double *result_final,unsigned long long n,unsigned long long m,double *v)
{


    MPI_Scatter(matrix,n_p*m,MPI_DOUBLE,sub_matrix,n_p*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(x,m,MPI_DOUBLE,0,MPI_COMM_WORLD);
    //calcul du produit local
    //double *result_p=(double*) malloc(sizeof(double)*n),sum;
    for (int i = 0; i < n; ++i) {
        sum=0;
        for (int j = 0; j < m; ++j) {
            sum=sum+matrix[i*m+j]*v[j];
        }

        result_p[i]=sum;
    }
    MPI_Allgather(result_p,n,MPI_DOUBLE,result_final,n,
                  MPI_DOUBLE,MPI_COMM_WORLD);


}
void blas3_p(double * matrix,double * matrix_p,unsigned long long n,unsigned long long m,unsigned long long limite,
                double * matrix_1,unsigned long long n1,unsigned long long m1,double *result,double *resultat_final)
{
    MPI_Bcast(matrix_1,m*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatter(matrix,n_p*m,MPI_DOUBLE,matrix_p,n_p*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n1; ++j) {
            result[i*n1+j]=0;
            for (int k = 0; k < m1; ++k) {
                result[i*n1+j]=result[i*n1+j]+matrix[i*m+k]*matrix_1[j*m1+k];
            }
        }
    }
    MPI_Allgather(result,n*m,MPI_DOUBLE,resultat_final,n*m,
                MPI_DOUBLE,MPI_COMM_WORLD);
}
*/

double *horner_p(double * matrix,double * x ,int limite,int n_p,int n,int m,int degre,int print,int world_size,int rank)
{

    double  *sub_matrix= alloc_matrix_p(n_p,m);
    double *v= malloc(sizeof (double )*(m));



    MPI_Scatter(matrix,n_p*n,MPI_DOUBLE,sub_matrix,n_p*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(x,m,MPI_DOUBLE,0,MPI_COMM_WORLD);
    double * y_p = calloc(n_p,sizeof(double ) );
    double * y1 = calloc(n_p*world_size,sizeof(double ) );
    y_p=blas2(sub_matrix,limite,m,x);
    y_p= add_vector(y_p,1,x,1,0,n_p);
    MPI_Allgather(y_p,n_p,MPI_DOUBLE,y1,n_p,
               MPI_DOUBLE,MPI_COMM_WORLD);
    for (int i = 1; i < degre; ++i) {
        y_p=blas2(sub_matrix,limite,m,y1);
        MPI_Allgather(y_p,n_p,MPI_DOUBLE,y1,n_p,
                     MPI_DOUBLE,MPI_COMM_WORLD);

    }
    if(print==1 && rank==0)
    {

       // print_matrix_p(sub_matrix,n,m,"test");
        y1= add_vector(y1,1,x,1,0,m);
        //print_vector(y1,n," y");
        //print_vector(x,n," x");
    }

    return y1;
}
//
double *Classical_Gram_Schmidt(double* x, int deg,int world_size,int rank){

    double *v_p= malloc(sizeof (double )*deg);
    double *v_p_sum= malloc(sizeof (double )*deg);
    int debut,fin;
    double *q=alloc_matrix_p(deg,deg);
    char s[200];
    for(int j=0;j<deg;j++)
    {
        if(rank==0)
        {
            for(int j1=0;j1<deg;j1++)
            {
                v_p[j1] =x[j*deg+j1];
            }
        }
        else
        {
            for(int j1=0;j1<deg;j1++)
            {
                v_p[j1] =0;
            }
        }

        if(j!=0)
        {
        //***********************
        debut=(j/world_size)*rank;
        fin=(j/world_size)*(rank+1);
        if(rank==world_size-1)
        {
            fin=j;
        }
        if(debut==fin && rank!=0)
        {
            for (int i = 0; i < deg; ++i) {
                v_p[i]=0;
            }
        }
        //***********
        for(int k=debut;k<fin;k++)
        {
            double scl = 0.0;
            for (int s=0;s<deg;s++)
            {
                scl = scl + q[k*deg+s]*x[j*deg+s];
            }
            for(int j1=0;j1<deg;j1++)
            {
                v_p[j1] = v_p[j1]-scl*q[k*deg+j1];

            }

        }
        MPI_Allreduce(v_p,v_p_sum,deg,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        }
        else
        {
            for(int j1=0;j1<deg;j1++)
            {
                v_p_sum[j1] =x[j1];
            }
        }


        for(int j1=0;j1<deg;j1++)
        {
            q[j*deg+j1] = v_p_sum[j1]/sqrt(blas1(v_p_sum,v_p_sum,deg));
        }
            //printf("%d %d %d\n",debut,fin,rank);
            //print_matrix_p(q,deg,deg,"Q");


    }
    return q;
}

//
double *Classical_Gram_Schmidt_seq(double* x, int deg){

    double *v = alloc_matrix_p(deg,deg);

    double *q = alloc_matrix_p(deg,deg);

    for(int j=0;j<deg;j++)
    {
        for(int j1=0;j1<deg;j1++)
        {
            v[j*deg+j1] =x[j*deg+j1];
        }
        for(int k=0;k<j;k++)
        {
            double scl = 0.0;
            for (int s=0;s<deg;s++)
            {
                scl = scl + q[k*deg+s]*x[j*deg+s];
            }
            for(int j1=0;j1<deg;j1++)
            {
                v[j*deg+j1] = v[j*deg+j1]-scl*q[k*deg+j1];
            }
        }
        print_vector(&v[j*deg],deg,"vect ");

        for(int j1=0;j1<deg;j1++)
        {
            q[j*deg+j1] = v[j*deg+j1]/sqrt(blas1(&v[j*deg],&v[j*deg],deg));
        }
    }
    return q;
}

//
double *Modified_Gram_Schmidt(double* x, int deg,int world_size,int rank){
    //*****************
    int n_p;
    if(deg%world_size==0)
    {
        n_p=( ( deg - 1 ) - ( ( deg - 1 ) % world_size ) ) / world_size;
    }
    else
    {

        n_p=( deg - ( deg % world_size ) ) / world_size;
    }



    //(81 - 1) / 4 == 20
    n_p=deg - n_p * ( world_size - 1);
    // 81 - ( 20 * 3 ) = 21;
    int limite=n_p;
    if(rank==world_size-1)
    {
        limite=deg - n_p * ( world_size - 1);
        // 81 - 21 * 3 == 18;
    }
    //**************************
    double *v = alloc_matrix_p(n_p*(world_size+3),deg);
    double *v_p=alloc_matrix_p(n_p*2,deg);
    double *q = alloc_matrix_p(deg,deg);

    for(int i=0;i<deg;i++){
        for(int j=0;j<deg;j++){
            v[i*deg+j] = x[i*deg+j];
        }
    }

    for(int j=0;j<deg;j++)
    {

        for(int j1=0;j1<deg;j1++)
        {
            q[j*deg+j1] =v[j*deg+j1]/ sqrt(blas1(&v[j*deg],&v[j*deg],deg));
        }
        if((deg - ( j + 1))>world_size) {
            MPI_Bcast(&q[j*deg],deg,MPI_DOUBLE,0,MPI_COMM_WORLD);
            //***************
            n_p = ((deg - (j + 1)) - ((deg - (j + 1)) % world_size)) / world_size;
            //(81 - 1) / 4 == 20
            n_p = (deg - (j + 1)) - n_p * (world_size - 1);
            // 81 - ( 20 * 3 ) = 21;
            limite = n_p;
            if (rank == world_size - 1) {
                limite = (deg - (j + 1)) - n_p * (world_size - 1);
                // 81 - 21 * 3 == 18;
            }

            //**********************
            //MPI_Scatter(&v[(j+1)*deg], n_p * deg, MPI_DOUBLE, v_p, n_p * deg, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            //sprintf(s,"v_p limite=%d rank=%d j=%d ",(deg - (j + 1)),rank,j);

            //print_matrix_p(v_p,deg,deg,s);
            for (int k = 0; k < limite; k++) {
                double scl = 0.0;
                for (int s = 0; s < deg; s++) {
                    scl = scl + q[j * deg + s] * v_p[k * deg + s];
                }
                for (int j1 = 0; j1 < deg; j1++) {
                    v_p[k * deg + j1] = v_p[k * deg + j1] - scl * q[j * deg + j1];
                }
            }
            MPI_Gather(v_p, n_p * deg, MPI_DOUBLE, &v[(j+1)*deg], n_p * deg,
                  MPI_DOUBLE,0, MPI_COMM_WORLD);

        }
        else
        {
            for (int k = j+1; k < deg; k++) {
                double scl = 0.0;
                for (int s = 0; s < deg; s++) {
                    scl = scl + q[j * deg + s] * v[k * deg + s];
                }
                for (int j1 = 0; j1 < deg; j1++) {
                    v[k * deg + j1] = v[k * deg + j1] - scl * q[j * deg + j1];
                }
            }
        }

    }

    free(q);
    free(v);
    free(v_p);
    return NULL;
}

//
int return_n_p(int n,int world_size,int *limite,int rank)
{
    int n_p;
    if(n%world_size==0)
    {
        n_p=( ( n - 1 ) - ( ( n - 1 ) % world_size ) ) / world_size;
    }
    else
    {
        n_p=( n - ( n % world_size ) ) / world_size;
    }
    n_p= n - n_p * ( world_size - 1);

    if(world_size==0)
    {
        *limite=n;
    }
    if(rank==world_size-1)
    {
        *limite=n - n_p * ( world_size - 1);
        // 81 - 21 * 3 == 18;
    }
    return n_p;
}
/*
int Arnoldi_Modified_Graham_Schmidt(double* A,double * q, double* h, int deg, int deg_k,double *init,int rank,int world_size){





    // initialisation aléatoire de q[0]
    for (int j = 0; j < deg; ++j) {
        //q[0][j]=(rand()/(RAND_MAX+1.))*100;
        q[j]=init[j];

        //q[0][j]=j;
    }

    blas1_p(double *vecteur2,double *vecteur1,double *vect_temp_1,double vect_temp_2,int n)

    double norm = sqrt(blas1(q,q,deg));
    //normalisation de q[0]
    for(int i=0;i<deg;i++){ // normalise q[0]
        q[i] = q[i]/norm;horner_p(matrix,v,limite,n_p,n,n,2,1,world_size,rank);
    }
    int cpt=0;
    for(int k=1;k<deg_k+1;k++){
        q[k*deg] =    blas2(A,&q[(k-1)*deg],deg,deg);
        for(int j=0;j<k;j++){

            h[j][k-1] =blas1(&q[j*deg],q[k*deg],deg);
            for(int i=0;i<deg;i++){
                q[k*deg+i] = q[k*deg+i] - h[j*deg+k-1]*q[j*deg+i];
            }
        }

        h[k*deg+k-1] = sqrt(blas1(&q[k*deg],&q[k*deg],deg));
        if(h[k][k-1]<0.0000000000001)
        {
            printf("ici %d",cpt);
            return k;
        }
        for(int i=0;i<deg;i++){
            q[k*deg+i] = q[k*deg+i]/h[k*deg+k-1];
        }

    }
    return deg_k;
}
*/
double *declare_vectors(unsigned long long m_v)
{
    double *vect1=(double *) malloc(sizeof(double)*m_v);
    for (int i = 0; i < m_v; ++i) {
        //vect1[i]=rand();
        vect1[i]=i;
    }
    return vect1;
}
void print_vectors(double *vect1,double *vect2,unsigned long long m_v)
{
    printf("vector 1 value:[");
    for (int i = 0; i < m_v; ++i) {
        printf("%f,",vect1[i]);
    }
    printf("]\n vector 2 value:[");
    for (int i = 0; i < m_v; ++i) {
        printf("%f,",vect2[i]);
    }
    printf("]\n");
}
int produit_scalaire(double *vect1,double *vect2,unsigned long long m_v,int rank,int world_size)
{
    //length of process vector
    unsigned long long taille_vect_p;
    //vector of each process
    double *vect1_p=(double*) malloc(sizeof(double )*m_v);
    double *vect2_p=(double*) malloc(sizeof(double )*m_v);
    //send data to other process
    MPI_Scatter(vect1,m_v/world_size,MPI_DOUBLE,vect1_p,m_v/world_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatter(vect2,m_v/world_size,MPI_DOUBLE,vect2_p,m_v/world_size,MPI_DOUBLE,0,MPI_COMM_WORLD);

    taille_vect_p=m_v/world_size;
    if(m_v%world_size!=0)
    {

        if(rank==0)
        {
            MPI_Send(&vect1[m_v-m_v%world_size],m_v%world_size,MPI_DOUBLE,world_size-1,0,MPI_COMM_WORLD);
            MPI_Send(&vect2[m_v-m_v%world_size],m_v%world_size,MPI_DOUBLE,world_size-1,0,MPI_COMM_WORLD);
        }
        else if( rank==world_size-1)
        {
            MPI_Recv(&vect1_p[m_v/world_size],m_v%world_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Recv(&vect2_p[m_v/world_size],m_v%world_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            taille_vect_p=taille_vect_p+m_v%world_size;
        }


    }
    //local scaler product
    double sum=0,sum_global;
    for (int i = 0; i < taille_vect_p; ++i) {
        sum=sum+vect1_p[i]*vect2_p[i];
    }
    //sum local
    //printf("sum of process %d=%f\n",rank,sum);
    //global scaler product
    MPI_Reduce(&sum,&sum_global,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    return sum_global;
}
double *declare_matrix(unsigned long long line_length,unsigned long long column_length)
{
    double *vect1=(double *) malloc(sizeof(double)*line_length*column_length);
    for (int i = 0; i < line_length*column_length; ++i) {
        //vect1[i]=rand();
        vect1[i]=i;
    }
    return vect1;
}

double *declare_matrix_column(unsigned long long line_length,unsigned long long column_length)
{
    double *vect1=(double *) malloc(sizeof(double)*line_length*column_length);
    int k=0;
    for (int i = 0; i < column_length; ++i) {
        //vect1[i]=rand();
        for (int j = 0; j < line_length; ++j)
        {
            vect1[i+j*column_length]=k;
            k=k+1;
        }
    }
    return vect1;
}
int main(int argc, char *argv[]) {
    int rank,world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //main vector
    if(argc==2)
    {

        double *vect1,*vect2;
        //length of main vector
        unsigned long long m_v=strtol(argv[1], NULL, 10);

        //initialisation of main vector

        if(rank==0)
        {
            vect1= declare_vectors(m_v);
            vect2= declare_vectors(m_v);
            //print_vectors(vect1,vect2,m_v);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        //produit scalaire
        int sum_global=produit_scalaire(vect1,vect2,m_v,rank,world_size);
        //print the result
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank==0)
        {
            printf("sum global %d \n",sum_global);

        }

    }
    else if(argc==4)
    {
        //decalaration des variable de taille pour le main
        unsigned long long n=strtol(argv[1], NULL, 10)*500;
        unsigned long long m=strtol(argv[2], NULL, 10)*500;
        unsigned long long m_v=strtol(argv[3], NULL, 10);

        //
        int n_p;
        if(n%world_size==0)
        {
            n_p=( ( n - 1 ) - ( ( n - 1 ) % world_size ) ) / world_size;
        }
        else
        {
            n_p=( n - ( n % world_size ) ) / world_size;
        }
        n_p= n - n_p * ( world_size - 1);
        int limite=n_p;
        if(world_size==0)
        {
            limite=n;
        }
        if(rank==world_size-1)
        {
            limite=n - n_p * ( world_size - 1);
            // 81 - 21 * 3 == 18;
        }
        double *matrix= alloc_matrix_p(n_p*world_size,m);
        //double  *sub_matrix= alloc_matrix_p(n_p,m);
        double *v= malloc(sizeof (double )*(n));
        //double *resultat= malloc(sizeof (double )*m);
        if(rank==0)
        {
            printf("%llu %llu %d \n",n,n*m,limite);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    matrix[i*m+j]=m*i+j;
                    v[j]=1;
                }
            }
        }
        clock_t start, end;
        double cpu_time_used;
        start = clock();
        horner_p(matrix,v,limite,n_p,n,n,2,1,world_size,rank);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        if(rank==0)
        {
            FILE *plot= fopen("honer.txt","a");
            char s[200];
            sprintf(s,"%d %lld %lf\n",world_size,n,cpu_time_used);
            fprintf(plot,s);
            fclose(plot);
        }

        //blas 2 para
        //envoi des données
        /*
        MPI_Scatter(matrix,n_p*m,MPI_DOUBLE,sub_matrix,n_p*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Bcast(v,m,MPI_DOUBLE,0,MPI_COMM_WORLD);

        double *result= blas2(sub_matrix,n_p,m,v);
        //partage resultat
        MPI_Allgather(result,n_p,MPI_DOUBLE,resultat,n_p,
                  MPI_DOUBLE,MPI_COMM_WORLD);
        */

    }
    else if(argc==5)
    {

        unsigned long long n=strtol(argv[1], NULL, 10)*500;
        unsigned long long m=strtol(argv[2], NULL, 10)*500;
        unsigned long long n_1=strtol(argv[3], NULL, 10)*500;
        unsigned long long m_1=strtol(argv[4], NULL, 10)*500;

        //
        int n_p;
        if(n%world_size==0)
        {
            n_p=( ( n - 1 ) - ( ( n - 1 ) % world_size ) ) / world_size;
        }
        else
        {
            n_p=( n - ( n % world_size ) ) / world_size;
        }
        n_p= n - n_p * ( world_size - 1);
        int limite=n_p;
        if(world_size==0)
        {
            limite=n;
        }
        if(rank==world_size-1)
        {
            limite=n - n_p * ( world_size - 1);
            // 81 - 21 * 3 == 18;
        }
        double *matrix=alloc_matrix_p(n_p*world_size,m);
        //double *resultat= alloc_matrix_p(n_p*world_size,n_1);
        double *matrix1=alloc_matrix_p(n,m);
        double *sub_matrix=(double*) malloc(sizeof(double)*n*m);
        if(rank==0)
        {
            printf("%llu %llu %d",n_p*world_size*m,n*m,limite);
            for (unsigned long long i = 0; i < n; ++i) {
                for (unsigned long long j = 0; j < m; ++j) {
                    matrix[i*m+j]= (rand()%10);
                }
            }
        }
        clock_t start, end;
        double cpu_time_used;
        start = clock();
        //MPI_Bcast(matrix,m*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
        //MPI_Bcast(matrix1,m*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
        //MPI_Scatter(matrix,n_p*m,MPI_DOUBLE,sub_matrix,n_p*m,MPI_DOUBLE,0,MPI_COMM_WORLD);
        double *result=blas3(sub_matrix,limite,m,matrix1,n_1,m_1);
        //MPI_Gather(result,n_p*m,MPI_DOUBLE,resultat,n_p*m,
          //            MPI_DOUBLE,0,MPI_COMM_WORLD);
        //double *test= write_matrix_p(n,m,0,"",0,0);
        //Classical_Gram_Schmidt(matrix,n,world_size,rank);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        if(rank==0)
        {
            FILE *plot= fopen("blas3.txt","a");
            char s[200];
            sprintf(s,"%d %lld %lf\n",world_size,n,cpu_time_used);
            fprintf(plot,s);
            fclose(plot);
        }
        free(matrix);
        free(sub_matrix);
        free(matrix1);





    }
    MPI_Finalize();
    return 0;
}

