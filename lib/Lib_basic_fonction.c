
#include "Lib_basic_fonction.h"
/*
||===================================================================||
||--------------------------INIT FUNCTION----------------------------||
||===================================================================||
*/

double *alloc_matrix(int n,int m)
{
    double *matrix= malloc(sizeof (double)*n*m);
    return matrix;
}
double *write_matrix(int n,int m)
{
    double *matrix=(double *)malloc(sizeof(double)*n*m);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j) {
            //matrix[i][j]=rand()%INT_MAX;
            matrix[i*m+j]= (rand()%10);
        }



    }
    return matrix;
}
/*
||===================================================================||
||--------------------------Print Function---------------------------||
||===================================================================||
*/

void print_matrix(double * matrix,int n,int m,char *s) {
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

/*
||===================================================================||
||---------------------Matrice/vector operation----------------------||
||===================================================================||
*/

double *inverse_tri(double *matrix,unsigned long long n)
{
    //init var
    double phin=matrix[(n-1)*n+n-1],phin1=1;
    double *phi= malloc(sizeof(double )*(n+2));
    double *theta= malloc(sizeof(double )*(n+2));
    int puisssance;
    double *matrix_inv=(double *)malloc(sizeof(double )*n*n);

    //calcaulate phi
    for (long long i = n-2; i >-1 ; --i) {
        phi[i]=matrix[i*n+i]*phin-matrix[(i+1)*n+i]*matrix[i*n+i+1]*phin1;
        phin1=phin;
        phin=phi[i];

    }
    phi[n-1]=matrix[(n-1)*n+n-1];
    phi[n]=1;

    //calculate theta
    theta[0]=1;
    theta[1]=matrix[0];
    for (unsigned long long i = 2; i < n+1; ++i) {
        theta[i]=theta[i-1]*matrix[(i-1)*n+i-1]-theta[i-2]*matrix[(i-2)*n+i-1]*matrix[(i-1)*n+i-2];
    }
    for (unsigned long long i = 0; i < n; ++i) {
        for (unsigned long long j = 0; j < n; ++j) {
            matrix_inv[i*n+j]=1;
        }
    }

    //calculate the inverse
    for (unsigned long long i = 0; i < n; ++i) {
        for (unsigned long long j = 0; j < n; ++j) {
            puisssance=1*( ((i+j)%2)*(-2) +1 );
            if(i<j)
            {
                for (int k = i; k < j; ++k)
                {
                    matrix_inv[i*n+j]=matrix_inv[i*n+j]*matrix[k*n+k+1];
                }
                matrix_inv[i*n+j]=puisssance*matrix_inv[i*n+j]*theta[i]*phi[j+1]/theta[n];
            }
            else if(i==j)
            {
                matrix_inv[i*n+i]=(double)theta[i]*phi[i+1]/theta[n];

            }
            else
            {
                for (unsigned long long k = j; k < i; ++k) {
                    matrix_inv[i*n+j]=matrix_inv[i*n+j]*matrix[(k+1)*n+k];
                }
                matrix_inv[i*n+j]=puisssance*matrix_inv[i*n+j]*theta[j]*phi[i+1]/theta[n];
            }

        }
    }

    return matrix_inv;
}
double *transpose_matrix(double *matrix,int n,int m)
{
    double *matrix_transpore= alloc_matrix(m,n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            matrix_transpore[j*m+i]=matrix[i*m+j];
        }

    }
    return matrix_transpore;
}
double blas1(double *vecteur2,double *vecteur1,int n)
{
    double resultat=0;
    for (int i = 0; i < n; ++i) {
        resultat=vecteur2[i]*vecteur1[i]+resultat;
    }
    return resultat;
}
double *blas3(double * matrix,unsigned long long n,unsigned long long m,double * matrix_1,unsigned long long n1,unsigned long long m1)
{
    double *result= alloc_matrix(n,n1);


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
||===================================================================||
||------------------------------Methode------------------------------||
||===================================================================||
*/

double *Modified_Gram_Schmidt(double* x, int deg){

    double *v = alloc_matrix(deg,deg);

    double *q = alloc_matrix(deg,deg);

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
        for(int k=j+1;k<deg;k++)
        {
            double scl = 0.0;
            for (int s=0;s<deg;s++)
            {
                scl = scl + q[j*deg+s]*v[k*deg+s];
            }
            for(int j1=0;j1<deg;j1++)
            {
                v[k*deg+j1] = v[k*deg+j1]-scl*q[j*deg+j1];
            }
        }
        //print_matrix(v,deg,deg,"v");
    }
    return q;
}

//Methode qr
double max_sous_diagonal(double *matrix,int n)
{
    double max=0;
    double temp;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            temp=fabs(matrix[i*n+j]);
            if(max<temp)
            {
                max=temp;
            }
        }
    }
    return max;
}
double *decomposition_QR(double *matrix,double *R,int n)
{
    double *tran_matrix= transpose_matrix(matrix,n,n);
    double *q= Modified_Gram_Schmidt(tran_matrix,n);
    //double **tran_q=transpose_matrix(q,n,n);
    double *temp= blas3(q,n,n,tran_matrix,n,n);
    //temp=transpose_matrix(temp,n,n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            R[i*n+j]=temp[i*n+j];
        }
    }
    return q;
}
double *methode_QR(double *matrix,double *v,int n,double arret)
{
    double *q,*vecteur_propre=alloc_matrix(n,n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            vecteur_propre[i*n+j]=0;
        }
        vecteur_propre[i*n+i]=1;
    }
    double *An= alloc_matrix(n,n);
    for (int i = 0; i <n ; ++i) {
        for (int j = 0; j < n; ++j) {
            An[i*n+j]=matrix[i*n+j];
        }
    }

    double cpt=0;
    double *R= alloc_matrix(n,n);
    double max=1;
    double *tran_q;
    while(max>arret)
    {

        q=decomposition_QR(An,R,n);
        tran_q=transpose_matrix(q,n,n);
        vecteur_propre=blas3(vecteur_propre,n,n,q,n,n);

        //print_matrix(q,n,n,"vecteur propre");
        An= blas3(R,n,n,q,n,n);
        //print_matrix(R,n,n,"R");
        //print_matrix(An,n,n,"An");


        max= max_sous_diagonal(An,n);
        if(cpt==100000)
        {
            //print_matrix(An,n,n,"An");
            cpt=max;
            max=0;

        }
        cpt=cpt+1;
        //printf("%lf %d",max,max>arret);

    }
    //printf("%lf \n",cpt-1);
    print_matrix(An,n,n,"An");
    //print_matrix(q,n,n,"Q");
    for (int i = 0; i < n; ++i) {
        v[i]=An[i*n+i];
    }
    vecteur_propre= Gauss_Jordan(matrix,v,n);
    return vecteur_propre;
}
double *Gauss_Jordan(double *matrix,double *valeur_propre,int n)
{
    double *vecteur_propre=(double *) malloc(sizeof (double )*(n*n));
    int max_ind,ind_pivot=0;
    double max_value,max_value_abs,temp;
    double *matrix_temp= alloc_matrix(n,n);
    for (int ind = 0; ind < n; ++ind)
    {
        vecteur_propre[ind*n+n-1]=1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix_temp[i*n+j]=matrix[i*n+j];
            }
            matrix_temp[i*n+i]=matrix_temp[i*n+i]-valeur_propre[ind];

        }
        ind_pivot=0;
        for (int i = 0; i < n-1; ++i)
        {
            max_value=matrix_temp[i*n+ind_pivot];
            max_value_abs= fabs(matrix_temp[i*n+ind_pivot]);
            max_ind=i;
            for (int j = i+1; j < n; ++j) {
                if(max_value_abs< fabs(matrix_temp[j*n+ind_pivot]))
                {
                    max_value_abs=fabs(matrix_temp[j*n+ind_pivot]);
                    max_value=matrix_temp[j*n+ind_pivot];
                    max_ind=j;
                }
            }
            //printf("%d \n",i);
            if(max_value!=0)
            {
            if(max_ind!=i)
            {
                for (int j = ind_pivot; j < n; ++j) {
                    temp=matrix_temp[i*n+j];
                    matrix_temp[i*n+j]=matrix_temp[max_ind*n+j];
                    matrix_temp[max_ind*n+j]=temp;
                }
            }
            for (int j = ind_pivot; j < n; ++j) {
                matrix_temp[i*n+j]=matrix_temp[i*n+j]/max_value;

            }
            //print_matrix(matrix_temp,n,n,"guass");
            for (int k = i+1; k < n; ++k)
            {
                //printf("%d \n",k);
                if(matrix_temp[k*n+ind_pivot]!=0)
                {
                    temp=matrix_temp[k*n+ind_pivot];
                    for (int j = ind_pivot; j < n; ++j) {
                        //printf("%lf,%lf ",matrix_temp[k*n+j],temp);
                        matrix_temp[k*n+j]=matrix_temp[k*n+j]/temp;
                        //printf("%lf\n",matrix_temp[k*n+j]);
                    }
                    //printf("\n");

                    for (int j = ind_pivot; j < n; ++j) {
                        //printf("%lf,%lf ",matrix_temp[k*n+j],matrix_temp[max_ind*n+j]);
                        matrix_temp[k*n+j]=matrix_temp[k*n+j]-matrix_temp[i*n+j];
                        //printf("%lf\n",matrix_temp[k*n+j]);
                    }
                    //printf("\n");

                }
                vecteur_propre[ind*n+i]=-1;
            }
            }
            else
            {
                vecteur_propre[ind*n+i]=0;
            }

            ind_pivot=ind_pivot+1;
        }
        //print_vector(&vecteur_propre[ind*n],n,"avant test");
        if(vecteur_propre[ind*n+n-2]==0)
        {
            vecteur_propre[ind*n+n-1]=0;
        }
        else
        {
            vecteur_propre[ind*n+n-1]=1;
        }
        int nul=0;
        for (int i = n-2; i >-1 ; --i) {
            if(matrix_temp[i*n+i]!=0)
            {
                vecteur_propre[ind*n+i]=0;
                for (int j = i+1; j < n; ++j) {
                    vecteur_propre[ind*n+i]=vecteur_propre[ind*n+i]-matrix_temp[i*n+j]*vecteur_propre[ind*n+j];
                }
                if(vecteur_propre[ind*n+i]==0)
                {
                    nul=nul+1;
                }

            }
            else
            {
                vecteur_propre[ind*n+i]=0;
            }

        }
        if(nul==n-1)
        {
            vecteur_propre[ind*n+0]=1;
        }
        //print_matrix(matrix_temp,n,n,"guass");
    }
    return vecteur_propre;
}

