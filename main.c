#include <stdio.h>
#include <stdlib.h>
#include "Lib_basic_fonction.h"
int main(int argc, char *argv[]) {
    unsigned long long n;

    n=strtoul(argv[1], NULL, 10);
    double *matrix=(double *)malloc(sizeof(double)*n*n);
    matrix=write_matrix(n,n);
    /*
    matrix[0]=4;
    matrix[1]=6;
    matrix[2]=6;
    matrix[3]=1;
    matrix[4]=3;
    matrix[5]=2;
    matrix[6]=-1;
    matrix[7]=-5;
    matrix[8]=-2;
     */
    print_matrix(matrix,n,n,"matrice a inveser");
    double *matrix_inv= inverse_tri(matrix,n);
    double *matrix_result=(double *)malloc(sizeof(double )*n*n);
    for (unsigned long long i = 0; i < n; ++i) {
        for (unsigned long long j = 0; j < n; ++j) {
            matrix_result[i*n+j]=0;
        }
    }
    for (unsigned long long i = 0; i < n; ++i) {
        for (unsigned long long j = 0; j < n; ++j) {
            for (unsigned long long k = 0; k < n; ++k) {
                matrix_result[i*n+j]=matrix[i*n+k]*matrix_inv[k*n+j]+matrix_result[i*n+j];
            }
        }
    }
    double *v= malloc(sizeof(double)*n*n);
    double *test=methode_QR(matrix,v,n,0.00000001);
    print_matrix(test,n,n,"vecteur propre");
    print_vector(v,n,"valeur propres");






}
