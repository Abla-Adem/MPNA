#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/*
||===================================================================||
||--------------------------INIT FUNCTION----------------------------||
||===================================================================||
*/
double *alloc_matrix(int n,int m);
double *write_matrix(int n,int m);

/*
||===================================================================||
||--------------------------Print Function---------------------------||
||===================================================================||
*/
void print_matrix(double * matrix,int n,int m,char *s) ;
void print_vector(double *vect,int n,char *s);

/*
||===================================================================||
||---------------------Matrice/vector operation----------------------||
||===================================================================||
*/
double *inverse_tri(double *matrix,unsigned long long n);
double *transpose_matrix(double *matrix,int n,int m);
double blas1(double *vecteur2,double *vecteur1,int n);
double *blas3(double * matrix,unsigned long long n,unsigned long long m,double * matrix_1,unsigned long long n1,unsigned long long m1);

/*
||===================================================================||
||------------------------------Methode------------------------------||
||===================================================================||
*/
double *Modified_Gram_Schmidt(double* x, int deg);
double max_sous_diagonal(double *matrix,int n);
double *decomposition_QR(double *matrix,double *R,int n);
double *methode_QR(double *matrix,double *v,int n,double arret);
double *Gauss_Jordan(double *matrix,double *valeur_propre,int n);

