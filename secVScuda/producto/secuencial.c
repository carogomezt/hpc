#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define a 1000
#define b 3000
#define c 5000

void llenarMatriz(double *w, int li, int lj){
  double count = 0;
  for(int i=0; i<li; i++){
    for(int j=0; j<lj; j++){
      w[i*li+j] = count;
      count++;
    }
  }
}

void product(double *x, double *y, double *z){
  for(int i=0; i<a; i++){
    for(int j=0; j<c; j++){
      double acum=0;
      for(int k=0; k<b; k++){
        acum+=x[i*a+k]*y[k*b+j];
      }
      z[i*a+j]=acum;
    }
  }
}

void print(double *w, int li, int lj){
  for(int i=0; i<li; i++){
    for(int j=0; j<lj; j++){
      printf("%.4lf ", w[i*li+j]);
    }
    printf("\n");
  }
}

int main(){

  //double **x = malloc(a * sizeof *x + (a * (b * sizeof **x)));
  //double **y = malloc(b * sizeof *y + (b * (c * sizeof **y)));
  //double **z = malloc(a * sizeof *z + (a * (c * sizeof **z)));

  int size1 = a*b*sizeof(double);
  int size2 = b*c*sizeof(double);
  int size3 = a*c*sizeof(double);

  double *x = (double*)malloc(size1);
  double *y = (double*)malloc(size2);
  double *z = (double*)malloc(size3);

  llenarMatriz(x,a,b);
  llenarMatriz(y,b,c);

  clock_t begin, end;
  double time_spent;
  begin = clock();

  product(x,y,z);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%lf\n", time_spent);

  //print(z,a,c);

  //print(z,a,c);

}
