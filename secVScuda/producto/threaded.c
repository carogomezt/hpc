#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define a 1000
#define b 3000
#define c 5000
#define 

void llenarMatriz(double *w, int li, int lj){
  double count = 0;
  for(int i=0; i<li; i++){
    for(int j=0; j<lj; j++){
      w[i*li+j] = count;
      count++;
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

int main(int argc, char const *argv[])
{
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

  cudaMalloc((void**)&d_x, size1);
  cudaMalloc((void**)&d_y, size2);
  cudaMalloc((void**)&d_z, size3);

  cudaMemcpy(d_x, x, size1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size2, cudaMemcpyHostToDevice);

  product

  return 0;
}