#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

#define a 3
#define b 5
#define c 4

void llenarMatriz(double *w, int li, int lj){
  double count = 0;
  for(int i=0; i<li; i++){
    for(int j=0; j<lj; j++){
      w[i*lj+j] = count;
      count++;
    }
  }
}

void print(double *w, int li, int lj){
  for(int i=0; i<li; i++){
    for(int j=0; j<lj; j++){
      printf("%.4lf ", w[i*lj+j]);
    }
    printf("\n");
  }
}

__global__
void add(double *d_x, double *d_y, double *d_z){


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

  double *d_x = (double*)malloc(size1);
  double *d_y = (double*)malloc(size2);
  double *d_z = (double*)malloc(size3);

  cudaMalloc((void**)&d_x, size1);
  cudaMalloc((void**)&d_y, size2);
  cudaMalloc((void**)&d_z, size3);

  cudaMemcpy(d_x, x, size1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size2, cudaMemcpyHostToDevice);

//  product<<<,>>>

  for(int i=0; i<a*b; i++){
    printf("%.4lf\n", x[i]);
  }

  print(x,a,b);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  //printf("%lf\n", time_spent);


  return 0;
}
