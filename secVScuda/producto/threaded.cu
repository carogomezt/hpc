#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

#define a 2
#define b 3
#define c 2

void fillMatrix(double *w, int li, int lj){
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
void product(double *d_x, double *d_y, double *d_z){

  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  double sum = 0;
  if ((row < a) && (col < c)){
    for (int i = 0; i < b; i++) sum += d_x[b*row + i] * d_y[i*c+col];
    d_z[row*c+col] = sum;
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

  fillMatrix(x,a,b);
  fillMatrix(y,b,c);

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

  int threads = 16;
  dim3 dimBlock(threads,threads);
  dim3 dimGrid((c+dimBlock.x-1)/dimBlock.x, (a+dimBlock.y-1)/dimBlock.y);

  product<<<dimGrid,dimBlock>>>(d_x, d_y, d_z);

  cudaMemcpy(z,d_z,size3,cudaMemcpyDeviceToHost);

  print(x,a,b);
  printf("\n");
  print(y,b,c);
  printf("\n");
  print(z,a,c);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%lf\n", time_spent);

  return 0;
}
