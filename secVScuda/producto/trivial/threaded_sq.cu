#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

#define n 6

void fillMatrix(double *w){
  double count = 0;
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      w[i*n+j] = count;
      count++;
    }
  }
}

void print(double *w){
  for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
      printf("%.4lf ", w[i*n+j]);
    }
    printf("\n");
  }
}

__global__
void product(double *d_x, double *d_y, double *d_z){

  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  double sum = 0;
  if ((row < n) && (col < n)){
    for (int i = 0; i < n; i++) sum += d_x[n*row + i] * d_y[i*n+col];
    d_z[row*n+col] = sum;
  }
}

int main(int argc, char const *argv[])
{
  int size = n*n*sizeof(double);

  double *x = (double*)malloc(size);
  double *y = (double*)malloc(size);
  double *z = (double*)malloc(size);

  fillMatrix(x);
  fillMatrix(y);

  clock_t begin, end;
  double time_spent;
  begin = clock();

  double *d_x;
  double *d_y;
  double *d_z;

  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_y, size);
  cudaMalloc((void**)&d_z, size);

  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

  int threads = 16;
  dim3 dimBlock(threads,threads);
  dim3 dimGrid((n+dimBlock.x-1)/dimBlock.x, (n+dimBlock.y-1)/dimBlock.y);

  product<<<dimGrid,dimBlock>>>(d_x, d_y, d_z);

  cudaMemcpy(z,d_z,size,cudaMemcpyDeviceToHost);

  print(x);
  printf("\n");
  print(y);
  printf("\n");
  print(z);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%lf\n", time_spent);

  return 0;
}
