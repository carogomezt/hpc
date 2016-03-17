#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

#define n 10
#define m 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

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

__global__ void product(double* d_x, double* d_y, double* d_z){
  __shared__ float Mds[m][m];
  __shared__ float Nds[m][m];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * m + ty;
  int Col = bx * m + tx;
  float Pvalue = 0;
  for (int i = 0; i < n/m; ++i) {
      Mds[ty][tx] = d_x[Row*n + (i*m + tx)];
      Nds[ty][tx] = d_y[Col + (i*m + ty)*n];
      __syncthreads();
      for (int k = 0; k < m; ++k)
        Pvalue += Mds[ty][k] * Nds[k][tx];
      __syncthreads();
    }
  d_z[Row*n+Col] = Pvalue;
}

int main(int argc, char const *argv[])
{
  int size1 = n*n*sizeof(double);
  int size2 = n*n*sizeof(double);
  int size3 = n*n*sizeof(double);

  double *x = (double*)malloc(size1);
  double *y = (double*)malloc(size2);
  double *z = (double*)malloc(size3);

  fillMatrix(x,n,n);
  fillMatrix(y,n,n);

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
  dim3 dimGrid((n+dimBlock.x-1)/dimBlock.x, (n+dimBlock.y-1)/dimBlock.y);

  product<<<dimGrid,dimBlock>>>(d_x, d_y, d_z);

  cudaMemcpy(z,d_z,size3,cudaMemcpyDeviceToHost);

  print(x,n,n);
  printf("\n");
  print(y,n,n);
  printf("\n");
  print(z,n,n);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%lf\n", time_spent);

  return 0;
}

