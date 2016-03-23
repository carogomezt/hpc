#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

#define n 3
#define m 3

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
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


__global__ void product_shared(double* d_x, double* d_y, double* d_z) {

    double CValue = 0;

    int Row = blockIdx.y*m + threadIdx.y;
    int Col = blockIdx.x*m + threadIdx.x;

    __shared__ double As[m][m];
    __shared__ double Bs[m][m];

    for (int k = 0; k < (m + n - 1)/m; k++) {

         if (k*m + threadIdx.x < n && Row < n)   As[threadIdx.y][threadIdx.x] = d_x[Row*n + k*m + threadIdx.x];
         else                                                   As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*m + threadIdx.y < n && Col < n)   Bs[threadIdx.y][threadIdx.x] = d_y[(k*m + threadIdx.y)*n + Col];
         else                                                   Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int i = 0; i < m; ++i) CValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];

         __syncthreads();
    }

    if (Row < n && Col < n) d_z[((blockIdx.y * blockDim.y + threadIdx.y)*n)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
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

  int threads = m;
  dim3 dimBlock(threads,threads);
  dim3 dimGrid((n+dimBlock.x-1)/dimBlock.x, (n+dimBlock.y-1)/dimBlock.y);

  product_shared<<<dimGrid,dimBlock>>>(d_x, d_y, d_z);

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