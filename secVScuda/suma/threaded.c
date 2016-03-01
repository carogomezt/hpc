#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define lim 99
#define threads 10

void print(int *w){
	for(int i=0; i<lim; i++){
		printf("%d\n", w[i]);
	}
}

void llenarVector(int *w){
	for(int i=0; i<lim; i++){
		w[i]=i;
	}
}

__global__
void add(int *d_x, int *d_y, int *d_z){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<lim){
		d_z[i] = d_x[i] + d_y[i];
	}
}

int main(int argc, char const *argv[])
{
	
	int *x = (int*)malloc(lim*sizeof(int));
	int *y = (int*)malloc(lim*sizeof(int));
	int *z = (int*)malloc(lim*sizeof(int));

	llenarVector(x);
	llenarVector(y);

	int *d_x, *d_y, *d_z;

	clock_t begin, end;
	double time_spent;
	begin = clock();
	
	cudaMalloc((void**)&d_x, lim*sizeof(int));
	cudaMalloc((void**)&d_y, lim*sizeof(int));
	cudaMalloc((void**)&d_z, lim*sizeof(int));

	cudaMemcpy(d_x, x, lim*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, lim*sizeof(int), cudaMemcpyHostToDevice);

	int blocks = ceil((float)lim/threads);
	
	//printf("%d",blocks);
	
	add<<<blocks,threads>>>(d_x, d_y, d_z);

	cudaMemcpy(z, d_z, lim*sizeof(int), cudaMemcpyDeviceToHost);

	//print(z);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("%lf\n", time_spent);

	free(x);
	free(y);
	free(z);

	

	return 0;
}

//Combination threads and blocmks

// i = blockIdx.x * blockDim.x + threadIdx.x;

