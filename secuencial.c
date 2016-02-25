#include <stdio.h>
//#include <cuda.h>
#include <time.h>
#include <stdlib.h>

#define lim 10

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

void add(int *x, int *y, int *z){
	for(int i=0; i<lim; i++){
		z[i] = x[i] + y[i];
	}
}

int main(int argc, char const *argv[])
{
  clock_t begin, end;
  double time_spent;
  begin = clock();
  int *x = (int*)malloc(lim*sizeof(int));
  int *y = (int*)malloc(lim*sizeof(int));
  int *z = (int*)malloc(lim*sizeof(int));

  llenarVector(x);
  llenarVector(y);

  add(x, y, z);

  //print(z);

  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("%lf\n", time_spent);
  return 0;
}
