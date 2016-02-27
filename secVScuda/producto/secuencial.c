#include <stdio.h>
#include <stdlib.h>

#define dimiz 10
#define dimmd 10
#define dimde 10

void llenarMatriz(double **w, int li, int lj){
  double count = 0;
  for(int i=0; i<li; i++){
    for(int j=0; j<lj; j++){
      w[i][j] = count;
      count++;
    }
  }
}

void product(double **x, double **y, double**z){
  for(int i=0; i<dimiz; i++){
    for(int j=0; j<dimde; j++){
      double acum=0;
      for(int k=0; k<dimmd; k++){
        acum+=x[i][k]*y[k][j];
      }
      z[i][j]=acum;
    }
  }
}

void print(double **w, int li, int lj){
  for(int i=0; i<li; i++){
    for(int j=0; j<lj; j++){
      printf("%.4lf ", w[i][j]);
    }
    printf("\n");
  }
}

double** Make2DDoubleArray(int arraySizeX, int arraySizeY) {
  double** theArray;
  theArray = (double**) malloc(arraySizeX*sizeof(double*));
  for (int i = 0; i < arraySizeX; i++)
    theArray[i] = (double*) malloc(arraySizeY*sizeof(double));
  return theArray;
}

int main(){


  //double **x = malloc(dimiz * sizeof *x + (dimiz * (dimmd * sizeof **x)));
  //double **y = malloc(dimmd * sizeof *y + (dimmd * (dimde * sizeof **y)));
  //double **z = malloc(dimiz * sizeof *z + (dimiz * (dimde * sizeof **z)));

  double **x = Make2DDoubleArray(dimiz,dimmd);
  double **y = Make2DDoubleArray(dimmd,dimiz);
  double **z = Make2DDoubleArray(dimiz,dimde);

	llenarMatriz(x,dimiz,dimmd);
	llenarMatriz(y,dimmd,dimde);

  print(x,dimiz,dimmd);

	product(x,y,z);

	//print(z,dimiz,dimde);

}
