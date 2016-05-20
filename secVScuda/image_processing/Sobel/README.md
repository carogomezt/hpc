# Comparacion entre una implementacion secuencial (CPU), una paralela (CUDA), y una paralela usando Shared Memory (CUDA) para multiplicar 2 matrices

El tratamiento de imagenes es una de las aplicaciones en las que mas se emplean las GPU, debido a su capacidad de procesar matrices rapidamente y liberando a la CPU de esta carga.

Se puede evidenciar en reproductores de video, juegos de alta exigencia grafica y programas de edicion visual, que uno de los requisitos que suele aparecer es que la maquina tenga equipada una GPU.

En esta comparativa, se hace un analisis sobre un algoritmo secuencal y tres algoritmos sobre CUDA para efectuar el filtro de sobel.

Especificaciones:

- Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz
- DUAL SLI NVIDIA GPU GeForce GTX 780
- 16 GB RAM

**Implementacion secuencial:** Para la implementacion secuencial se utiliza `malloc` y `free`, ademas de las funciones de `OpenCV`.

**Implementacion CUDA:** La misma gestion de memoria para la implementacion secuencial + `cudaMalloc` y `cudaMemcpy` para manejar la memoria del dispositivo.

**Implementacion CUDA & Shared Memory:** La misma gestion de memoria para la implementacion CUDA + `__shared__` y `__constant__` como decorator para declarar la memoria compartida en el device.

**Implementacion CUDA & Caché:** La misma gestion de memoria para la implementacion Shared + `Cache`

## Pruebas

Para las pruebas se utiliza un dataset de 9 imagenes que oscilan entre 1 y 4 MPx y una ultima imagen que mide 70MPx. Cada prueba se ejecutó cien veces para disminuir el ruido en os experimentos --se anexa su respaldo estadistico--


## Resultados

En la siguiente tabla se muestran los promedios para cada una de las 10 pruebas en las 4 implementaciones:

### Secuencial
| Tamaño (px)   | Media (s)    |
| ------------- | ------------ |
| 157360        | 0,01463451   |
| 228000        | 0,02104237   |
| 298820        | 0,02743887   |
| 395200        | 0,03623523   |
| 475410        | 0,04367265   |
| 522000        | 0,04788549   |
| 583440        | 0,05348228   |
| 612480        | 0,05622632   |
| 627000        | 0,05749262   |
| 75659406      | 6,95359620   |


### CUDA Global
| Tamaño (px)   | Media (s)    |
| ------------- | ------------ |
| 157360        | 0,00066425   |
| 228000        | 0,00093569   |
| 298820        | 0,00120806   |
| 395200        | 0,00154624   |
| 475410        | 0,00181034   |
| 522000        | 0,00195562   |
| 583440        | 0,00214262   |
| 612480        | 0,00223046   |
| 627000        | 0,00228171   |
| 75659406      | 0,22500594   |


### CUDA Shared
| Tamaño (px)   | Media (s)    |
| ------------- | ------------ |
| 157360        | 0,00069182   |
| 228000        | 0,00097519   |
| 298820        | 0,00125898   |
| 395200        | 0,00161879   |
| 475410        | 0,00190521   |
| 522000        | 0,00205708   |
| 583440        | 0,00226021   |
| 612480        | 0,00235174   |
| 627000        | 0,00240582   |
| 75659406      | 0,19285223   |


### CUDA Caché
| Tamaño (px)   | Media (s)    |
| ------------- | ------------ |
| 157360        | 0,00069201   |
| 228000        | 0,00095560   |
| 298820        | 0,00124030   |
| 395200        | 0,00160307   |
| 475410        | 0,00188192   |
| 522000        | 0,00373328   |
| 583440        | 0,00220135   |
| 612480        | 0,00233893   |
| 627000        | 0,00236364   |
| 75659406      | 0,18326154   |

Los resultados se condensan en los siguiente grafico:

### Algoritmos paralelos
![alt tag](graph.png)

### Algoritmos paralelos y secuencial
![alt tag](graph2.png)

## Conclusiones

- Con base a los resultados obtenidos, se puede concluir que, en general, para la aplicar el filtro de sobel presenta un mejor desempeño cualquiera de las dos implementaciones paralelas en comparacion a la implementacion secuencial.

- La transferencia de datos a traves del PCI Express representa la mayor parte del consumo de tiempo en la implementacion paralela con GPU. A pesar de eso, una pequeña porcion de datos es utlilizada en multiples operaciones paralelizables (en contraste a la suma), lo cual permite que dicho costo de transferencia sea compensado ampliamente por el ahorro en tiempo de computo de la GPU frente a la CPU.

- En general, la implementacion usando Cache representa una mejora en tiempo, con respecto a su competidora nativa con Global y shared Memory. Sin embargo en ocasiones, al superponer las capas de memoria compartida, da lugar a que se generen tiles completas para procesar solo pequeñas porciones de la matriz original, caso en el cual se desperdicia tiempo de computo. Esto es que, la optimizacion con tiles funciona mejor cuando el tamaño de los tiles se ajusta a bien al tamaño total de la matriz.

- Cuanta mayor cantidad de operaciones sea posible acelerar en GPU, y cuanta menor memoria sea necesaria transferir, mejor es el desempeño de la GPU.

- Existe un comportamiento muy parejo y lineal entre la cantidad de pixeles y el tiempo que tarda cualquiera de los algoritmos al procesarlo. Al expresarse los pixeles como una magnitud total width*height, y no como una medicion cuadrada, se aprecia que el tiempo aumenta de forma lineal. Al expresarse como una medicion cuadrada, se apreciara una parabola (debido al aumento de pixeles)