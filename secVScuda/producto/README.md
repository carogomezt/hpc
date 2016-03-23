# Comparacion entre una implementacion secuencial (CPU), una paralela (CUDA), y una paralela usando Shared Memory (CUDA) para multiplicar 2 matrices

Uno de los algoritmos que mejor explota las capacidades computacionales de una GPU, es la multiplicacion matricial, debido a que las operaciones que se realizan son las mismas, solo que en diferentes sectores de memoria, ademas de que son altamente independientes.

En esta comparativa, se exponen tres algoritmos para la multiplicacion de matrices, uno implementado de manera secuencial en CPU, otro paralelizando con una GPU, y un ultimo (tambien en GPU) optimizado mediante el uso de memoria compartida.

Especificaciones:

- Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz
- DUAL SLI NVIDIA GPU GeForce GTX 780
- 16 GB RAM

**Implementacion secuencial:** Para la implementacion secuencial se utiliza `malloc` y `free`.

**Implementacion CUDA:** La misma gestion de memoria para la implementacion secuencial + `cudaMalloc` y `cudaMemcpy` para manejar la memoria del dispositivo.

**Implementacion CUDA & Shared Memory:** La misma gestion de memoria para la implementacion CUDA + `__shared__` como decorator para declarar la memoria compartida en el device.

## Pruebas

Para las pruebas se utiliza un dataset de vectores que varian en 100, 1000, 10000, 100000, 1000000 y 10000000 posiciones. Cada prueba fue ejecutada 10 veces para disminuir el ruido. Al final se analiza el factor de aceleracion en base a los tiempos de ejecucion en cada algoritmo. Se anexa un conjunto de pruebas mas usando 1024 hilos (usando la implementacion paralela), contrastando con los 32 hilos de la implementacion original.

**Implementacion secuencial:** Ninguna mencion especial.

**Implementacion CUDA:** Se utilizan 32 hilos por bloque bidimiensional para ejecutar el proceso.

**Implementacion CUDA & Shared Memory:** Se utilizan 32 hilos por bloque bidimiensional para ejecutar el proceso. El tamaño de las 2 tiles es de 32*32 posiciones igualmente.

## Resultados

En la siguiente tabla se muestran los promedios para cada una de las 6 pruebas en las 3 implementaciones:

### Secuencial
| Tamaño (n) | Media (s)  |
| -----------| ---------- |
| 100        | 0.000001   |
| 1000       | 0.000005   |
| 10000      | 0.000049   |
| 100000     | 0.000417   |
| 1000000    | 0.003857   |
| 10000000   | 0.026056   |
| 10000000   | 0.026056   |
| 10000000   | 0.026056   |
| 10000000   | 0.026056   |
| 10000000   | 0.026056   |

### CUDA 32 Threads
| Tamaño (n) | Media (s)  |
| -----------| ---------- |
| 100        | 0.093371   |
| 1000       | 0.094815   |
| 10000      | 0.088516   |
| 100000     | 0.093428   |
| 1000000    | 0.099781   |
| 10000000   | 0.118817   |
| 10000000   | 0.118817   |
| 10000000   | 0.118817   |
| 10000000   | 0.118817   |
| 10000000   | 0.118817   |


### CUDA 1024 Threads
| Tamaño (n) | Media (s)  |
| -----------| ---------- |
| 100        | 0.094012   |
| 1000       | 0.098624   |
| 10000      | 0.093407   |
| 100000     | 0.099471   |
| 1000000    | 0.101861   |
| 10000000   | 0.117360   |
| 10000000   | 0.117360   |
| 10000000   | 0.117360   |
| 10000000   | 0.117360   |
| 10000000   | 0.117360   |

Los resultados se condensan en el siguiente grafico:

![alt tag](graph.jpg)

## Conclusiones

- Con base a los resultados obtenidos, se puede concluir que, en general, para la suma de vectores presenta un mejor desempeño la implementacion en CPU secuencial que la codificacion en CUDA.

- La transferencia de datos a traves del PCI Express representa la mayor parte del consumo de tiempo en la implementacion paralela con GPU.

- La transferencia de datos a traves del PCI Express representa la mayor parte del consumo de tiempo en la implementacion paralela con GPU, y por tanto, la suma vectorial no explota al maximo las capcidades de la computacion en paralelo

- Cuanta mayor cantidad de operaciones sea posible acelerar en GPU, y cuanta menor memoria sea necesaria transferir, mejora el desempeño de la GPU
