#include <cuda_runtime.h>           // CUDA Runtime
#include <iostream>                 // Ввод-вывод
#include <vector>                   // Контейнер vector
#include <omp.h>                    // OpenMP
#include <chrono>                   // Замер времени

__global__ void multiplyKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Индекс потока
    if(idx < N) data[idx] *= 2;     // Умножение элемента на 2
}

int main() {
    int N = 1000000;                // Размер массива
    std::vector<int> data(N, 1);    // Массив, заполненный единицами

    int split = N / 2;              // Точка разделения массива

    int *d_data;                    // Указатель на GPU-память
    cudaMalloc(&d_data, split * sizeof(int)); // Выделение памяти для половины массива
    cudaMemcpy(d_data, data.data() + split, split * sizeof(int),
               cudaMemcpyHostToDevice);       // Копирование второй половины на GPU

    dim3 blockSize(256);            // Размер блока
    dim3 numBlocks((split + blockSize.x - 1) / blockSize.x); // Кол-во блоков

    cudaEvent_t start, stop;        // CUDA-события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);         // Начало замера времени

    multiplyKernel<<<numBlocks, blockSize>>>(d_data, split); // GPU обрабатывает вторую половину

    #pragma omp parallel for        // CPU обрабатывает первую половину
    for(int i = 0; i < split; i++) {
        data[i] *= 2;
    }

    cudaDeviceSynchronize();        // Ожидание завершения GPU
    cudaMemcpy(data.data() + split, d_data, split * sizeof(int),
               cudaMemcpyDeviceToHost); // Копирование результата GPU → CPU

    cudaEventRecord(stop);          // Конец замера времени
    cudaEventSynchronize(stop);
    float hybrid_time;
    cudaEventElapsedTime(&hybrid_time, start, stop); // Общее время

    std::cout << "Задание 3: Гибридная обработка массива CPU + GPU\n";
    std::cout << "1. Первая половина массива обработана на CPU\n";
    std::cout << "2. Вторая половина массива обработана на GPU\n";
    std::cout << "3. Общее время гибридной обработки: " << hybrid_time << " мс\n";

    std::cout << "Первые 5 элементов массива после обработки: ";
    for(int i = 0; i < 5; i++) std::cout << data[i] << " ";
    std::cout << std::endl;

    cudaFree(d_data);               // Освобождение памяти GPU
    return 0;
}
