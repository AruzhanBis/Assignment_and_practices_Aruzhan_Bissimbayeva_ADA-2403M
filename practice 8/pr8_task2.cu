#include <cuda_runtime.h>           // Подключение CUDA Runtime API
#include <iostream>                 // Ввод-вывод
#include <vector>                   // Контейнер vector

__global__ void multiplyKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    if(idx < N) data[idx] *= 2;     // Умножение элемента массива на 2
}

int main() {
    int N = 1000000;                // Размер массива
    std::vector<int> h_data(N, 1);  // Хост-массив, заполненный единицами

    int *d_data;                    // Указатель на память GPU
    cudaMalloc(&d_data, N * sizeof(int)); // Выделение памяти на GPU
    cudaMemcpy(d_data, h_data.data(), N * sizeof(int),
               cudaMemcpyHostToDevice);   // Копирование данных CPU → GPU

    dim3 blockSize(256);            // Количество потоков в блоке
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x); // Количество блоков

    cudaEvent_t start, stop;        // CUDA-события для замера времени
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);         // Начало замера времени GPU
    multiplyKernel<<<numBlocks, blockSize>>>(d_data, N); // Запуск ядра CUDA
    cudaEventRecord(stop);          // Конец замера времени
    cudaEventSynchronize(stop);     // Ожидание завершения GPU

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop); // Время выполнения ядра

    cudaMemcpy(h_data.data(), d_data, N * sizeof(int),
               cudaMemcpyDeviceToHost); // Копирование GPU → CPU

    std::cout << "Задание 2: Реализация обработки массива на GPU с использованием CUDA\n";
    std::cout << "1. Данные успешно скопированы на GPU\n";
    std::cout << "2. CUDA-ядро выполнило умножение элементов массива на 2\n";
    std::cout << "3. Время выполнения на GPU: " << gpu_time << " мс\n";

    std::cout << "Первые 5 элементов массива после обработки: ";
    for(int i = 0; i < 5; i++) std::cout << h_data[i] << " ";
    std::cout << std::endl;

    cudaFree(d_data);               // Освобождение памяти GPU
    return 0;                       // Завершение программы
}
