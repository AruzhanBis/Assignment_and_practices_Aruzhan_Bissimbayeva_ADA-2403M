#include <cuda_runtime.h>        // Подключение CUDA Runtime API для работы с GPU
#include <iostream>              // Библиотека для вывода информации в консоль
#include <vector>                // STL-контейнер vector для хранения массивов
#include <omp.h>                 // Поддержка OpenMP для параллельных вычислений на CPU
#include <chrono>                // Библиотека для измерения времени на CPU

// CUDA-ядро: умножает каждый элемент массива на 2
__global__ void multiplyKernel(int *data, int N) {
    // Вычисление глобального индекса потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка выхода за границы массива
    if(idx < N)
        data[idx] *= 2;          // Умножение элемента массива на 2
}

int main() {
    int N = 1000000;             // Размер массива (1 миллион элементов)

    // === Подготовка данных ===
    // Три копии массива для разных вариантов обработки
    std::vector<int> data_cpu(N, 1);     // Массив для CPU-вычислений
    std::vector<int> data_gpu = data_cpu; // Массив для GPU-вычислений
    std::vector<int> data_hybrid = data_cpu; // Массив для гибридной обработки


    // ===1. Обработка ТОЛЬКО на CPU ===


    // Запуск таймера CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // Параллельная обработка массива на CPU с использованием OpenMP
    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        data_cpu[i] *= 2;        // Каждый элемент умножается на 2

    // Остановка таймера CPU
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // Вычисление времени выполнения CPU в миллисекундах
    double cpu_time =
        std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // === 2. Обработка ТОЛЬКО на GPU ===

    int *d_data;                 // Указатель на память GPU

    // Выделение памяти на GPU под массив
    cudaMalloc(&d_data, N * sizeof(int));

    // Копирование данных с CPU в память GPU
    cudaMemcpy(d_data, data_gpu.data(), N * sizeof(int),
               cudaMemcpyHostToDevice);

    // Настройка конфигурации CUDA
    dim3 blockSize(256);         // 256 потоков в одном блоке
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x); // Количество блоков

    // CUDA-события для точного измерения времени на GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Начало замера времени GPU
    cudaEventRecord(start);

    // Запуск CUDA-ядра
    multiplyKernel<<<numBlocks, blockSize>>>(d_data, N);

    // Завершение замера времени
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Ожидание завершения GPU

    // Получение времени выполнения GPU
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Копирование результатов обратно на CPU (для корректности)
    cudaMemcpy(data_gpu.data(), d_data, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    // === 3. ГИБРИДНАЯ обработка CPU + GPU ===

    int split = N / 2;           // Точка разделения массива (50% / 50%)

    int *d_hybrid;               // Указатель на память GPU для гибридной части

    // Выделяем память на GPU только для второй половины массива
    cudaMalloc(&d_hybrid, split * sizeof(int));

    // Копируем вторую половину массива на GPU
    cudaMemcpy(d_hybrid, data_hybrid.data() + split, split * sizeof(int),
               cudaMemcpyHostToDevice);

    // Начало замера гибридного времени
    cudaEventRecord(start);

    // GPU обрабатывает вторую половину массива
    multiplyKernel<<<numBlocks, blockSize>>>(d_hybrid, split);

    // CPU параллельно обрабатывает первую половину массива
    #pragma omp parallel for
    for(int i = 0; i < split; i++)
        data_hybrid[i] *= 2;

    // Ожидание завершения GPU
    cudaDeviceSynchronize();

    // Копирование результатов GPU обратно в общий массив
    cudaMemcpy(data_hybrid.data() + split, d_hybrid, split * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Завершение замера гибридного времени
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Получение времени гибридной обработки
    float hybrid_time;
    cudaEventElapsedTime(&hybrid_time, start, stop);

    // === ВЫВОД РЕЗУЛЬТАТОВ ===

    std::cout << "Задание 4: Анализ производительности\n";
    std::cout << "1. Время обработки массива на CPU: " << cpu_time << " мс\n";
    std::cout << "2. Время обработки массива на GPU: " << gpu_time << " мс\n";
    std::cout << "3. Время гибридной обработки CPU + GPU: "
              << hybrid_time << " мс\n";

    // Освобождение памяти GPU
    cudaFree(d_data);
    cudaFree(d_hybrid);

    return 0;                    // Завершение программы
}
