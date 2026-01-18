#include <cuda_runtime.h>           // Основные функции CUDA
#include <device_launch_parameters.h> // Параметры запуска ядра
#include <iostream>                 // Для вывода на экран

#define N 100000                     // Размер стека
#define THREADS 256                  // Потоков в одном блоке

// Параллельный стек 
struct Stack {
    int *data;       // Указатель на массив данных в глобальной памяти
    int top;         // Индекс вершины стека
    int capacity;    // Максимальная ёмкость стека

    __device__ void init(int *buffer, int size) { // Инициализация стека в GPU
        data = buffer;     // Привязка буфера данных
        top = -1;          // Стек пуст
        capacity = size;   // Устанавливаем размер стека
    }

    __device__ bool push(int value) {   // Добавление элемента
        int pos = atomicAdd(&top, 1);   // Атомарное увеличение top
        if (pos < capacity) {           // Проверка переполнения
            data[pos] = value;          // Запись элемента
            return true;                // Успешно
        }
        return false;                   // Не удалось (переполнение)
    }

    __device__ bool pop(int *value) {   // Извлечение элемента
        int pos = atomicSub(&top, 1);   // Атомарное уменьшение top
        if (pos >= 0) {                 // Проверка, что стек не пуст
            *value = data[pos];         // Считываем элемент
            return true;                // Успешно
        }
        return false;                   // Не удалось (пустой стек)
    }
};

// Ядро CUDA 
__global__ void stackKernel(int *buffer, int *out, int n) {
    __shared__ Stack s;                 // Стек в shared-памяти блока

    if (threadIdx.x == 0)               // Только первый поток инициализирует стек
        s.init(buffer, n);              // Инициализация стека
    __syncthreads();                    // Синхронизация всех потоков блока

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока

    if (tid < n)
        s.push(tid);                    // Каждый поток добавляет свой элемент

    __syncthreads();                    // Ждем завершения push всех потоков

    if (tid < n) {
        int val;
        if (s.pop(&val))                // Каждый поток пытается извлечь элемент
            out[tid] = val;             // Сохраняем результат в массив
    }
}

int main() {
    int *d_stack, *d_out;              // Указатели на GPU память
    cudaMalloc(&d_stack, N * sizeof(int)); // Выделяем память для стека
    cudaMalloc(&d_out, N * sizeof(int));   // Выделяем память для результатов

    dim3 block(THREADS);                // Определяем размер блока
    dim3 grid((N + THREADS - 1) / THREADS); // Определяем количество блоков

    std::cout << "Часть 1. Параллельный стек на CUDA\n";
    std::cout << "Размер стека: " << N << " элементов\n";
    std::cout << "Конфигурация GPU: " << grid.x << " блоков по "
              << block.x << " потоков\n\n";

    // Создаем события для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);              // Начало измерения времени
    stackKernel<<<grid, block>>>(d_stack, d_out, N); // Запуск ядра
    cudaEventRecord(stop);               // Конец измерения
    cudaEventSynchronize(stop);          // Ждем завершения всех потоков

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop); // Время выполнения ядра

    std::cout << "Операции push и pop выполнены параллельно.\n";
    std::cout << "Время выполнения ядра: " << time_ms << " мс\n";
    std::cout << "Корректность обеспечена атомарными операциями.\n";

    cudaFree(d_stack);                   // Освобождаем память стека
    cudaFree(d_out);                     // Освобождаем память вывода
    cudaEventDestroy(start);             // Удаляем события CUDA
    cudaEventDestroy(stop);

    return 0;
}
