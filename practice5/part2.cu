#include <cuda_runtime.h>              // Библиотека CUDA для работы с памятью и ядрами
#include <device_launch_parameters.h>  // Определения для threadIdx, blockIdx и т.д.
#include <iostream>                   // Ввод-вывод в C++

#define N 100000                      // Размер очереди и стека
#define THREADS 256                  // Количество потоков в одном блоке

// ---------------- Параллельная очередь ----------------
struct Queue {
    int *data;       // Указатель на массив данных в глобальной памяти
    int head;        // Указатель начала очереди (индекс чтения)
    int tail;        // Указатель конца очереди (индекс записи)
    int capacity;    // Максимальная ёмкость очереди

    __device__ void init(int *buffer, int size) {
        data = buffer;   // Привязываем буфер в глобальной памяти
        head = 0;        // Начальный индекс чтения
        tail = 0;        // Начальный индекс записи
        capacity = size;// Размер очереди
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1); // Атомарно увеличиваем tail и получаем позицию
        if (pos < capacity) {          // Проверка на переполнение
            data[pos] = value;         // Записываем элемент
            return true;               // Успешная вставка
        }
        return false;                  // Очередь переполнена
    }

    __device__ bool dequeue(int *value) {
        int pos = atomicAdd(&head, 1); // Атомарно увеличиваем head
        if (pos < tail) {              // Проверка, что элементы ещё есть
            *value = data[pos];        // Считываем элемент
            return true;               // Успешное извлечение
        }
        return false;                  // Очередь пуста
    }
};

// ---------------- Параллельный стек ----------------
struct Stack {
    int *data;        // Массив в глобальной памяти
    int top;          // Указатель вершины стека
    int capacity;     // Размер стека

    __device__ void init(int *buffer, int size) {
        data = buffer;    // Привязываем буфер
        top = 0;          // Вершина стека
        capacity = size; // Ёмкость
    }

    __device__ bool push(int value) {
        int pos = atomicAdd(&top, 1); // Атомарно увеличиваем top
        if (pos < capacity) {         // Проверка на переполнение
            data[pos] = value;        // Запись элемента
            return true;              // Успешная операция
        }
        return false;                 // Стек переполнен
    }

    __device__ bool pop(int *value) {
        int pos = atomicSub(&top, 1) - 1; // Атомарно уменьшаем top
        if (pos >= 0) {                  // Проверка на пустоту
            *value = data[pos];         // Считываем вершину
            return true;                // Успешное извлечение
        }
        return false;                   // Стек пуст
    }
};

// ---------------- Ядро для очереди ----------------
__global__ void queueKernel(int *buffer, int *out, int n) {
    __shared__ Queue q;                // Очередь в shared-памяти блока

    if (threadIdx.x == 0)              // Только один поток инициализирует структуру
        q.init(buffer, n);             // Инициализация очереди
    __syncthreads();                  // Синхронизация всех потоков

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    if (tid < n)
        q.enqueue(tid);               // Параллельная вставка элементов

    __syncthreads();                  // Ждём завершения всех вставок

    if (tid < n) {
        int val;
        if (q.dequeue(&val))          // Параллельное извлечение
            out[tid] = val;           // Запись результата
    }
}

// ---------------- Ядро для стека ----------------
__global__ void stackKernel(int *buffer, int *out, int n) {
    __shared__ Stack s;               // Стек в shared-памяти

    if (threadIdx.x == 0)
        s.init(buffer, n);            // Инициализация стека
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        s.push(tid);                 // Параллельная операция push

    __syncthreads();

    if (tid < n) {
        int val;
        if (s.pop(&val))             // Параллельная операция pop
            out[tid] = val;
    }
}

int main() {
    int *d_buffer, *d_out;                         // Указатели на память GPU
    cudaMalloc(&d_buffer, N * sizeof(int));       // Выделение памяти под очередь/стек
    cudaMalloc(&d_out, N * sizeof(int));          // Выделение памяти под результат

    dim3 block(THREADS);                          // Конфигурация блока
    dim3 grid((N + THREADS - 1) / THREADS);       // Количество блоков

    cudaEvent_t startQ, stopQ, startS, stopS;     // События для измерения времени
    cudaEventCreate(&startQ);
    cudaEventCreate(&stopQ);
    cudaEventCreate(&startS);
    cudaEventCreate(&stopS);

    // Очередь
    cudaEventRecord(startQ);                      // Начало замера
    queueKernel<<<grid, block>>>(d_buffer, d_out, N); // Запуск ядра очереди
    cudaEventRecord(stopQ);                       // Конец замера
    cudaEventSynchronize(stopQ);                  // Ожидание завершения

    float timeQueue;
    cudaEventElapsedTime(&timeQueue, startQ, stopQ); // Время работы очереди

    // Стек
    cudaEventRecord(startS);
    stackKernel<<<grid, block>>>(d_buffer, d_out, N); // Запуск ядра стека
    cudaEventRecord(stopS);
    cudaEventSynchronize(stopS);

    float timeStack;
    cudaEventElapsedTime(&timeStack, startS, stopS); // Время работы стека

    std::cout << "Часть 2. Параллельные структуры данных на CUDA\n\n";
    std::cout << "Размер: " << N << " элементов\n";
    std::cout << "Время работы очереди (atomic enqueue/dequeue): " << timeQueue << " мс\n";
    std::cout << "Время работы стека (atomic push/pop): " << timeStack << " мс\n\n";

    if (timeQueue < timeStack)
        std::cout << "Очередь работает быстрее стека.\n";
    else
        std::cout << "Стек работает быстрее очереди.\n";

    cudaFree(d_buffer);                          // Освобождение памяти GPU
    cudaFree(d_out);

    return 0;                                   // Завершение программы
}

