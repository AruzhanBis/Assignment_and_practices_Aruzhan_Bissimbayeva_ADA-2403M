#include <iostream>                 // Подключение библиотеки ввода-вывода
#include <vector>                   // Подключение контейнера vector
#include <omp.h>                    // Подключение OpenMP для параллельных вычислений
#include <chrono>                   // Подключение библиотеки для замера времени

int main() {
    int N = 1000000;                // Размер массива
    std::vector<int> data(N, 1);    // Создание массива и инициализация значением 1

    auto start = std::chrono::high_resolution_clock::now(); // Начало замера времени

    #pragma omp parallel for        // Параллельный цикл OpenMP
    for(int i = 0; i < N; i++) {    // Проход по всем элементам массива
        data[i] *= 2;               // Умножение каждого элемента на 2
    }

    auto end = std::chrono::high_resolution_clock::now();   // Конец замера времени
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count(); // Время в мс

    std::cout << "Задание 1: Реализация обработки массива на CPU с использованием OpenMP\n";
    std::cout << "1. Массив размером N = " << N << " успешно создан\n";
    std::cout << "2. Каждый элемент массива умножен на 2 с использованием OpenMP\n";
    std::cout << "3. Время выполнения на CPU: " << elapsed_ms << " мс\n";

    std::cout << "Первые 5 элементов массива после обработки: ";
    for(int i = 0; i < 5; i++) std::cout << data[i] << " "; // Проверка корректности
    std::cout << std::endl;

    return 0;                       // Завершение программы
}
