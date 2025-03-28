#!/bin/bash

# компиляция программы
nvcc -o strassen2 str2.cu -arch=sm_60

# проверка успешности компиляции
if [ $? -eq 0 ]; then
    echo "Компиляция успешна. Запуск программы..."
    ./strassen2
else
    echo "Ошибка компиляции!"
fi
