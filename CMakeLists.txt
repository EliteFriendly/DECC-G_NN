cmake_minimum_required(VERSION 3.10)
project(DECC-G_NN)


set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE Debug)  # Включает отладочную информацию
#set(CMAKE_BUILD_TYPE Debug)
#add_compile_options(-O3 -g3)  # Отключаем оптимизации


# Add include directory
include_directories(include)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")



add_executable(Source
    main.cpp
    Diff_evolution/DiffEvolution.cpp
    Diff_evolution/IndividualDiffEvolution.cpp 
    Diff_evolution/DiffMutation.cpp
    DECC_G/Decc_g.cpp
    Neural_network/Neuron.cpp
    Neural_network/NeuronNetwork.cpp
    # Добавьте другие исходные файлы
)

# Копируем данные в папку сборки (опционально)
file(COPY results/ DESTINATION ${CMAKE_BINARY_DIR}/results)
file(COPY test_task/ DESTINATION ${CMAKE_BINARY_DIR}/test_task)
#file(COPY algorithm_results/ DESTINATION ${CMAKE_BINARY_DIR}/algorithm_results)