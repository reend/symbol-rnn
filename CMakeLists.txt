cmake_minimum_required(VERSION 3.14)
project(rnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rnn src/main.cpp)

target_include_directories(rnn PRIVATE "C:/msys64/ucrt64/include/openblas")
target_link_directories(rnn PRIVATE "C:/msys64/ucrt64/lib")
target_link_libraries(rnn openblas)