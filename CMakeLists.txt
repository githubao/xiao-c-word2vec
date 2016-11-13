cmake_minimum_required(VERSION 3.3)
project(xiao_c_word2vec)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    compute-accuracy.c
    distance.c
    hello.c
    word-analogy.c
    word2vec.c)

add_executable(xiao_c_word2vec ${SOURCE_FILES})