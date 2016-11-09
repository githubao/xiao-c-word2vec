cmake_minimum_required(VERSION 3.3)
project(xiao_c_word2vec)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    hello.c word2vec.c compute-accuracy.c compute-accuracy.c compute-accuracy.c word2phrase.c compute-accuracy.c distance.c word-analogy.c)

add_executable(xiao_c_word2vec ${SOURCE_FILES} word2vec.c)