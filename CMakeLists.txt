cmake_minimum_required(VERSION 3.19)
project(PegasusTools)

set(CMAKE_CXX_STANDARD 14)

include_directories(pegasustools/include)

add_executable(TestPqubit "tests/pqubit.cpp" "pegasustools/include/pqubit.h")

#add_executable(PegasusTools
#        pegasustools/include/pqubit.h
#        pegasustools/pgq/cypqubit.cpp
#        pegasustools/pgq/util.cpp tests/pqubit.cpp)
