cmake_minimum_required(VERSION 3.13)
project(_pywrap_oneFile)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Interpreter Development)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(INCLUDE_DIRS "../../third_party/oneStorage/include")

set(SOURCES "./io.cpp" "./io.h")

include_directories(${INCLUDE_DIRS})

#find_package(pybind11 REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/oneStorage/pybind11 ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/oneStorage/pybind11/build)
pybind11_add_module(_pywrap_oneFile ${SOURCES} "./pywrapper.cpp" ../../third_party/oneStorage/src/uring/onefile_uring.cpp ../../third_party/oneStorage/src/onefile.cpp)

target_link_libraries(_pywrap_oneFile PUBLIC uring)