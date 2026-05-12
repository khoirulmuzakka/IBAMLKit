cmake_minimum_required(VERSION 3.18)
project(Pileup++)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Handle CMP0148 policy
if (POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Compiler flags and definitions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED CONFIG)

# OpenMP (required)
find_package(OpenMP REQUIRED)

# + ---------------------------------------------- +
# | Compile pileupcpp library for python binding|
# + ---------------------------------------------- +
# Collect source files for the Python bindings
file(GLOB SOURCE_FILES src/*.cpp)
pybind11_add_module(pileupcpp ${SOURCE_FILES})
target_include_directories(pileupcpp PUBLIC include)
set_target_properties(pileupcpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../ibamlkit/pileup/lib/$<0:>
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../ibamlkit/pileup/lib/$<0:>
)

# Link OpenMP and Windows COM libraries
target_link_libraries(pileupcpp PRIVATE OpenMP::OpenMP_CXX ${COM_LIBS})

if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

