cmake_minimum_required(VERSION 3.18)
project(ctdr)

# CUDA support
enable_language(CUDA)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda)

# CUDA kernels library
add_library(ctdr_kernels SHARED
  cuda/kernels.cu
)

# Set CUDA architecture to sm_90 (H100)
set_target_properties(ctdr_kernels PROPERTIES
  CUDA_ARCHITECTURES "90"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Python bindings
find_package(pybind11 REQUIRED)

# Pybind11 module - bindings must be CUDA file to use kernel launch syntax
# Note: pybind11_add_module doesn't support .cu files directly, need to set language
pybind11_add_module(ctdr_python
  cuda/kernels_bindings.cu
)

# Set CUDA language for the module
set_source_files_properties(cuda/kernels_bindings.cu PROPERTIES LANGUAGE CUDA)

# Link kernels library to Python module (CUDA libs added below)

# Set CUDA architecture for Python module
set_target_properties(ctdr_python PROPERTIES
  CUDA_ARCHITECTURES "90"
  POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(ctdr_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

target_include_directories(ctdr_python PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda
  ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries for Python module
target_link_libraries(ctdr_python PRIVATE
  ctdr_kernels
  ${CUDAToolkit_LIBRARIES}
)

# Install (будет дополнено позже)
# install(TARGETS ctdr_kernels ctdr_python
#   LIBRARY DESTINATION lib
# )

