find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS:" ${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_LIBRARIES:" ${CUDA_LIBRARIES})