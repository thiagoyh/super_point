cmake_minimum_required(VERSION 3.0.0)
project(mc_vins)

# SET(CMAKE_BUILD_TYPE "Release")

ADD_COMPILE_OPTIONS(-std=c++14)
ADD_COMPILE_OPTIONS(-std=c++14)
# set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_Release "${CMAKE_C_FLAGS_Release} -O3")

find_package(PkgConfig REQUIRED)
include(cmake/eigen.cmake)
include(cmake/opencv.cmake)
include(cmake/cuda.cmake)
include(cmake/cudnn.cmake)
include(cmake/torch.cmake)
# include(cmake/nlohmann_json.cmake) # nlohmann_json::nlohmann_json
include_directories(include)

add_library(features include/features/feature_extraction.cpp include/features/feature_matching.cpp include/features/feature_tracker.cpp)
