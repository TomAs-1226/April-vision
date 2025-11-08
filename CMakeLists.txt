cmake_minimum_required(VERSION 3.15)
project(apriltag_vision VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# CRITICAL: Define NOMINMAX to prevent Windows min/max macros
add_definitions(-DNOMINMAX)

# Compiler optimizations - FIXED for Visual Studio
if(MSVC)
    # Remove conflicting flags
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Od" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

    # Set proper flags for Debug and Release
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W4 /Zi /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W4 /O2 /arch:AVX2")

    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native -ffast-math)
endif()

# Find packages via vcpkg
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# AprilTag paths
set(APRILTAG_INCLUDE_DIR "C:/apriltag" CACHE PATH "AprilTag include directory")
set(APRILTAG_LIB_DIR "C:/apriltag/build" CACHE PATH "AprilTag library directory")

# Include directories
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${APRILTAG_INCLUDE_DIR}
)

# Source files
set(SOURCES
        src/main.cpp
        src/Detector.cpp
        src/PoseEstimator.cpp
        src/Tracker.cpp
        src/NetworkPublisher.cpp
        src/FrameProcessor.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

# Find AprilTag library
find_library(APRILTAG_LIB
        NAMES apriltag
        PATHS
        ${APRILTAG_LIB_DIR}/Release
        ${APRILTAG_LIB_DIR}/Debug
        ${APRILTAG_LIB_DIR}
        NO_DEFAULT_PATH
)

if(APRILTAG_LIB)
    message(STATUS "Found AprilTag: ${APRILTAG_LIB}")
    target_link_libraries(${PROJECT_NAME} ${APRILTAG_LIB})
else()
    message(WARNING "AprilTag library not found in ${APRILTAG_LIB_DIR}")
endif()

# Windows specific
if(WIN32)
    target_link_libraries(${PROJECT_NAME} ws2_32)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")