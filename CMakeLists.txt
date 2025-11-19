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

option(ENABLE_NETWORKTABLES "Build with WPILib ntcore support" ON)
if(ENABLE_NETWORKTABLES)
    find_package(ntcore QUIET)
    if(ntcore_FOUND)
        message(STATUS "ntcore located; NetworkTables publishing enabled")
    else()
        message(WARNING "ntcore not found - NetworkTables publishing disabled")
    endif()
endif()

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
        src/ArucoPoseHelper.cpp
        src/FieldLayout.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

if(ENABLE_NETWORKTABLES AND ntcore_FOUND)
    target_link_libraries(${PROJECT_NAME} ntcore)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NTCORE=1)
endif()

#
# Windows runtime dependency staging
# STATUS_DLL_NOT_FOUND (0xC0000135) occurs when the executable cannot locate
# dependent DLLs such as OpenCV or AprilTag at runtime.  In the developer
# environment these DLLs typically live inside the vcpkg or AprilTag build
# directories and are not copied next to the built executable by default.
# Ensure we stage any discovered DLLs into the target output directory so the
# binary starts successfully without requiring manual copying.
#
if(WIN32)
    # Helper to register a copy command only if the DLL exists at configure time.
    function(_apriltag_stage_dll dll_path)
        if(EXISTS "${dll_path}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${dll_path}"
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                    VERBATIM)
        endif()
    endfunction()

    # Try to gather OpenCV runtime DLLs.  vcpkg keeps them in "bin" alongside
    # the "lib" directory exposed through OpenCV's config.
    set(_opencv_bin_dir "")
    if(DEFINED OpenCV_BIN_DIR)
        set(_opencv_bin_dir "${OpenCV_BIN_DIR}")
    elseif(OpenCV_DIR)
        get_filename_component(_opencv_share_dir "${OpenCV_DIR}" DIRECTORY)
        get_filename_component(_opencv_root_dir "${_opencv_share_dir}" DIRECTORY)
        set(_opencv_bin_dir "${_opencv_root_dir}/bin")
    endif()

    if(_opencv_bin_dir AND EXISTS "${_opencv_bin_dir}")
        file(GLOB _opencv_runtime_dlls "${_opencv_bin_dir}/*.dll")
        list(REMOVE_DUPLICATES _opencv_runtime_dlls)
        foreach(_dll ${_opencv_runtime_dlls})
            _apriltag_stage_dll("${_dll}")
        endforeach()
    endif()

    # Gather AprilTag runtime DLLs from the configured library directory.
    set(_apriltag_candidate_dirs
            "${APRILTAG_LIB_DIR}"
            "${APRILTAG_LIB_DIR}/Release"
            "${APRILTAG_LIB_DIR}/Debug")
    foreach(_dir ${_apriltag_candidate_dirs})
        if(EXISTS "${_dir}")
            file(GLOB _apriltag_runtime_dlls "${_dir}/*.dll")
            foreach(_dll ${_apriltag_runtime_dlls})
                _apriltag_stage_dll("${_dll}")
            endforeach()
        endif()
    endforeach()
endif()

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