cmake_minimum_required(VERSION 3.15)
project(apriltag_vision VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

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

set(APRILTAG_ROOT "" CACHE PATH "Root directory of an AprilTag installation (optional)")
option(FORCE_BUNDLED_APRILTAG "Always build the bundled AprilTag dependency" OFF)

# Tuning knobs for producing a ready-to-copy portable install folder (so the
# Windows build can be copied to a Linux Pi target without extra steps).
set(_portable_hint "${CMAKE_BINARY_DIR}/portable")
if(DEFINED ENV{HOME})
    file(TO_CMAKE_PATH "$ENV{HOME}/Desktop/AprilVisionPortable" _portable_hint)
endif()
if(WIN32 AND DEFINED ENV{USERPROFILE})
    file(TO_CMAKE_PATH "$ENV{USERPROFILE}/Desktop/AprilVisionPortable" _portable_hint)
endif()
set(PORTABLE_INSTALL_PREFIX "${_portable_hint}" CACHE PATH
        "Destination directory for the ready-to-run portable bundle")
option(AUTO_PORTABLE_BUNDLE "Automatically stage the portable bundle after building" ON)

set(_apriltag_include_hints)
if(APRILTAG_ROOT)
    list(APPEND _apriltag_include_hints
        "${APRILTAG_ROOT}"
        "${APRILTAG_ROOT}/include"
        "${APRILTAG_ROOT}/include/apriltag")
endif()

set(_apriltag_library_hints)
if(APRILTAG_ROOT)
    list(APPEND _apriltag_library_hints
        "${APRILTAG_ROOT}/lib"
        "${APRILTAG_ROOT}/lib64"
        "${APRILTAG_ROOT}/build"
        "${APRILTAG_ROOT}/build/Release"
        "${APRILTAG_ROOT}/build/Debug")
endif()

set(APRILTAG_BUNDLED FALSE)
if(NOT FORCE_BUNDLED_APRILTAG)
    find_path(APRILTAG_INCLUDE_DIR
        NAMES apriltag.h
        HINTS ${_apriltag_include_hints}
        PATHS /usr/include /usr/local/include /opt/homebrew/include
        PATH_SUFFIXES apriltag
    )

    find_library(APRILTAG_LIB
        NAMES apriltag
        HINTS ${_apriltag_library_hints}
        PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu /usr/lib/arm-linux-gnueabihf /opt/homebrew/lib
    )

    if(APRILTAG_INCLUDE_DIR AND APRILTAG_LIB)
        message(STATUS "Found AprilTag (system): ${APRILTAG_LIB}")
    endif()
endif()

if(NOT APRILTAG_INCLUDE_DIR OR NOT APRILTAG_LIB)
    message(STATUS "System AprilTag not found, fetching source...")
    include(FetchContent)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared AprilTag binaries" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable AprilTag demo executables" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable AprilTag tests" FORCE)
    set(BUILD_PYTHON_WRAPPER OFF CACHE BOOL "Skip AprilTag python bindings" FORCE)
    # Some toolchains cache AprilTag-specific options instead of the generic
    # BUILD_* knobs.  Force those off as well so dependency builds remain
    # headless across regenerations.
    set(APRILTAG_BUILD_EXAMPLES OFF CACHE BOOL "Disable AprilTag demo executables" FORCE)
    set(APRILTAG_BUILD_TESTS OFF CACHE BOOL "Disable AprilTag tests" FORCE)
    set(APRILTAG_BUILD_PYTHON_WRAPPER OFF CACHE BOOL "Skip AprilTag python bindings" FORCE)
    FetchContent_Declare(
            apriltag
            GIT_REPOSITORY https://github.com/AprilRobotics/apriltag.git
            GIT_TAG 3.2.0
    )
    FetchContent_MakeAvailable(apriltag)
    set(APRILTAG_BUNDLED TRUE)
    message(STATUS "Using bundled AprilTag from ${apriltag_SOURCE_DIR}")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

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

# Visual Studio/CLion sometimes default to the first executable target.  Make
# sure our camera app is always the startup target and provide an "apriltag_demo"
# alias so existing IDE configurations continue to resolve to the correct
# binary instead of the dependency's sample executable.
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
if(NOT TARGET apriltag_demo)
    add_executable(apriltag_demo ALIAS ${PROJECT_NAME})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

if(NOT APRILTAG_BUNDLED)
    target_include_directories(${PROJECT_NAME} PRIVATE ${APRILTAG_INCLUDE_DIR})
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)

# Link libraries
set(_apriltag_link_target ${APRILTAG_LIB})
if(APRILTAG_BUNDLED)
    set(_apriltag_link_target apriltag)
endif()

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${_apriltag_link_target}
)

option(ENABLE_NETWORKTABLES "Enable NetworkTables publishing via ntcore" ON)
if(ENABLE_NETWORKTABLES)
    find_package(ntcore QUIET)
    if(ntcore_FOUND)
        message(STATUS "ntcore found: enabling NetworkTables output")
        target_link_libraries(${PROJECT_NAME} ntcore)
        target_compile_definitions(${PROJECT_NAME} PRIVATE APRILV_HAS_NTCORE)
    else()
        message(STATUS "ntcore not found; NetworkTables output will fall back to UDP")
    endif()
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "$ORIGIN")
endif()

if(UNIX AND NOT APPLE AND APRILTAG_BUNDLED)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:apriltag>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
            VERBATIM)
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
    set(APRILTAG_RUNTIME_DIR "" CACHE PATH "Directory containing AprilTag runtime DLLs (optional)")
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

    # Gather AprilTag runtime DLLs
    set(_apriltag_candidate_dirs)
    if(APRILTAG_RUNTIME_DIR)
        list(APPEND _apriltag_candidate_dirs "${APRILTAG_RUNTIME_DIR}")
    endif()
    if(APRILTAG_LIB)
        get_filename_component(_apriltag_lib_dir "${APRILTAG_LIB}" DIRECTORY)
        list(APPEND _apriltag_candidate_dirs
                "${_apriltag_lib_dir}"
                "${_apriltag_lib_dir}/Release"
                "${_apriltag_lib_dir}/Debug")
    endif()
    list(REMOVE_DUPLICATES _apriltag_candidate_dirs)
    foreach(_dir ${_apriltag_candidate_dirs})
        if(EXISTS "${_dir}")
            file(GLOB _apriltag_runtime_dlls "${_dir}/*.dll")
            foreach(_dll ${_apriltag_runtime_dlls})
                _apriltag_stage_dll("${_dll}")
            endforeach()
        endif()
    endforeach()
endif()

# Windows specific
if(WIN32)
    target_link_libraries(${PROJECT_NAME} ws2_32)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(APRILTAG_BUNDLED)
    install(TARGETS apriltag
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

set(_portable_install_cmd ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix "${PORTABLE_INSTALL_PREFIX}")
if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND _portable_install_cmd --config $<CONFIG>)
endif()

add_custom_target(portable_bundle
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${PORTABLE_INSTALL_PREFIX}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PORTABLE_INSTALL_PREFIX}"
        COMMAND ${_portable_install_cmd}
        DEPENDS ${PROJECT_NAME}
        COMMENT "Staging portable bundle to ${PORTABLE_INSTALL_PREFIX}")

if(AUTO_PORTABLE_BUNDLE)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${PORTABLE_INSTALL_PREFIX}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${PORTABLE_INSTALL_PREFIX}"
            COMMAND ${_portable_install_cmd}
            COMMENT "Auto-staging portable bundle to ${PORTABLE_INSTALL_PREFIX}")
endif()
