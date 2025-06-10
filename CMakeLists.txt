cmake_minimum_required(VERSION 3.30)
project(onnxtt)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CUDA_LIBRARIES CUDA::cudart_static)

set(OPENCV_ROOT ${PROJECT_SOURCE_DIR}/3rdParty/opencv-4.11.0)
set(ORT_ROOT ${PROJECT_SOURCE_DIR}/3rdParty/onnxruntime-win-x64-gpu-1.21.0)
set(GSL_ROOT ${PROJECT_SOURCE_DIR}/3rdParty/gsl)
set(SPDLOG_ROOT ${PROJECT_SOURCE_DIR}/3rdParty/spdlog-1.15.0)
set(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/3rdParty/eigen-3.4.0)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

file(GLOB PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.hpp
)

# Libraries glob
file(GLOB OPENCV_LIBS ${OPENCV_ROOT}/link/*${CMAKE_STATIC_LIBRARY_SUFFIX})
file(GLOB ORT_LIBS ${ORT_ROOT}/lib/*${CMAKE_STATIC_LIBRARY_SUFFIX})

# Main executable
add_executable(onnxtt ${PROJECT_SOURCES})

# Include directories
target_include_directories(onnxtt PRIVATE
    ${GSL_ROOT}
    ${EIGEN_ROOT}
    ${OPENCV_ROOT}/include
    ${ORT_ROOT}/include
    ${SPDLOG_ROOT}/include
)

# Precompiled header
target_precompile_headers(onnxtt PRIVATE ${CMAKE_SOURCE_DIR}/src/pch.hpp)

if(MSVC)
    target_compile_options(onnxtt PRIVATE /arch:AVX2 /openmp /MT /utf-8 /W4 /fp:fast /GL /Oi /Qpar /Qpar-report:1 /MP)
    target_link_options(onnxtt PRIVATE /LTCG /INCREMENTAL:NO)
else()
    target_compile_options(onnxtt PRIVATE -march=native -fopenmp -Wall -Wpedantic) # release
endif()

target_link_libraries(onnxtt PRIVATE
    OpenMP::OpenMP_CXX
    ${OPENCV_LIBS}
	${CUDA_LIBRARIES}
    ${ORT_LIBS}
)

# Post-build: copy runtimes and config
file(GLOB RUNTIME_DLLS
    ${OPENCV_ROOT}/dylib/*${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${ORT_ROOT}/lib/*${CMAKE_SHARED_LIBRARY_SUFFIX}
)

add_custom_command(TARGET onnxtt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${RUNTIME_DLLS}
    $<TARGET_FILE_DIR:onnxtt>
)
