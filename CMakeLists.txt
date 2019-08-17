cmake_minimum_required(VERSION 3.12)

project(GPU-path-tracing-tutorial-4)

find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})

set(bvh_srcs
    src/BVH.h
    src/BVH.cpp
    src/BVHNode.h
    src/BVHNode.cpp
    src/SplitBVHBuilder.h
    src/SplitBVHBuilder.cpp
)

set(renderer_srcs
    src/Camera.h
    src/Camera.cpp
    src/Geometry.h
    src/Scene.h
    src/SceneLoader.h
    src/SceneLoader.cpp
    src/tonemap.cuh
    src/Util.h
    src/Util.cpp
    src/CudaBVH.h
    src/CudaBVH.cpp
    src/CudaRenderKernel.h
    src/renderkernel.cuh
    src/renderkernel.cu
)

set(utils_srcs
    src/Array.h
    src/cutil_math.h
    src/linear_math.h
    src/resource.h
    src/HDRloader.h
    src/HDRloader.cpp
    src/MouseKeyboardInput.h
    src/rgbe.h
    src/rgbe.cpp
    src/Sort.h
    src/Sort.cpp
)

set(srcs
    src/main.cpp
)

source_group("src\\bvh\\" FILES ${bvh_srcs})
source_group("src\\renderer\\" FILES ${renderer_srcs})
source_group("src\\utils\\" FILES ${utils_srcs})
source_group("src\\" FILES ${srcs})

add_compile_options("/Zi")
add_link_options("/DEBUG")

cuda_add_executable(tutorial4 ${bvh_srcs} ${renderer_srcs} ${utils_srcs} ${srcs})

target_link_libraries(tutorial4
    ${CUDA_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

target_compile_options(tutorial4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61 --expt-extended-lambda -Xcompiler>)

if (MSVC)
    set_target_properties(tutorial4 PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()