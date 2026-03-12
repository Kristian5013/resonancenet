add_library(rnet_gpu STATIC
    backend.cpp
    tensor.cpp
    context.cpp
    cuda/cuda_backend.cpp
    cuda/cuda_context.cpp
    vulkan/vulkan_backend.cpp
    vulkan/vulkan_context.cpp
)

# Metal only on Apple
if(APPLE)
    target_sources(rnet_gpu PRIVATE
        metal/metal_backend.cpp
        metal/metal_context.cpp
    )
    target_compile_definitions(rnet_gpu PRIVATE RNET_HAS_METAL)
endif()

# CPU fallback always available
target_sources(rnet_gpu PRIVATE cpu/cpu_backend.cpp)

target_include_directories(rnet_gpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(rnet_gpu PUBLIC rnet_util)

# Optional CUDA
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_compile_definitions(rnet_gpu PUBLIC RNET_HAS_CUDA)
    target_link_libraries(rnet_gpu PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Optional Vulkan
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    target_compile_definitions(rnet_gpu PUBLIC RNET_HAS_VULKAN)
    target_link_libraries(rnet_gpu PRIVATE Vulkan::Vulkan)
endif()
