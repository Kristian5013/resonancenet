add_library(rnet_inference STATIC
    engine.cpp
    state.cpp
    sampler.cpp
    quantize.cpp
    batch.cpp
    server.cpp
)

target_include_directories(rnet_inference PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_inference PUBLIC
    rnet_training
    rnet_gpu
    rnet_util
)

if(WIN32)
    target_link_libraries(rnet_inference PRIVATE ws2_32)
endif()
