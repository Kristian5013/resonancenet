add_library(rnet_node STATIC
    context.cpp
    init.cpp
    shutdown.cpp
)

target_include_directories(rnet_node PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_node PUBLIC
    rnet_net
    rnet_chain
    rnet_mempool
    rnet_consensus
    rnet_common
    rnet_primitives
    rnet_crypto
    rnet_util
)
