add_library(rnet_mempool STATIC
    ancestors.cpp
    entry.cpp
    fee_estimator.cpp
    orphans.cpp
    pool.cpp
    rbf.cpp
)

target_include_directories(rnet_mempool PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_mempool PUBLIC
    rnet_chain
    rnet_consensus
    rnet_primitives
    rnet_crypto
    rnet_util
)
