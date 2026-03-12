add_library(rnet_consensus STATIC
    params.cpp
    growth_policy.cpp
    block_reward.cpp
    proof_of_training.cpp
    block_verify.cpp
    tx_verify.cpp
    merkle.cpp
    validation.cpp
    sigcache.cpp
    genesis.cpp
)

target_include_directories(rnet_consensus PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rnet_consensus PUBLIC rnet_script rnet_primitives rnet_crypto rnet_util)
