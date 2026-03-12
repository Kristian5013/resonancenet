add_library(rnet_miner STATIC
    miner.cpp
    worker.cpp
    training_solver.cpp
    block_template.cpp
    block_assembler.cpp
    coinbase.cpp
    difficulty.cpp
)

target_include_directories(rnet_miner PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(rnet_miner PUBLIC
    rnet_training
    rnet_consensus
    rnet_gpu
    rnet_primitives
    rnet_crypto
    rnet_util
)
