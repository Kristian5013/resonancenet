cmake_minimum_required(VERSION 3.28)
project(ResonanceNet VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Core library (always built)
add_subdirectory(src/core)
add_subdirectory(src/crypto)
add_subdirectory(src/primitives)
add_subdirectory(src/script)
add_subdirectory(src/consensus)
add_subdirectory(src/common)
add_subdirectory(src/gpu)
add_subdirectory(src/training)
add_subdirectory(src/inference)
add_subdirectory(src/miner)
add_subdirectory(src/wallet)
add_subdirectory(src/chain)
add_subdirectory(src/mempool)
add_subdirectory(src/lightning)
add_subdirectory(src/net)
add_subdirectory(src/node)
add_subdirectory(src/rpc)

# ─── Entry point executables ─────────────────────────────────────────

# rnetd — main daemon
add_executable(rnetd src/rnetd.cpp)
target_include_directories(rnetd PRIVATE src)
target_link_libraries(rnetd PRIVATE
    rnet_rpc rnet_node rnet_net rnet_chain rnet_mempool
    rnet_consensus rnet_primitives rnet_script
    rnet_common rnet_gpu rnet_crypto rnet_util
    rnet_training rnet_inference rnet_miner
    rnet_wallet rnet_lightning
)
if(WIN32)
    target_link_libraries(rnetd PRIVATE ws2_32)
endif()

# rnet-cli — RPC client
add_executable(rnet-cli src/rnet-cli.cpp)
target_include_directories(rnet-cli PRIVATE src)
target_link_libraries(rnet-cli PRIVATE rnet_util)
if(WIN32)
    target_link_libraries(rnet-cli PRIVATE ws2_32)
endif()

# rnet-tx — offline transaction utility
add_executable(rnet-tx src/rnet-tx.cpp)
target_include_directories(rnet-tx PRIVATE src)
target_link_libraries(rnet-tx PRIVATE
    rnet_primitives rnet_script rnet_crypto
    rnet_common rnet_util
)

# rnet-wallet-tool — wallet management utility
add_executable(rnet-wallet-tool src/rnet-wallet-tool.cpp)
target_include_directories(rnet-wallet-tool PRIVATE src)
target_link_libraries(rnet-wallet-tool PRIVATE
    rnet_wallet rnet_primitives rnet_script rnet_crypto
    rnet_common rnet_util
)

# rnet-util — general utility
add_executable(rnet-util-bin src/rnet-util.cpp)
set_target_properties(rnet-util-bin PROPERTIES OUTPUT_NAME rnet-util)
target_include_directories(rnet-util-bin PRIVATE src)
target_link_libraries(rnet-util-bin PRIVATE
    rnet_crypto rnet_common rnet_util
)

# rnet-miner — standalone miner
add_executable(rnet-miner src/rnet-miner.cpp)
target_include_directories(rnet-miner PRIVATE src)
target_link_libraries(rnet-miner PRIVATE
    rnet_miner rnet_crypto rnet_common rnet_util
)
if(WIN32)
    target_link_libraries(rnet-miner PRIVATE ws2_32)
endif()

# ─── Test suite ──────────────────────────────────────────────────────

add_subdirectory(test)
