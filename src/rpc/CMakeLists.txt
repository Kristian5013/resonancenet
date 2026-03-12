add_library(rnet_rpc STATIC
    request.cpp
    util.cpp
    server.cpp
    client.cpp
    blockchain.cpp
    mining.cpp
    network.cpp
    rawtransaction.cpp
    wallet_rpc.cpp
    training_rpc.cpp
    lightning_rpc.cpp
    control.cpp
    misc.cpp
)

target_include_directories(rnet_rpc PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_rpc PUBLIC
    rnet_node
    rnet_net
    rnet_chain
    rnet_mempool
    rnet_consensus
    rnet_wallet
    rnet_miner
    rnet_training
    rnet_lightning
    rnet_inference
    rnet_primitives
    rnet_crypto
    rnet_util
)

if(WIN32)
    target_link_libraries(rnet_rpc PRIVATE ws2_32 psapi)
endif()
