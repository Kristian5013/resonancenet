add_library(rnet_net STATIC
    protocol.cpp
    transport.cpp
    connection.cpp
    conn_manager.cpp
    msg_handler.cpp
    sync_manager.cpp
    peer.cpp
    sync.cpp
    addr_man.cpp
    ban.cpp
    message.cpp
)

target_include_directories(rnet_net PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_net PUBLIC
    rnet_chain
    rnet_mempool
    rnet_consensus
    rnet_primitives
    rnet_crypto
    rnet_util
)

if(WIN32)
    target_link_libraries(rnet_net PRIVATE ws2_32)
endif()
