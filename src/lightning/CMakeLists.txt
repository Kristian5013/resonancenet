add_library(rnet_lightning STATIC
    channel.cpp
    channel_state.cpp
    htlc.cpp
    onion.cpp
    invoice.cpp
    router.cpp
    gossip.cpp
    watchtower.cpp
    channel_db.cpp
    peer.cpp
    payment.cpp
)

target_include_directories(rnet_lightning PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_lightning PUBLIC
    rnet_consensus
    rnet_script
    rnet_primitives
    rnet_crypto
    rnet_util
)
