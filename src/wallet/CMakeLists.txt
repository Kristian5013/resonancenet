add_library(rnet_wallet STATIC
    wallet.cpp
    walletdb.cpp
    keys.cpp
    hd.cpp
    addresses.cpp
    coins.cpp
    balance.cpp
    create_tx.cpp
    sign_tx.cpp
    spend.cpp
    fees.cpp
    encrypt.cpp
    backup.cpp
    recovery.cpp
    heartbeat.cpp
    notify.cpp
)

target_include_directories(rnet_wallet PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_wallet PUBLIC
    rnet_common
    rnet_consensus
    rnet_script
    rnet_primitives
    rnet_crypto
    rnet_util
)
