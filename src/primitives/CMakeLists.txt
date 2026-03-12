add_library(rnet_primitives STATIC
    amount.cpp
    outpoint.cpp
    txin.cpp
    txout.cpp
    witness.cpp
    transaction.cpp
    block_header.cpp
    block.cpp
    fees.cpp
    address.cpp
)

target_link_libraries(rnet_primitives PUBLIC rnet_crypto rnet_util)
