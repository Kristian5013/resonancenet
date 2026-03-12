add_library(rnet_crypto STATIC
    keccak.cpp
    hash.cpp
    ed25519.cpp
    secp256k1.cpp
    bip32.cpp
    bip39.cpp
    aes.cpp
    argon2.cpp
    chacha20.cpp
    siphash.cpp
    merkle.cpp
)

target_link_libraries(rnet_crypto PUBLIC rnet_util)

# OpenSSL
find_package(OpenSSL REQUIRED)
target_link_libraries(rnet_crypto PRIVATE OpenSSL::Crypto)

# secp256k1
find_library(SECP256K1_LIB NAMES libsecp256k1 secp256k1)
if(SECP256K1_LIB)
    target_link_libraries(rnet_crypto PUBLIC ${SECP256K1_LIB})
else()
    message(WARNING "secp256k1 not found — rnet_crypto will have missing symbols")
endif()
# Use static linking on Windows (avoids __declspec(dllimport))
if(WIN32)
    target_compile_definitions(rnet_crypto PUBLIC SECP256K1_STATIC)
endif()
