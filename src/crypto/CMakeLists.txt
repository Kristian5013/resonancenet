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

# OpenSSL — prefer static on Windows to avoid DLL dependency
if(WIN32)
    set(OPENSSL_USE_STATIC_LIBS TRUE)
    set(OPENSSL_MSVC_STATIC_RT TRUE)
endif()
find_package(OpenSSL REQUIRED)
target_link_libraries(rnet_crypto PRIVATE OpenSSL::Crypto)
# Static OpenSSL on Windows needs crypt32 and ws2_32
if(WIN32 AND OPENSSL_USE_STATIC_LIBS)
    target_link_libraries(rnet_crypto PRIVATE crypt32 ws2_32)
endif()

# secp256k1
find_library(SECP256K1_LIB NAMES libsecp256k1 secp256k1)
if(SECP256K1_LIB)
    target_link_libraries(rnet_crypto PUBLIC ${SECP256K1_LIB})
    # vcpkg builds secp256k1 with precomputed tables in a separate library
    find_library(SECP256K1_PRECOMPUTED NAMES secp256k1_precomputed)
    if(SECP256K1_PRECOMPUTED)
        target_link_libraries(rnet_crypto PUBLIC ${SECP256K1_PRECOMPUTED})
    endif()
else()
    message(WARNING "secp256k1 not found — rnet_crypto will have missing symbols")
endif()
# Use static linking on Windows (avoids __declspec(dllimport))
if(WIN32)
    target_compile_definitions(rnet_crypto PUBLIC SECP256K1_STATIC)
endif()
