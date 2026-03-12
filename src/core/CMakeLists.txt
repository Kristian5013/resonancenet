add_library(rnet_util STATIC
    types.cpp
    serialize.cpp
    stream.cpp
    varint.cpp
    error.cpp
    logging.cpp
    config.cpp
    hex.cpp
    base58.cpp
    bech32.cpp
    fs.cpp
    time.cpp
    random.cpp
    thread.cpp
    sync.cpp
    channel.cpp
    signal.cpp
    arena.cpp
)

target_include_directories(rnet_util PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_features(rnet_util PUBLIC cxx_std_20)

# Platform-specific link libraries
if(WIN32)
    target_link_libraries(rnet_util PRIVATE bcrypt shell32)
endif()

# Optional debug lock ordering
option(RNET_DEBUG_LOCKORDER "Enable lock order debugging" OFF)
if(RNET_DEBUG_LOCKORDER)
    target_compile_definitions(rnet_util PUBLIC RNET_DEBUG_LOCKORDER)
endif()
