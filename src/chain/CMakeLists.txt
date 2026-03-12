add_library(rnet_chain STATIC
    block_index.cpp
    chainstate.cpp
    coins.cpp
    reorg.cpp
    storage.cpp
    utxo_db.cpp
)

target_include_directories(rnet_chain PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(rnet_chain PUBLIC
    rnet_consensus
    rnet_primitives
    rnet_crypto
    rnet_util
)

# LevelDB
find_package(leveldb CONFIG QUIET)
if(leveldb_FOUND)
    target_link_libraries(rnet_chain PRIVATE leveldb::leveldb)
else()
    find_path(LEVELDB_INCLUDE_DIR leveldb/db.h)
    find_library(LEVELDB_LIBRARY leveldb)
    if(LEVELDB_INCLUDE_DIR AND LEVELDB_LIBRARY)
        target_include_directories(rnet_chain PRIVATE ${LEVELDB_INCLUDE_DIR})
        target_link_libraries(rnet_chain PRIVATE ${LEVELDB_LIBRARY})
    endif()
endif()
