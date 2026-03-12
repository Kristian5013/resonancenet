add_executable(rnet_tests
    main.cpp
    test_core.cpp
    test_crypto.cpp
    test_primitives.cpp
    test_consensus.cpp
    test_growth.cpp
    test_chain.cpp
    test_mempool.cpp
    test_wallet.cpp
    test_recovery.cpp
    test_net.cpp
    test_miner.cpp
    test_lightning.cpp
    test_inference.cpp
)
target_include_directories(rnet_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(rnet_tests PRIVATE
    rnet_node rnet_net rnet_chain rnet_mempool
    rnet_wallet rnet_miner rnet_lightning rnet_inference
    rnet_training rnet_consensus rnet_primitives rnet_script
    rnet_common rnet_gpu rnet_crypto rnet_util
)
if(WIN32)
    target_link_libraries(rnet_tests PRIVATE ws2_32)
endif()
