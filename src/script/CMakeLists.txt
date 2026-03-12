add_library(rnet_script STATIC
    opcodes.cpp
    script.cpp
    standard.cpp
    interpreter.cpp
    sign.cpp
    verify.cpp
    recovery_script.cpp
)

target_include_directories(rnet_script PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(rnet_script PUBLIC rnet_primitives rnet_crypto rnet_util)
