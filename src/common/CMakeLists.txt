add_library(rnet_common STATIC chainparams.cpp)
target_link_libraries(rnet_common PUBLIC rnet_consensus)
