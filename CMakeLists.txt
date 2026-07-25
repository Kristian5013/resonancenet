cmake_minimum_required(VERSION 3.24)

project(ResonanceNet
  VERSION 0.1.0
  DESCRIPTION "ResonanceNet — a protocol for decentralized, verifiable model training"
  HOMEPAGE_URL "https://github.com/resonancenet"
  LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CompilerOptions)

option(BUILD_TESTS "Build the unit test suite" ON)
option(WITH_SANITIZERS "Build with address+undefined sanitizers" OFF)

# ---------------------------------------------------------------------------
# third-party: SHA3-256 (FIPS 202). One hash primitive for the whole stack,
# matching Lattica's consensus hashing so artifacts anchor natively.
# ---------------------------------------------------------------------------
add_library(rnet_sha3 STATIC third_party/sha3/sha3.c)
# Exposed as "sha3/sha3.h" so the vendored header cannot be confused with our own
# src/crypto/sha3.h.
target_include_directories(rnet_sha3 PUBLIC third_party)
set_target_properties(rnet_sha3 PROPERTIES C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# libraries, lowest layer first. Each layer may only depend on layers below it.
#   util  <- crypto <- canon <- consensus <- dataset <- diloco <- verification
# ---------------------------------------------------------------------------
add_library(rnet_util STATIC
  src/util/time.cpp
  src/util/hex.cpp
  src/util/logging.cpp
  src/util/fs.cpp
  src/util/json.cpp
  src/util/args.cpp)
target_include_directories(rnet_util PUBLIC src)

add_library(rnet_crypto STATIC
  src/crypto/random.cpp
  src/crypto/sha3.cpp
  src/crypto/merkle.cpp)
target_link_libraries(rnet_crypto PUBLIC rnet_util rnet_sha3)

add_library(rnet_canon STATIC
  src/canon/canon.cpp
  src/canon/objects.cpp)
target_link_libraries(rnet_canon PUBLIC rnet_crypto)

add_library(rnet_consensus STATIC
  src/consensus/init.cpp
  src/consensus/params.cpp
  src/consensus/genesis.cpp)
target_link_libraries(rnet_consensus PUBLIC rnet_canon)

add_library(rnet_dataset STATIC
  src/dataset/manifest.cpp
  src/dataset/index.cpp
  src/dataset/scheduler.cpp)
target_link_libraries(rnet_dataset PUBLIC rnet_consensus)

# ---------------------------------------------------------------------------
# diloco: joining independent machines into one trainer
# ---------------------------------------------------------------------------
add_library(rnet_diloco STATIC
  src/diloco/producer.cpp
  src/diloco/quantize.cpp
  src/diloco/aggregation.cpp
  src/diloco/outer_optimizer.cpp
  src/diloco/round.cpp
  src/diloco/chain.cpp)
target_link_libraries(rnet_diloco PUBLIC rnet_dataset)

# ---------------------------------------------------------------------------
# verification: recomputation, verdicts, evidence
# ---------------------------------------------------------------------------
add_library(rnet_verification STATIC
  src/verification/challenge.cpp
  src/verification/replay.cpp
  src/verification/evidence.cpp)
target_link_libraries(rnet_verification PUBLIC rnet_diloco)

# ---------------------------------------------------------------------------
# net: the peer-to-peer layer
# ---------------------------------------------------------------------------
add_library(rnet_net STATIC
  src/net/message.cpp
  src/net/protocol.cpp
  src/net/addrman.cpp
  src/net/socket.cpp
  src/net/peer.cpp
  src/net/node.cpp
  src/net/corpus.cpp
  src/net/weightsstore.cpp
  src/net/objectstore.cpp
  src/net/bootstrap.cpp)
target_link_libraries(rnet_net PUBLIC rnet_verification)

# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
# The local channel: how a worker process on this machine talks to the node.
# Above net because a worker's submission is validated on exactly the same terms
# as a peer's, and below nothing — it is the outermost surface.
add_library(rnet_ipc STATIC
  src/ipc/cbor.cpp
  src/ipc/frame.cpp
  src/ipc/server.cpp)
target_link_libraries(rnet_ipc PUBLIC rnet_util)

# The participant: what a node does with the objects that arrive on its wire.
# Above net, because it is the consensus meaning of what net merely transported.
add_library(rnet_protocol STATIC
  src/protocol/participant.cpp
  src/protocol/service.cpp
  src/protocol/workerservice.cpp)
target_link_libraries(rnet_protocol PUBLIC rnet_net rnet_ipc)

add_executable(rnet-tool src/tools/rnet_tool.cpp)
target_link_libraries(rnet-tool PRIVATE rnet_net)

# The daemon: the thing an external operator actually runs.
add_executable(rnetd src/tools/rnetd.cpp)
target_link_libraries(rnetd PRIVATE rnet_protocol)

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
  enable_testing()
  add_executable(rnet_tests
    src/test/test_main.cpp
    src/test/util_tests.cpp
    src/test/crypto_tests.cpp
    src/test/canon_tests.cpp
    src/test/consensus_tests.cpp
    src/test/dataset_tests.cpp
    src/test/diloco_tests.cpp
    src/test/verification_tests.cpp
    src/test/net_tests.cpp
    src/test/protocol_tests.cpp
    src/test/ipc_tests.cpp)
  target_link_libraries(rnet_tests PRIVATE rnet_protocol)
  add_test(NAME unit COMMAND rnet_tests)
  set_tests_properties(unit PROPERTIES TIMEOUT 300)
endif()

message(STATUS "ResonanceNet ${PROJECT_VERSION} | C++${CMAKE_CXX_STANDARD} | ${CMAKE_BUILD_TYPE}"
               " | tests=${BUILD_TESTS} sanitizers=${WITH_SANITIZERS}")
