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
if(WIN32)
  # BCryptGenRandom: the OS generator, so there is no configuration through which
  # a caller could weaken it.
  target_link_libraries(rnet_crypto PUBLIC bcrypt)
endif()

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
if(UNIX)
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
endif()

# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
# The local channel: how a worker process on this machine talks to the node.
# Above net because a worker's submission is validated on exactly the same terms
# as a peer's, and below nothing — it is the outermost surface.
# The canonical CBOR profile and the frame layout are pure byte manipulation and
# build anywhere. The SERVER is not: it rests on AF_UNIX with SCM_RIGHTS and
# sealed memfds, and Windows has neither. That is not an oversight to be shimmed
# — the seal is what makes "the bytes I hashed are the bytes I will serve" a fact
# rather than a hope, and a Windows port would need a different argument for the
# same property, not a different API for the same call.
add_library(rnet_ipc STATIC
  src/ipc/cbor.cpp
  src/ipc/frame.cpp)
target_link_libraries(rnet_ipc PUBLIC rnet_util)
if(UNIX)
  target_sources(rnet_ipc PRIVATE src/ipc/server.cpp)
endif()

# rnet-tool works entirely on consensus artifacts — genesis, policy, weights,
# corpus manifests — and touches no socket. It therefore builds everywhere, which
# matters: it is how someone on any platform checks that the network's anchors are
# what they claim before trusting anything else.
add_executable(rnet-tool src/tools/rnet_tool.cpp)
target_link_libraries(rnet-tool PRIVATE rnet_dataset)

# Everything below needs POSIX sockets. Guarded rather than shimmed so a Windows
# build fails at configure time with a reason, instead of at link time with forty
# undefined symbols.
if(UNIX)
  # The participant: what a node does with the objects that arrive on its wire.
  add_library(rnet_protocol STATIC
    src/protocol/participant.cpp
    src/protocol/service.cpp
    src/protocol/workerservice.cpp)
  target_link_libraries(rnet_protocol PUBLIC rnet_net rnet_ipc)

  # The daemon: the thing an external operator actually runs.
  add_executable(rnetd src/tools/rnetd.cpp)
  target_link_libraries(rnetd PRIVATE rnet_protocol)
endif()

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
  enable_testing()
  # The consensus half runs everywhere. Two implementations of the protocol on
  # two platforms must agree about hashes, so these tests are exactly the ones a
  # Windows build needs to prove it did not drift.
  add_executable(rnet_tests
    src/test/test_main.cpp
    src/test/util_tests.cpp
    src/test/crypto_tests.cpp
    src/test/canon_tests.cpp
    src/test/consensus_tests.cpp
    src/test/dataset_tests.cpp
    src/test/diloco_tests.cpp
    src/test/verification_tests.cpp)
  target_link_libraries(rnet_tests PRIVATE rnet_dataset rnet_diloco rnet_verification rnet_ipc)

  # The transport half needs sockets, Unix domain or otherwise.
  if(UNIX)
    target_sources(rnet_tests PRIVATE
      src/test/net_tests.cpp
      src/test/protocol_tests.cpp
      src/test/ipc_tests.cpp)
    target_link_libraries(rnet_tests PRIVATE rnet_protocol)
  endif()
  add_test(NAME unit COMMAND rnet_tests)
  set_tests_properties(unit PROPERTIES TIMEOUT 300)
endif()

message(STATUS "ResonanceNet ${PROJECT_VERSION} | C++${CMAKE_CXX_STANDARD} | ${CMAKE_BUILD_TYPE}"
               " | tests=${BUILD_TESTS} sanitizers=${WITH_SANITIZERS}")
