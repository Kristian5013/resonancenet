# Toolchain policy for ResonanceNet.
#
# C++26 with GCC >= 14 / Clang >= 19 — the same floor as the Lattica line, so a
# developer who can build one can build the other.

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The exact flags CMake uses for every file, written to build/compile_commands.json.
# Editors and clang-tidy read it instead of guessing at include paths and the
# language standard — which is the difference between an editor that understands
# C++26 and one that underlines std::format in red.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR "GCC >= 14 required for C++26 (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
  message(FATAL_ERROR "Clang >= 19 required for C++26 (found ${CMAKE_CXX_COMPILER_VERSION})")
endif()

# Consensus code: a silent conversion or an uninitialised read is a chain split,
# so the warning set is deliberately strict.
add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wcast-qual
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wdouble-promotion
  -Wformat=2)

# Integer overflow in consensus arithmetic must trap in debug, never wrap silently.
add_compile_options($<$<CONFIG:Debug>:-fno-omit-frame-pointer>)

if(WITH_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()
