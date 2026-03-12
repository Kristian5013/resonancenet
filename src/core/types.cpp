#include "core/types.h"

// Blob<N> is fully implemented in the header as a template.
// This translation unit forces instantiation of common sizes
// and ensures the header compiles cleanly.

namespace rnet::core {

// Explicit instantiations for common blob sizes
template class Blob<20>;
template class Blob<32>;
template class Blob<64>;

}  // namespace rnet::core
