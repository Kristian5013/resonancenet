#include "node/context.h"

// Full includes so unique_ptr destructors can see the complete types.
#include "chain/chainstate.h"
#include "core/config.h"
#include "mempool/pool.h"
#include "net/addr_man.h"
#include "net/conn_manager.h"
#include "net/sync.h"

namespace rnet::node {

NodeContext::NodeContext() = default;

// Destructor must live in the .cpp where all types are complete.
NodeContext::~NodeContext() = default;

} // namespace rnet::node
