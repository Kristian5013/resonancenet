// The node: accepting, connecting, polling, dispatching.
//
// Connection slots are the scarce resource an attacker goes after, so they are
// budgeted separately:
//
//   * INBOUND and OUTBOUND have their own limits. Sharing one pool would let an
//     attacker fill every slot with inbound connections, leaving the node unable
//     to reach out to peers of its own choosing — which is eclipse by another
//     route.
//
//   * A HANDSHAKE TIMEOUT exists because a peer that connects and then says
//     nothing costs nothing to open and holds a slot indefinitely.
//
//   * BANS EXPIRE. A permanent ban list is itself a resource an attacker can
//     grow, and addresses change hands.
//
// The loop is single-threaded and non-blocking: no peer can stall it, and there
// is no shared state to race over.
#pragma once

#include <cstdint>
#include <functional>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "consensus/params.h"
#include "net/addrman.h"
#include "net/corpus.h"
#include "net/objectstore.h"
#include "net/weightsstore.h"
#include "net/peer.h"
#include "net/socket.h"
#include "util/result.h"

namespace rnet::net {

struct NodeConfig {
    consensus::Network network{consensus::Network::Regtest};
    NetAddress bind_address{};          // where to listen; port 0 disables listening
    uint32_t max_inbound{64};
    uint32_t max_outbound{8};
    int64_t handshake_timeout_ms{20'000};
    int64_t ping_interval_ms{120'000};
    int64_t peer_timeout_ms{600'000};   // silence for this long ends the connection
    int64_t ban_duration_ms{86'400'000};
    uint64_t services{kServiceRelay};
    std::string user_agent{"rnet/0.1.0"};
    uint64_t worker_id{0};
    uint16_t determinism_class{0};

    Status Validate() const;
};

// Handler for a message from a fully-established peer. Returning an error scores
// misbehaviour against that peer, so a handler can reject nonsense without
// reaching into connection management.
//
// `now_ms` is passed rather than read from a clock, for the same reason the node
// loop takes it: a handler that consults its own clock could reach a different
// decision than the node that dispatched to it, and consensus decisions must not
// depend on which of two clocks was read.
using MessageHandler = std::function<Status(Peer&, const ReceivedMessage&, int64_t now_ms)>;

class Node {
public:
    Node(NodeConfig config, crypto::Hash256 addrman_key, uint64_t nonce);
    ~Node();

    Status Start();
    void Stop();

    // Runs one iteration: poll, accept, read, write, maintenance. `now_ms` is
    // supplied rather than read from a clock so the loop is testable and two
    // nodes cannot disagree because of clock skew.
    Status Tick(int64_t now_ms, int timeout_ms = 100);

    // Queues an outbound connection attempt.
    Status ConnectTo(const NetAddress& address, int64_t now_ms);

    void SetHandler(std::string command, MessageHandler handler);

    // Sends to every established peer. Returns how many it reached.
    size_t Broadcast(std::string_view command, std::span<const uint8_t> payload);

    size_t PeerCount() const { return peers_.size(); }
    size_t InboundCount() const;
    size_t OutboundCount() const;
    size_t ReadyCount() const;

    // The peers past the handshake, so a caller can ask one of them for something
    // that is addressed by index rather than by hash and therefore has no
    // announcement to match against.
    std::vector<uint64_t> ReadyPeerIds() const;
    AddressManager& addresses() { return addresses_; }
    ObjectStore& objects() { return objects_; }

    // Checkpoint weights, held on disk. A node without one can follow the chain
    // but cannot answer a joining worker asking for the state to start from, so a
    // network of such nodes is a network nobody can join.
    void SetWeightsStore(WeightsStore* store) { weights_ = store; }
    WeightsStore* weights() { return weights_; }

    // The corpus this node can serve. Without one a node relays and trains but
    // cannot help anyone else read the data the round pinned — which is what a
    // seed node is for.
    void SetCorpus(CorpusSource* corpus) { corpus_ = corpus; }
    CorpusSource* corpus() { return corpus_; }

    // Asks a peer for one chunk of the corpus.
    //
    // The serving half of this has always been here — a node answers `getcorpus`
    // by streaming parts — but nothing asked, so a worker could only train on a
    // corpus its own machine held. That makes a seed pointless: the whole reason
    // chunks travel with inclusion proofs is so they can come from a stranger.
    //
    // `dataset_root` is what the assembled chunk is checked against, and it comes
    // from the round descriptor rather than from the peer, so a seed that serves
    // altered text is caught by arithmetic.
    Status RequestCorpusChunk(uint64_t peer_id, uint64_t chunk_index,
                              const crypto::Hash256& dataset_root);

    // A chunk that arrived and proved itself, or nullptr. Borrowed: a chunk is a
    // megabyte and the caller copies it only if it wants to keep it.
    const std::vector<uint8_t>* TakeCorpusChunk(uint64_t chunk_index);
    bool CorpusChunkInFlight(uint64_t chunk_index) const;

    // Publishes an object: stores it locally and announces it to every peer.
    Result<crypto::Hash256> Publish(InvType type, std::vector<uint8_t> data, int64_t now_ms);

    // Starts a bulk transfer. `total_bytes` is not asked of the sender: it comes
    // from the header that announced this payload (n_params and the dtype fix it
    // exactly), so a peer cannot make the node allocate by claiming a size.
    Status RequestBulk(uint64_t peer_id, InvType type, const crypto::Hash256& hash,
                       uint64_t total_bytes, int64_t now_ms);
    bool BulkInFlight(const crypto::Hash256& hash) const;
    const NodeConfig& config() const { return config_; }

    bool IsBanned(const NetAddress& address, int64_t now_ms) const;
    void Ban(const NetAddress& address, int64_t now_ms);

    // The version this node presents. Carries both consensus anchors, so a peer
    // on a different round or policy is refused at the handshake.
    VersionMessage LocalVersion(int64_t now_ms) const;

private:
    void AcceptPending(int64_t now_ms);
    Status ServicePeer(Peer& peer, PollEntry& entry, int64_t now_ms);
    Status DispatchMessage(Peer& peer, const ReceivedMessage& message, int64_t now_ms);
    void DropDisconnected(int64_t now_ms);
    void Maintenance(int64_t now_ms);
    Status HandleInv(Peer& peer, const ReceivedMessage& message, int64_t now_ms);
    Status HandleGetData(Peer& peer, const ReceivedMessage& message);
    Status HandleObject(Peer& peer, const ReceivedMessage& message, int64_t now_ms);
    Status HandleGetChunks(Peer& peer, const ReceivedMessage& message);
    Status HandleGetCorpus(Peer& peer, const ReceivedMessage& message);
    Status HandleCorpus(Peer& peer, const ReceivedMessage& message);
    void ServeCorpusFeeds();
    Status HandleChunk(Peer& peer, const ReceivedMessage& message, int64_t now_ms);
    Status NotifyHandler(Peer& peer, const ReceivedMessage& message, int64_t now_ms);
    void RetryExpiredRequests(int64_t now_ms);

    // Feeds outstanding bulk requests as the send queues drain. Serving a whole
    // object inside the handler would push a gigabyte at an 8 MiB queue, and the
    // overflow rule would disconnect the peer being served.
    void ServeChunkFeeds();

    // How many bytes an object holds, whether it lives in memory or on disk.
    Result<uint64_t> ObjectSize(const crypto::Hash256& hash) const;
    // Reads one slice of an object from wherever it is. Weights are served from
    // disk a chunk at a time, so a two-gigabyte model is never resident just
    // because a peer asked for it.
    Result<size_t> ReadObjectRange(const crypto::Hash256& hash, uint64_t offset,
                                   std::span<uint8_t> out) const;
    void AnnounceToOthers(uint64_t origin_peer, InvType type, const crypto::Hash256& hash);

    NodeConfig config_;
    const consensus::NetworkParams& params_;
    std::array<uint8_t, 4> magic_;
    uint64_t nonce_;

    Socket listener_;
    std::map<uint64_t, std::unique_ptr<Peer>> peers_;
    std::map<uint64_t, int64_t> connected_at_;
    std::map<NetAddress, int64_t> banned_until_;
    std::map<std::string, MessageHandler> handlers_;
    AddressManager addresses_;
    ObjectStore objects_;
    WeightsStore* weights_{nullptr};

    // Bulk transfers in progress, keyed by object hash. One assembler per object,
    // not per peer: two peers feeding the same transfer would let either of them
    // corrupt it, and the whole-object hash check would blame both.
    struct BulkTransfer {
        uint64_t peer_id{0};
        InvType type{InvType::Checkpoint};
        int64_t started_at{0};
        std::unique_ptr<ChunkAssembler> assembler;
    };
    std::map<crypto::Hash256, BulkTransfer> bulk_;

    // Bulk transfers this node is SERVING, one per peer. A peer that asks again
    // replaces its own feed rather than accumulating them: two feeds for one peer
    // would let it multiply the work a single connection can demand.
    struct ChunkFeed {
        InvType type{InvType::Checkpoint};
        crypto::Hash256 hash{};
        uint64_t next_index{0};
        uint64_t last_index{0};   // exclusive
    };
    std::map<uint64_t, ChunkFeed> chunk_feeds_;

    // Corpus requests in progress, one per peer. Fed as the send queue drains, for
    // the same reason bulk objects are: a sixty-four chunk request is 256 MB, and
    // writing it into an 8 MiB queue in one tick would disconnect the peer being
    // served.
    // What a peer has asked for and not yet been sent.
    //
    // A QUEUE, not a single range. A worker's schedule scatters it across the
    // corpus — 200 unrelated chunks in a round — so it asks for them one at a
    // time, and a single feed per peer meant the second request erased the first.
    // The worker then waited a full round trip per chunk with the GPU idle, which
    // is what a training run looked like: 0% utilisation and a poll every 250 ms.
    struct CorpusFeed {
        uint64_t next_chunk{0};
        uint64_t last_chunk{0};   // exclusive
        uint32_t next_part{0};
    };
    // Bounded: a peer that asks for everything at once must not make this node
    // hold an unbounded list on its behalf.
    static constexpr size_t kMaxQueuedCorpusFeeds = 64;
    std::map<uint64_t, std::deque<CorpusFeed>> corpus_feeds_;
    CorpusSource* corpus_{nullptr};

    // Chunks being fetched from peers, and chunks that arrived and verified.
    struct CorpusFetch {
        std::unique_ptr<CorpusAssembler> assembler;
        crypto::Hash256 dataset_root{};
        int64_t started_at{0};
    };
    std::map<uint64_t, CorpusFetch> corpus_fetches_;
    std::map<uint64_t, std::vector<uint8_t>> corpus_ready_;
    uint64_t next_peer_id_{1};
    bool running_{false};
};

}  // namespace rnet::net
