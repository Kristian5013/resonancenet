#include "primitives/transaction.h"

#include "crypto/keccak.h"

namespace rnet::primitives {

// ---------------------------------------------------------------------------
// CMutableTransaction
// ---------------------------------------------------------------------------

std::vector<uint8_t> CMutableTransaction::serialize_no_witness() const {
    core::DataStream ss;

    core::Serialize(ss, version);

    // Input count + inputs (without witness)
    core::serialize_compact_size(ss, vin.size());
    for (const auto& txin : vin) {
        txin.serialize(ss);
    }

    // Output count + outputs
    core::serialize_compact_size(ss, vout.size());
    for (const auto& txout : vout) {
        txout.serialize(ss);
    }

    core::Serialize(ss, locktime);

    return std::vector<uint8_t>(ss.data(), ss.data() + ss.size());
}

std::vector<uint8_t> CMutableTransaction::serialize_with_witness() const {
    if (!has_witness()) {
        return serialize_no_witness();
    }

    core::DataStream ss;

    core::Serialize(ss, version);

    // Witness marker and flag
    uint8_t marker = 0x00;
    uint8_t flag = SERIALIZE_TRANSACTION_WITNESS;
    ss.write(&marker, 1);
    ss.write(&flag, 1);

    // Input count + inputs (without witness in the input itself)
    core::serialize_compact_size(ss, vin.size());
    for (const auto& txin : vin) {
        txin.serialize(ss);
    }

    // Output count + outputs
    core::serialize_compact_size(ss, vout.size());
    for (const auto& txout : vout) {
        txout.serialize(ss);
    }

    // Witness data for each input
    for (const auto& txin : vin) {
        txin.witness.serialize(ss);
    }

    core::Serialize(ss, locktime);

    return std::vector<uint8_t>(ss.data(), ss.data() + ss.size());
}

rnet::uint256 CMutableTransaction::compute_txid() const {
    auto data = serialize_no_witness();
    return crypto::keccak256d(std::span<const uint8_t>(data));
}

rnet::uint256 CMutableTransaction::compute_wtxid() const {
    auto data = serialize_with_witness();
    return crypto::keccak256d(std::span<const uint8_t>(data));
}

bool CMutableTransaction::has_witness() const {
    for (const auto& txin : vin) {
        if (!txin.witness.is_null()) return true;
    }
    return false;
}

std::string CMutableTransaction::to_string() const {
    std::string result = "CMutableTransaction(";
    result += "ver=" + std::to_string(version);
    result += ", vin.size=" + std::to_string(vin.size());
    result += ", vout.size=" + std::to_string(vout.size());
    result += ", locktime=" + std::to_string(locktime);
    result += ")";
    return result;
}

// ---------------------------------------------------------------------------
// CTransaction
// ---------------------------------------------------------------------------

CTransaction::CTransaction()
    : version_(TX_VERSION_DEFAULT)
    , locktime_(0)
{
    compute_hashes();
}

CTransaction::CTransaction(const CMutableTransaction& mtx)
    : version_(mtx.version)
    , vin_(mtx.vin)
    , vout_(mtx.vout)
    , locktime_(mtx.locktime)
{
    compute_hashes();
}

CTransaction::CTransaction(CMutableTransaction&& mtx)
    : version_(mtx.version)
    , vin_(std::move(mtx.vin))
    , vout_(std::move(mtx.vout))
    , locktime_(mtx.locktime)
{
    compute_hashes();
}

void CTransaction::compute_hashes() {
    auto no_wit = serialize_no_witness();
    txid_ = crypto::keccak256d(std::span<const uint8_t>(no_wit));

    auto with_wit = serialize_with_witness();
    wtxid_ = crypto::keccak256d(std::span<const uint8_t>(with_wit));
}

std::vector<uint8_t> CTransaction::serialize_no_witness() const {
    core::DataStream ss;

    core::Serialize(ss, version_);

    core::serialize_compact_size(ss, vin_.size());
    for (const auto& txin : vin_) {
        txin.serialize(ss);
    }

    core::serialize_compact_size(ss, vout_.size());
    for (const auto& txout : vout_) {
        txout.serialize(ss);
    }

    core::Serialize(ss, locktime_);

    return std::vector<uint8_t>(ss.data(), ss.data() + ss.size());
}

std::vector<uint8_t> CTransaction::serialize_with_witness() const {
    if (!has_witness()) {
        return serialize_no_witness();
    }

    core::DataStream ss;

    core::Serialize(ss, version_);

    uint8_t marker = 0x00;
    uint8_t flag = SERIALIZE_TRANSACTION_WITNESS;
    ss.write(&marker, 1);
    ss.write(&flag, 1);

    core::serialize_compact_size(ss, vin_.size());
    for (const auto& txin : vin_) {
        txin.serialize(ss);
    }

    core::serialize_compact_size(ss, vout_.size());
    for (const auto& txout : vout_) {
        txout.serialize(ss);
    }

    for (const auto& txin : vin_) {
        txin.witness.serialize(ss);
    }

    core::Serialize(ss, locktime_);

    return std::vector<uint8_t>(ss.data(), ss.data() + ss.size());
}

bool CTransaction::has_witness() const {
    for (const auto& txin : vin_) {
        if (!txin.witness.is_null()) return true;
    }
    return false;
}

int64_t CTransaction::get_value_out() const {
    int64_t total = 0;
    for (const auto& txout : vout_) {
        if (!MoneyRange(txout.value)) return -1;
        total += txout.value;
        if (!MoneyRange(total)) return -1;
    }
    return total;
}

size_t CTransaction::get_total_size() const {
    auto data = serialize_with_witness();
    return data.size();
}

size_t CTransaction::get_base_size() const {
    auto data = serialize_no_witness();
    return data.size();
}

size_t CTransaction::get_weight() const {
    return get_base_size() * 3 + get_total_size();
}

size_t CTransaction::get_virtual_size() const {
    size_t weight = get_weight();
    return (weight + 3) / 4;
}

std::string CTransaction::to_string() const {
    std::string result = "CTransaction(";
    result += "txid=" + txid_.to_hex_rev();
    result += ", ver=" + std::to_string(version_);
    result += ", vin.size=" + std::to_string(vin_.size());
    result += ", vout.size=" + std::to_string(vout_.size());
    result += ", locktime=" + std::to_string(locktime_);
    result += ")";
    return result;
}

}  // namespace rnet::primitives
