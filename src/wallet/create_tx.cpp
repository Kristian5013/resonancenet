#include "wallet/create_tx.h"

#include "core/logging.h"
#include "primitives/amount.h"

namespace rnet::wallet {

Result<CreateTxResult> create_transaction(
    const std::vector<WalletCoin>& available_coins,
    const CreateTxParams& params) {

    // Validate recipients
    if (params.recipients.empty()) {
        return Result<CreateTxResult>::err("no recipients specified");
    }

    int64_t total_send = 0;
    for (const auto& r : params.recipients) {
        if (r.amount <= 0 && !r.subtract_fee) {
            return Result<CreateTxResult>::err("invalid recipient amount");
        }
        if (r.address.empty()) {
            return Result<CreateTxResult>::err("empty recipient address");
        }
        total_send += r.amount;
    }

    if (!primitives::MoneyRange(total_send)) {
        return Result<CreateTxResult>::err("total send amount out of range");
    }

    // Coin selection
    CoinSelectionParams cs_params;
    cs_params.target_value = total_send;
    cs_params.fee_rate = params.fee_rate;

    auto selection_result = select_coins(available_coins, cs_params);
    if (!selection_result) {
        return Result<CreateTxResult>::err("coin selection failed: " +
                                           selection_result.error());
    }
    auto& selection = selection_result.value();

    // Build transaction
    primitives::CMutableTransaction mtx;
    mtx.version = params.version;
    mtx.locktime = params.locktime;

    // Add inputs
    for (const auto& coin : selection.selected) {
        primitives::CTxIn txin(coin.outpoint);
        if (params.rbf) {
            txin.sequence = primitives::SEQUENCE_FINAL - 2;  // RBF signal
        }
        mtx.vin.push_back(std::move(txin));
    }

    // Add recipient outputs
    int64_t fee = selection.fee;
    for (const auto& r : params.recipients) {
        auto decoded = primitives::decode_address(r.address, params.network);
        if (!decoded) {
            return Result<CreateTxResult>::err("invalid address: " + r.address);
        }
        auto script = primitives::script_from_address(decoded.value());

        int64_t output_value = r.amount;
        if (r.subtract_fee) {
            output_value -= fee / static_cast<int64_t>(params.recipients.size());
            if (output_value <= 0) {
                return Result<CreateTxResult>::err(
                    "fee exceeds output after subtraction");
            }
        }

        mtx.vout.emplace_back(output_value, std::move(script));
    }

    // Add change output if needed
    CreateTxResult result;
    if (selection.has_change && selection.change > 0) {
        std::vector<uint8_t> change_script;
        if (!params.change_pubkey_hash.is_zero()) {
            change_script = primitives::make_p2wpkh_script(
                params.change_pubkey_hash.data());
        } else if (!params.change_address.empty()) {
            auto decoded = primitives::decode_address(params.change_address,
                                                       params.network);
            if (!decoded) {
                return Result<CreateTxResult>::err("invalid change address");
            }
            change_script = primitives::script_from_address(decoded.value());
        } else {
            return Result<CreateTxResult>::err("no change address specified");
        }

        mtx.vout.emplace_back(selection.change, std::move(change_script));
        result.change_address = params.change_address;
    }

    result.tx = std::move(mtx);
    result.coin_selection = std::move(selection);
    result.total_fee = fee;

    return Result<CreateTxResult>::ok(std::move(result));
}

}  // namespace rnet::wallet
