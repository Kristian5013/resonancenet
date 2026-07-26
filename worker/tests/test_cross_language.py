#!/usr/bin/env python3
"""Cross-language consensus check: the Python worker must agree with the C++ node.

If these ever diverge, workers and verifiers would disagree about what was
trained — a network split. The check is deliberately end-to-end against the real
rnet-tool binary rather than against recorded constants, so a change on either
side is caught.

Run from the repository root:
    python worker/tests/test_cross_language.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "worker"))

from rn_worker.consensus import canon, diloco, genesis, scheduler  # noqa: E402

TOOL = os.path.join(ROOT, "build", "rnet-tool")
GENESIS_DIR = os.path.join(ROOT, "share", "genesis")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n        {detail}")
        failures.append(name)


def run_tool(*args: str) -> str:
    return subprocess.run([TOOL, *args], capture_output=True, text=True, check=True).stdout


def test_genesis_hashes_agree() -> None:
    print("[genesis]")
    for network in ("main", "test", "regtest"):
        info = json.loads(run_tool("genesis-show", "--network", network))
        cpp_hash = info["genesis_hash"]

        check(
            f"{network}: python anchor == C++ hash",
            genesis.GENESIS_HASH[network] == cpp_hash,
            f"python {genesis.GENESIS_HASH[network]} != C++ {cpp_hash}",
        )

        path = os.path.join(GENESIS_DIR, f"{network}.rnet")
        if not os.path.exists(path):
            check(f"{network}: artifact present", False, f"missing {path}")
            continue

        round_descriptor = genesis.load_network(network, GENESIS_DIR)
        model = info["model"]
        check(
            f"{network}: model fields decode identically",
            (
                round_descriptor.model.d_model == model["d_model"]
                and round_descriptor.model.n_layers == model["n_layers"]
                and round_descriptor.model.n_heads == model["n_heads"]
                and round_descriptor.model.n_kv_heads == model["n_kv_heads"]
                and round_descriptor.model.d_ff == model["d_ff"]
                and round_descriptor.model.vocab_size == model["vocab_size"]
                and round_descriptor.model.seq_len == model["seq_len"]
                and round_descriptor.model.rope_theta == model["rope_theta"]
            ),
            f"python {round_descriptor.model} vs C++ {model}",
        )
        check(
            f"{network}: tokenizer hash matches",
            round_descriptor.tokenizer_hash.hex() == info["tokenizer_hash"],
            f"{round_descriptor.tokenizer_hash.hex()} != {info['tokenizer_hash']}",
        )


def test_tampered_genesis_is_refused() -> None:
    print("[genesis-tamper]")
    path = os.path.join(GENESIS_DIR, "main.rnet")
    with open(path, "rb") as f:
        raw = bytearray(f.read())
    raw[20] ^= 0x01
    try:
        genesis.load_genesis(bytes(raw), genesis.GENESIS_HASH["main"])
        check("tampered artifact refused", False, "load succeeded on modified bytes")
    except Exception:
        check("tampered artifact refused", True)

    try:
        genesis.load_genesis(bytes(raw), genesis.GENESIS_HASH["test"])
        check("cross-network artifact refused", False, "main artifact loaded under test anchor")
    except Exception:
        check("cross-network artifact refused", True)


def test_schedule_agrees() -> None:
    """The decisive check: identical training windows on both sides."""
    print("[schedule]")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.txt")
        # Text, because that is what a corpus is: documents separated by a blank
        # line, of varying length so chunk boundaries do not land at regular
        # intervals and hide a disagreement about where they go.
        with open(corpus, "w", encoding="utf-8") as f:
            for i in range(4000):
                body = "".join(chr(ord("a") + ((i * 31 + j * 7) % 26)) for j in range(200 + i % 700))
                f.write(f"Document {i}. {body}\n\n")

        prefix = os.path.join(tmp, "corpus")
        run_tool("dataset-build", "--file", corpus, "--out", prefix,
                 "--chunk-bytes", "65536", "--network", "regtest")

        manifest_container = canon.load_container_file(prefix + ".rnds", canon.OBJ_DATASET_MANIFEST)
        manifest = canon.parse_dataset_manifest(manifest_container.content)

        with open(prefix + ".json") as f:
            manifest_json = json.load(f)
        check(
            "dataset root: container == C++ json",
            manifest.dataset_root.hex() == manifest_json["dataset_root"],
            f"{manifest.dataset_root.hex()} != {manifest_json['dataset_root']}",
        )

        info = json.loads(run_tool("genesis-show", "--network", "regtest"))
        seq_len = info["model"]["seq_len"]
        micro_batch = info["micro_batch"]
        round_id = info["round_id"]

        for worker_id in (0, 1, 7):
            for step in (0, 1, 999):
                out = run_tool("schedule-show", "--manifest", prefix + ".rnds",
                               "--network", "regtest", "--worker", str(worker_id),
                               "--step", str(step))
                cpp_chunks = [
                    int(line.split("chunk=")[1]) for line in out.splitlines() if "chunk=" in line
                ]
                py_chunks = scheduler.batch_chunks(
                    manifest.dataset_root, round_id, worker_id, step, micro_batch,
                    manifest.n_chunks,
                )
                check(
                    f"worker={worker_id} step={step}: chunks identical",
                    cpp_chunks == py_chunks,
                    f"C++ {cpp_chunks} != python {py_chunks}",
                )

        # The offset within a chunk is the other half of the derivation, and the
        # node never computes it — only the worker and the verifier do, from the
        # token count of a chunk they each tokenized. Checked here against fixed
        # counts so a divergence in the domain separation shows up.
        for tokens_in_chunk in (seq_len + 1, seq_len + 1000, 200_000):
            seed = scheduler.window_seed(manifest.dataset_root, round_id, 7, 3, 0)
            off = scheduler.offset_in_chunk(seed, tokens_in_chunk, seq_len)
            check(
                f"offset in a chunk of {tokens_in_chunk} tokens stays in bounds",
                0 <= off <= tokens_in_chunk - seq_len - 1,
                f"offset {off} for {tokens_in_chunk} tokens, seq_len {seq_len}",
            )


def test_chunking_agrees() -> None:
    """Both sides must cut the corpus in the same places.

    The two implementations do it differently on purpose. Python seeks to each
    boundary and searches from there; C++ streams the file once and finds
    boundaries in the block already in hand, because seeking seven million times
    on a network volume took eighteen hours. Different algorithms, one rule — and
    if they ever disagree, two honest workers train on different text and each
    looks like a cheat.
    """
    print("[chunking]")
    import random, tempfile
    sys.path.insert(0, os.path.join(ROOT, "worker"))
    from rn_worker.corpus import LocalCorpus

    random.seed(11)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "corpus.txt")
        # Shapes chosen to be awkward: runs of blank lines, documents far shorter
        # and far longer than the target, boundaries landing on block edges.
        with open(path, "w", encoding="utf-8") as f:
            for i in range(3000):
                n = random.choice([50, 400, 3000, 12000])
                f.write(f"Doc {i}. " + "x" * n)
                f.write("\n\n" if i % 97 else "\n\n\n")

        for target in (4096, 8192, 65536):
            py_offsets = LocalCorpus(path, target, tokenizer=None).offsets
            out = run_tool("dataset-build", "--file", path, "--out", os.path.join(tmp, "m"),
                           "--chunk-bytes", str(target), "--network", "regtest")
            cpp = json.loads(out[out.index("{"):out.rindex("}") + 1])
            check(
                f"target {target}: same number of chunks",
                len(py_offsets) - 1 == cpp["n_chunks"],
                f"python {len(py_offsets) - 1} != C++ {cpp['n_chunks']}",
            )
            check(
                f"target {target}: same corpus length",
                py_offsets[-1] == cpp["n_bytes"],
                f"python {py_offsets[-1]} != C++ {cpp['n_bytes']}",
            )


def test_policy_agrees() -> None:
    """Policy is the second consensus artifact; both sides must read it identically."""
    print("[policy]")
    for network in ("main", "test", "regtest"):
        info = json.loads(run_tool("genesis-show", "--network", network))
        check(
            f"{network}: python policy anchor == C++ hash",
            genesis.POLICY_HASH[network] == info["policy_hash"],
            f"python {genesis.POLICY_HASH[network]} != C++ {info['policy_hash']}",
        )

        policy = genesis.load_policy_for_network(network, GENESIS_DIR)
        check(
            f"{network}: policy fields decode identically",
            (
                policy.inner_steps == info["inner_steps"]
                and policy.micro_batch == info["micro_batch"]
                and policy.challenge_percent == info["challenge_percent"]
                and policy.challenge_deadline_steps == info["challenge_deadline_steps"]
                and policy.retained_checkpoints == info["retained_checkpoints"]
                and policy.slash_quorum == info["slash_quorum"]
            ),
            f"python {policy} vs C++ {info}",
        )
        # The invariant that makes verification answerable at all.
        check(
            f"{network}: challenge window fits inside retention",
            policy.challenge_deadline_steps < policy.retained_checkpoints,
        )

    # Tuning policy must not move the round's identity — the reason for the split.
    main_round = genesis.load_network("main", GENESIS_DIR)
    check(
        "round and policy are independent artifacts",
        main_round.genesis_hash != genesis.POLICY_HASH["main"],
    )


def test_diloco_math_agrees() -> None:
    """Quantisation and aggregation must match bit for bit, or an honest worker's
    payload would be unreproducible and look like fraud."""
    print("[diloco]")

    # Power-of-two scaling is exact on dyadic values in both languages.
    deltas = [0.5, -0.25, 0.125, -1.0, 0.0, 0.0625]
    exp = diloco.choose_scale_exp(deltas)
    values, clamped = diloco.quantize(deltas, exp)
    back = diloco.dequantize(values, exp)
    check("quantise round-trips exactly on dyadic values", back == deltas and clamped == 0,
          f"{back} != {deltas}")

    # Rounding rule: halves away from zero, not banker's rounding.
    check("rounding is ties-away-from-zero",
          diloco.round_ties_away_from_zero(0.5) == 1
          and diloco.round_ties_away_from_zero(-0.5) == -1
          and diloco.round_ties_away_from_zero(1.5) == 2
          and diloco.round_ties_away_from_zero(2.5) == 3)

    check("rounded_div is symmetric about zero",
          diloco.rounded_div(5, 2) == 3 and diloco.rounded_div(-5, 2) == -3
          and diloco.rounded_div(7, 3) == 2 and diloco.rounded_div(-7, 3) == -2)

    # Aggregation includes every contribution and is order-independent.
    contributions = [(1, 10, [10, 20, 30]), (2, 10, [20, 40, 60])]
    averaged, scale, count = diloco.average_contributions(contributions, 3)
    check("average includes every contribution", averaged == [15, 30, 45] and count == 2,
          f"{averaged}")

    reversed_avg, _, _ = diloco.average_contributions(list(reversed(contributions)), 3)
    check("aggregation is order-independent", averaged == reversed_avg)

    # Differing scales align exactly by shifting, never by rescaling in floats.
    summed, target = diloco.sum_contributions([(1, 8, [1]), (2, 10, [4])], 1)
    check("differing scales align exactly", summed == [8] and target == 10, f"{summed} @ {target}")

    # A worker orders of magnitude away has diverged, not merely rounded.
    try:
        diloco.sum_contributions([(1, -30, [1]), (2, 30, [1])], 1)
        check("divergent scale rejected", False, "accepted a 2^60 scale gap")
    except diloco.DilocoError:
        check("divergent scale rejected", True)

    # Non-finite deltas must never be quantised into plausible integers.
    try:
        diloco.choose_scale_exp([0.5, float("nan")])
        check("non-finite pseudo-gradient rejected", False)
    except diloco.DilocoError:
        check("non-finite pseudo-gradient rejected", True)

    # The outer optimizer is deterministic and its state hashes reproducibly.
    policy = genesis.load_policy_for_network("main", GENESIS_DIR)
    a = diloco.OuterOptimizer(3, 10, policy.outer_momentum_q16, policy.outer_lr_q16, policy.nesterov)
    b = diloco.OuterOptimizer(3, 10, policy.outer_momentum_q16, policy.outer_lr_q16, policy.nesterov)
    for _ in range(20):
        ua = a.step([100, -200, 300], 10)
        ub = b.step([100, -200, 300], 10)
        if ua != ub:
            check("outer optimizer is deterministic", False, f"{ua} != {ub}")
            break
    else:
        check("outer optimizer is deterministic", True)
    check("outer optimizer state hashes agree", a.state_hash() == b.state_hash())

    # The vectorised path is what actually runs at a billion parameters; the
    # scalar one above is the reference the C++ side is compared against. If they
    # ever diverge, a worker and a verifier on different paths would disagree
    # about identical work.
    import numpy as np

    rng = np.random.default_rng(0)
    quant_ok = agg_ok = outer_ok = True
    for _ in range(10):
        d = (rng.standard_normal(1000) * float(10.0 ** int(rng.integers(-4, 2)))).astype(np.float32)
        e_ref, e_vec = diloco.choose_scale_exp(d), diloco.choose_scale_exp_array(d)
        q_ref, c_ref = diloco.quantize(d, e_ref)
        q_vec, c_vec = diloco.quantize_array(d, e_vec)
        if e_ref != e_vec or c_ref != c_vec or not np.array_equal(
                np.asarray(q_ref, dtype=np.int8), q_vec):
            quant_ok = False
    check("vectorised quantise == scalar reference", quant_ok)

    for _ in range(10):
        n = 400
        contribs = [(i, int(rng.integers(6, 12)), rng.integers(-127, 128, n).astype(np.int8))
                    for i in range(4)]
        a_ref, _, _ = diloco.average_contributions(
            [(w, s, [int(x) for x in v]) for w, s, v in contribs], n)
        a_vec, _, _ = diloco.average_contributions_array(contribs, n)
        if not np.array_equal(np.asarray(a_ref, dtype=np.int64), a_vec):
            agg_ok = False
    check("vectorised aggregate == scalar reference", agg_ok)

    n = 200
    scalar = diloco.OuterOptimizer(n, 16, policy.outer_momentum_q16, policy.outer_lr_q16,
                                   policy.nesterov)
    momentum = np.zeros(n, dtype=np.int64)
    for _ in range(5):
        agg = rng.integers(-5000, 5000, n).astype(np.int64)
        u_ref = scalar.step([int(x) for x in agg], 16)
        u_vec = diloco.outer_step_array(momentum, agg, 0, policy.outer_momentum_q16,
                                        policy.outer_lr_q16, policy.nesterov)
        if not np.array_equal(np.asarray(u_ref, dtype=np.int64), u_vec) or not np.array_equal(
                np.asarray(scalar.momentum, dtype=np.int64), momentum):
            outer_ok = False
    check("vectorised outer step == scalar reference (momentum included)", outer_ok)


def test_initial_weights_agree() -> None:
    """The starting state must be identical in both implementations.

    This is the root of everything downstream: nodes that begin from different
    weights compute deltas that cannot be aggregated and cannot be verified, and
    the failure looks like every worker cheating at once.

    Derived here for regtest, which is small enough to run in a test. main and test
    take the same code path — only longer — and are checked against their anchors
    by `rnet-tool genesis-weights`.
    """
    import hashlib

    from rn_worker.consensus import init

    print("[initial weights]")
    network = "regtest"
    info = json.loads(run_tool("genesis-show", "--network", network))
    round_descriptor = genesis.load_network(network, GENESIS_DIR)

    layout = init.tensor_layout(round_descriptor.model)
    total = sum(t.elements for t in layout)
    check(
        f"{network}: layout and parameter count agree",
        total == round_descriptor.model.parameter_count(),
        f"layout holds {total}, the spec says {round_descriptor.model.parameter_count()}",
    )

    names = [t.name for t in layout]
    check(
        f"{network}: tensor names are distinct",
        len(set(names)) == len(names),
        "two tensors sharing a name would share a stream position and produce duplicates",
    )

    with open(os.path.join(GENESIS_DIR, f"{network}.rnet"), "rb") as handle:
        round_id_hash = hashlib.sha3_256(handle.read()).digest()
    derived = init.genesis_weights_hash(round_descriptor.model, round_id_hash).hex()
    check(
        f"{network}: python weights == C++ anchor",
        derived == info["genesis_weights_hash"],
        f"python {derived} != C++ {info['genesis_weights_hash']}",
    )


def test_bf16_rounding_agrees() -> None:
    """Round half to even, the same rule the hardware follows.

    Truncation would be the obvious implementation and would differ from every GPU
    on exactly half the boundary cases — a bias that shows up as a hash mismatch on
    some machines and not others.
    """
    import numpy as np

    from rn_worker.consensus import init

    print("[bf16]")
    cases = {
        0x3F800000: 0x3F80,   # 1.0 exactly
        0xBF800000: 0xBF80,   # -1.0
        0x00000000: 0x0000,   # zero
        0x3F808000: 0x3F80,   # tie: rounds down to the even neighbour
        0x3F818000: 0x3F82,   # tie: rounds up to the even neighbour
    }
    values = np.array(list(cases.keys()), dtype=np.uint32).view(np.float32)
    got = init.float_to_bf16(values)
    for i, (bits, expected) in enumerate(cases.items()):
        check(
            f"0x{bits:08X} -> 0x{expected:04X}",
            int(got[i]) == expected,
            f"got 0x{int(got[i]):04X}",
        )


def test_parameter_order_agrees() -> None:
    """The worker's flattening order must be the consensus layout, not alphabetical.

    This is the one disagreement in the whole system that has no symptom. Byte *i*
    of a payload would mean one parameter to the worker and a different one to
    every C++ node; aggregation would still run, hashes would still be produced,
    and the model would train into noise. Nothing on the wire can detect it.

    The decisive check is that a worker can materialise the genesis weights and
    reach the anchor the network pinned — that exercises the ordering, the
    counter-based init, the bf16 rounding and the hash in one number.
    """
    print("[parameter order]")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skip  torch not installed")
        return

    import hashlib
    from types import SimpleNamespace

    from rn_worker import consensus_order, inner_loop
    from rn_worker.consensus import init
    from rn_worker.model_llama import LlamaModel

    network = "regtest"
    info = json.loads(run_tool("genesis-show", "--network", network))
    round_descriptor = genesis.load_network(network, GENESIS_DIR)
    m = round_descriptor.model

    cfg = SimpleNamespace(
        d_model=m.d_model, n_layers=m.n_layers, n_heads=m.n_heads, n_kv_heads=m.n_kv_heads,
        d_ff=m.d_ff, vocab_size=m.vocab_size, qk_norm=m.qk_norm,
        rope_theta=float(m.rope_theta), ctx_train=m.seq_len, tie_embeddings=m.tie_embeddings,
    )
    model = LlamaModel(cfg, grad_checkpointing=False).to("cpu", dtype=torch.float32)

    ordered = [name for name, _ in inner_loop.flat_parameters(model, m)]
    layout = [t.name for t in init.tensor_layout(m)]
    check(
        f"{network}: flattening order == consensus layout",
        ordered == layout,
        f"first difference at index {next((i for i, (a, b) in enumerate(zip(ordered, layout)) if a != b), None)}",
    )

    # Alphabetical order is what a reasonable person writes and what must never be
    # used. Asserted explicitly so a regression is loud rather than subtle.
    check(
        f"{network}: alphabetical order is NOT the layout",
        sorted(layout) != layout,
        "if these ever coincide this test proves nothing and must be strengthened",
    )

    with open(os.path.join(GENESIS_DIR, f"{network}.rnet"), "rb") as handle:
        round_id_hash = hashlib.sha3_256(handle.read()).digest()
    consensus_order.load_genesis_weights(model, m, round_id_hash)
    got = consensus_order.consensus_weights_hash(model, m).hex()
    check(
        f"{network}: a loaded model hashes to the network's anchor",
        got == info["genesis_weights_hash"],
        f"worker {got} != anchor {info['genesis_weights_hash']}",
    )


def main() -> int:
    if not os.path.exists(TOOL):
        print(f"error: {TOOL} not built. Run: cmake -B build -S . && cmake --build build")
        return 2

    test_genesis_hashes_agree()
    test_policy_agrees()
    test_tampered_genesis_is_refused()
    test_schedule_agrees()
    test_chunking_agrees()
    test_diloco_math_agrees()
    test_initial_weights_agree()
    test_bf16_rounding_agrees()
    test_parameter_order_agrees()

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("all cross-language checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

