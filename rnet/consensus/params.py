"""The networks this build knows, and every value they are defined by.

A network is exactly two hashed artifacts: a RoundDescriptor saying what is
being trained, and a PolicyDescriptor saying how the training is operated. This
module is where those come from — the ONE place a consensus value is written
down. Anything a node decides that is not derived from here is a value it made
up, and a node that makes up consensus values is a node running a different
network while believing otherwise.

WHY MAIN IS NOT THE MIXTURE. The 29.4B mixture is real, described, and its
shape is pinned by tests. What is not settled is how workers divide the labour
of training a sharded mixture — whether a worker holds the whole model and
updates a subset, or holds its shard and routes locally. Pinning `main` to a
model nobody can train yet would launch a network dead. So `main` stays the
dense 400M that demonstrably trains, and the mixture gets its own network,
ready for the first participant who can hold a shard.

CHANGING ANY VALUE HERE FORKS THE NETWORK IT BELONGS TO. The artifacts hash to
the anchors in genesis.py; a node with different values computes a different
anchor and refuses to speak to anyone. That is the intended behaviour and the
reason these are values in a table rather than defaults spread through the code.
"""

from __future__ import annotations

from .model_spec import ModelSpec
from .numerics import (ROUND0, AttentionKernel, ContributionFormat, DType,
                       Numerics)
from .objects import OPT_ADAFACTOR, PolicyDescriptor, RoundDescriptor

PROTOCOL_VERSION = 0x0002_0000

# The corpus and tokenizer are carried from the implementation this replaces.
# Both are derived from bytes rather than from code — the root from a scan of
# 7,359,506,899,436 bytes of FineWeb-Edu, the tokenizer hash from the artifact
# itself — so they survive a rewrite unchanged and need no reindexing.
FINEWEB_EDU_ROOT = bytes.fromhex(
    "7195da139188f4a1b779bd380562718e080959531aee0cabbf777cd13501a3b8")
FINEWEB_EDU_CHUNKS = 6_956_933
TOKENIZER_32K = bytes.fromhex(
    "1f8d0c4bc23d000c4f602654e615a53fc61f0fb3f56afdce492b640ad54b9d93")

NO_CORPUS = bytes(32)


# ---------------------------------------------------------------------------
# Model shapes
# ---------------------------------------------------------------------------

# What round 0 trained: 397,728,768 parameters at a 16384-token context, which
# is what fits a 24 GB card in bf16 with activation checkpointing. Measured, and
# the loss fell from ln(32000) = 10.37 to 7.41 over 200 inner steps.
DENSE_400M = ModelSpec(
    d_model=1024, n_layers=24, n_heads=8, n_kv_heads=2, d_ff=4096,
    vocab_size=32000, seq_len=16384, rope_theta=1_000_000,
    tie_embeddings=True, qk_norm=True,
)

# 29,408,635,904 parameters held by the network, 3,078,892,544 multiplied per
# token, 8,344,841,216 on any one holder — 15.54 GiB in bf16 against a budget of
# roughly 19.9 GiB once activations are taken out. This is the shape that puts a
# 30B model on a consumer card without offloading anything.
MOE_29B = ModelSpec(
    d_model=3072, n_layers=32, n_heads=24, n_kv_heads=4, d_ff=8192,
    vocab_size=32000, seq_len=8192, rope_theta=1_000_000,
    tie_embeddings=True, qk_norm=True,
    n_experts=64, n_experts_active=4, n_shared_experts=1, d_ff_expert=1536,
    moe_first_layer=1, moe_layer_stride=1, expert_shard_count=4,
)

# Small enough that a full round takes seconds on any CUDA card, so the protocol
# can be exercised end to end without a datacentre or a corpus.
TINY_3M = ModelSpec(
    d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=1024,
    vocab_size=256, seq_len=256, rope_theta=10_000,
    tie_embeddings=True, qk_norm=True,
)

# Regtest runs on whatever GPU is present, so it takes the portable arithmetic:
# the math attention path is reproducible anywhere the dtype exists at all.
REGTEST_NUMERICS = Numerics(
    param_dtype=DType.BF16, compute_dtype=DType.BF16, accum_dtype=DType.FP32,
    attention_kernel=AttentionKernel.MATH,
    contribution_format=ContributionFormat.INT8_POW2,
)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def _round(magic: int, model: ModelSpec, numerics: Numerics, *,
           corpus: bool) -> RoundDescriptor:
    return RoundDescriptor(
        protocol_version=PROTOCOL_VERSION,
        network_magic=magic,
        round_id=0,
        model=model,
        numerics=numerics,
        optimizer_id=OPT_ADAFACTOR,
        tokenizer_hash=TOKENIZER_32K if corpus else NO_CORPUS,
        dataset_root=FINEWEB_EDU_ROOT if corpus else NO_CORPUS,
        dataset_chunks=FINEWEB_EDU_CHUNKS if corpus else 0,
    )


def _policy(magic: int, **kw) -> PolicyDescriptor:
    # Defaults shared by every network. The two Q16 constants are 0.9 and 0.7 —
    # outer momentum and outer learning rate — held as integers because the
    # outer step must be the same arithmetic on every node, and a float constant
    # parsed from a decimal literal is not.
    base = dict(
        protocol_version=PROTOCOL_VERSION, network_magic=magic, policy_version=1,
        micro_batch=1, checkpoint_interval=1,
        outer_momentum_q16=58_982,      # 0.9
        outer_lr_q16=45_875,            # 0.7
        nesterov=True,
        challenge_percent=25, challenge_deadline_steps=4, retained_checkpoints=16,
        slash_quorum=3,
        # Shadow mode: verdicts are recorded and nothing is punished. It stays on
        # until the verification path has run against real contributions for long
        # enough to trust it, because the first thing a new slashing rule does is
        # find a way to be wrong, and being wrong here costs an honest worker.
        shadow_mode=True,
    )
    return PolicyDescriptor(**{**base, **kw})


MAIN_MAGIC = 0x524E_4D31      # "RNM1"
MOE_MAGIC = 0x524E_4531       # "RNE1"
TEST_MAGIC = 0x524E_5431      # "RNT1"
REGTEST_MAGIC = 0x524E_5231   # "RNR1"

NETWORKS: dict[str, tuple[RoundDescriptor, PolicyDescriptor]] = {
    # The dense model, the pinned corpus, and a twenty-minute round.
    "main": (
        _round(MAIN_MAGIC, DENSE_400M, ROUND0, corpus=True),
        _policy(MAIN_MAGIC, inner_steps=200, min_contributors=2,
                round_deadline_ms=1_200_000),
    ),
    # The mixture. Same corpus, same tokenizer, a shorter round because a
    # sharded holder does less work per step than a dense one does.
    "moe": (
        _round(MOE_MAGIC, MOE_29B, ROUND0, corpus=True),
        _policy(MOE_MAGIC, inner_steps=100, min_contributors=4,
                round_deadline_ms=900_000),
    ),
    # The same model as main on a lighter schedule, for shaking out changes
    # without waiting twenty minutes to find out.
    "test": (
        _round(TEST_MAGIC, DENSE_400M, ROUND0, corpus=True),
        _policy(TEST_MAGIC, inner_steps=50, min_contributors=2,
                round_deadline_ms=300_000),
    ),
    # Seconds per round, no corpus, portable arithmetic.
    "regtest": (
        _round(REGTEST_MAGIC, TINY_3M, REGTEST_NUMERICS, corpus=False),
        _policy(REGTEST_MAGIC, inner_steps=4, min_contributors=2,
                round_deadline_ms=30_000, retained_checkpoints=8,
                challenge_deadline_steps=2, slash_quorum=2),
    ),
}


def round_descriptor(network: str) -> RoundDescriptor:
    return _lookup(network)[0]


def policy_descriptor(network: str) -> PolicyDescriptor:
    return _lookup(network)[1]


def _lookup(network: str) -> tuple[RoundDescriptor, PolicyDescriptor]:
    try:
        return NETWORKS[network]
    except KeyError:
        raise KeyError(
            f"unknown network {network!r}; this build knows "
            f"{', '.join(sorted(NETWORKS))}") from None
