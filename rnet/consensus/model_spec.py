"""What model a round trains — dense or mixture-of-experts, described in bytes.

This is a DESCRIPTION, not an implementation. It says how many of what, so that
every node computes the same parameter count, derives the same initial weights,
and agrees on what a checkpoint is supposed to contain. The tensors themselves
live in `rnet/model/`.

WHY MOE IS HERE AND NOT A FORK. The reason to want a mixture of experts in a
protocol like this one is not the usual reason. Centrally, MoE buys compute:
30B parameters' worth of capacity at 3B parameters' worth of FLOPs per token.
That matters here too, but the sharper point is that a decentralized network is
the natural home for a model NO SINGLE PARTICIPANT CAN HOLD. Experts partition
cleanly; attention does not.

WHAT IS DECIDED HERE AND WHAT IS NOT. This module fixes the model's shape:
how many experts, which layers have them, how many are active per token, how
they group into shards. It does NOT decide how workers divide the labour of
training them — whether a worker holds the whole model and updates a subset,
or holds only its shard and routes locally. That is a protocol question about
assignments, it is genuinely open, and encoding a guess about it in a hashed
artifact would be the expensive kind of wrong. `expert_shard_count` here means
"the experts are addressable in this many groups", which is a fact about the
weights, not a claim about who trains them.

    n_experts == 0                  a dense model, and every MoE field must be 0
    moe_layer_stride == 1           every layer from moe_first_layer is MoE
    n_shared_experts > 0            always-routed experts alongside the top-k
"""

from __future__ import annotations

from dataclasses import dataclass

from ..canon.stream import CanonError, Reader, Writer

# Wire values for the two pluggable pieces of the architecture. Fixed at one
# choice each today; the field exists so that changing it later is a value in an
# artifact rather than a format break.
NORM_RMSNORM = 1
ACT_SWIGLU = 1


@dataclass(frozen=True)
class ModelSpec:
    """The complete shape of a round's model.

    Frozen: this is hashed into the round descriptor, and a mutable shape would
    mean the hash described something that no longer exists.
    """

    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    seq_len: int
    rope_theta: int
    tie_embeddings: bool
    qk_norm: bool
    norm_id: int = NORM_RMSNORM
    activation_id: int = ACT_SWIGLU

    # -- mixture of experts. All zero means dense. -------------------------
    n_experts: int = 0
    n_experts_active: int = 0
    n_shared_experts: int = 0
    d_ff_expert: int = 0
    moe_first_layer: int = 0
    moe_layer_stride: int = 1
    expert_shard_count: int = 1

    # -- derived shape -----------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    def layer_is_moe(self, layer: int) -> bool:
        """Which layers carry experts.

        Counting from `moe_first_layer` rather than from zero because leaving
        the early layers dense is common practice and cheap to express: those
        layers route poorly before the representation has settled, and a router
        trained on noise stays bad.
        """
        if not self.is_moe or layer < self.moe_first_layer:
            return False
        return (layer - self.moe_first_layer) % self.moe_layer_stride == 0

    @property
    def n_moe_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.layer_is_moe(i))

    @property
    def experts_per_shard(self) -> int:
        return self.n_experts // self.expert_shard_count if self.is_moe else 0

    # -- parameter accounting ----------------------------------------------
    #
    # Three different numbers, and confusing them is how people end up
    # surprised by what fits. Written from the formula rather than derived from
    # the tensor layout, deliberately: the two stay independent descriptions of
    # one model, so a mistake in either can be caught by comparing them.
    #
    # Pinned against literals by ModelTests.TheDenseCountIsUnchangedFromRound0
    # and ModelTests.TheThirtyBillionShapeFitsAConsumerCard. The cross-check
    # against the actual tensors cannot exist until rnet/model/ does, and it is
    # the first test that module owes.

    def _attention_params(self) -> int:
        d, hd = self.d_model, self.head_dim
        return (d * (self.n_heads * hd)          # wq
                + d * (self.n_kv_heads * hd)     # wk
                + d * (self.n_kv_heads * hd)     # wv
                + (self.n_heads * hd) * d)       # wo

    def _layer_norm_params(self) -> int:
        return 2 * self.d_model + (2 * self.head_dim if self.qk_norm else 0)

    def _dense_ffn_params(self) -> int:
        return 3 * self.d_model * self.d_ff      # gate, up, down

    def _expert_params(self) -> int:
        """One expert's FFN."""
        return 3 * self.d_model * self.d_ff_expert

    def _moe_ffn_params(self) -> int:
        """Router plus every expert the layer holds, shared ones included."""
        router = self.d_model * self.n_experts
        return router + (self.n_experts + self.n_shared_experts) * self._expert_params()

    def _embedding_params(self) -> int:
        embed = self.vocab_size * self.d_model
        head = 0 if self.tie_embeddings else embed
        return embed + head + self.d_model      # + final norm

    def parameter_count(self) -> int:
        """Every parameter in the model. What a full checkpoint contains."""
        total = self._embedding_params()
        for i in range(self.n_layers):
            total += self._attention_params() + self._layer_norm_params()
            total += self._moe_ffn_params() if self.layer_is_moe(i) else self._dense_ffn_params()
        return total

    def active_parameter_count(self) -> int:
        """Parameters a single token actually multiplies against.

        The compute-relevant number, and the one that decides how fast a step
        is. For a dense model it equals `parameter_count`; for a mixture it is
        far smaller, which is the entire point of the architecture.
        """
        total = self._embedding_params()
        for i in range(self.n_layers):
            total += self._attention_params() + self._layer_norm_params()
            if self.layer_is_moe(i):
                active = self.n_experts_active + self.n_shared_experts
                total += self.d_model * self.n_experts + active * self._expert_params()
            else:
                total += self._dense_ffn_params()
        return total

    def shard_parameter_count(self) -> int:
        """What one shard's holder must store: everything that is not an
        expert, plus its own experts.

        The memory-relevant number. A model whose `parameter_count` is far
        beyond any consumer card can still have a `shard_parameter_count` that
        fits one, and that gap is what makes a model the network holds rather
        than any node.
        """
        if not self.is_moe:
            return self.parameter_count()
        total = self._embedding_params()
        for i in range(self.n_layers):
            total += self._attention_params() + self._layer_norm_params()
            if self.layer_is_moe(i):
                held = self.experts_per_shard + self.n_shared_experts
                total += self.d_model * self.n_experts + held * self._expert_params()
            else:
                total += self._dense_ffn_params()
        return total

    def bytes_per_shard(self, bits_per_param: int) -> int:
        return self.shard_parameter_count() * bits_per_param // 8

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Everything that must hold for the shape to describe a real model.

        Checked on parse, so a malformed descriptor is refused at the artifact
        rather than surfacing as a tensor-shape error mid-round on whichever
        node happened to get there first.
        """
        for name in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
                     "vocab_size", "seq_len"):
            if getattr(self, name) <= 0:
                raise CanonError(f"model: {name} must be positive")
        if self.d_model % self.n_heads:
            raise CanonError(
                f"model: d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        if self.n_heads % self.n_kv_heads:
            raise CanonError(
                f"model: n_heads {self.n_heads} is not divisible by n_kv_heads "
                f"{self.n_kv_heads} — grouped-query attention needs whole groups")
        if self.norm_id != NORM_RMSNORM:
            raise CanonError(f"model: unknown norm {self.norm_id}")
        if self.activation_id != ACT_SWIGLU:
            raise CanonError(f"model: unknown activation {self.activation_id}")

        if not self.is_moe:
            # A dense model must say so in every field, so that exactly one byte
            # pattern means "dense" and a half-filled mixture cannot slip past.
            for name in ("n_experts_active", "n_shared_experts", "d_ff_expert",
                         "moe_first_layer"):
                if getattr(self, name):
                    raise CanonError(
                        f"model: {name} is set but n_experts is 0 — a dense model "
                        f"must be dense in every field")
            if self.moe_layer_stride != 1 or self.expert_shard_count != 1:
                raise CanonError("model: dense models pin moe_layer_stride and "
                                 "expert_shard_count to 1")
            return

        if self.n_experts_active <= 0:
            raise CanonError("model: a mixture must route to at least one expert")
        if self.n_experts_active > self.n_experts:
            raise CanonError(
                f"model: routing to {self.n_experts_active} of {self.n_experts} experts")
        if self.d_ff_expert <= 0:
            raise CanonError("model: d_ff_expert must be positive in a mixture")
        if self.n_shared_experts < 0:
            raise CanonError("model: n_shared_experts cannot be negative")
        if self.moe_layer_stride <= 0:
            raise CanonError("model: moe_layer_stride must be positive")
        if not 0 <= self.moe_first_layer < self.n_layers:
            # This is also what guarantees the mixture is not empty: layer
            # `moe_first_layer` always satisfies the stride test against itself,
            # so a first layer inside the model means at least one mixture
            # layer, whatever the stride. Proved by
            # ModelTests.AFirstLayerInsideTheModelGuaranteesAMixtureLayer.
            raise CanonError(
                f"model: moe_first_layer {self.moe_first_layer} is outside "
                f"{self.n_layers} layers, so no layer would carry experts")
        if self.expert_shard_count <= 0:
            raise CanonError("model: expert_shard_count must be positive")
        if self.n_experts % self.expert_shard_count:
            # Uneven shards would make one holder's memory bound differ from
            # another's, and the smallest card that can join is set by the
            # largest shard. Even division keeps that answer a single number.
            raise CanonError(
                f"model: {self.n_experts} experts do not divide into "
                f"{self.expert_shard_count} shards evenly")

    # -- wire ---------------------------------------------------------------

    def serialize(self, w: Writer) -> Writer:
        return (w.u32(self.d_model)
                 .u32(self.n_layers)
                 .u32(self.n_heads)
                 .u32(self.n_kv_heads)
                 .u32(self.d_ff)
                 .u32(self.vocab_size)
                 .u32(self.seq_len)
                 .u32(self.rope_theta)
                 .boolean(self.tie_embeddings)
                 .boolean(self.qk_norm)
                 .u16(self.norm_id)
                 .u16(self.activation_id)
                 .u32(self.n_experts)
                 .u32(self.n_experts_active)
                 .u32(self.n_shared_experts)
                 .u32(self.d_ff_expert)
                 .u32(self.moe_first_layer)
                 .u32(self.moe_layer_stride)
                 .u32(self.expert_shard_count))

    @classmethod
    def parse(cls, r: Reader) -> "ModelSpec":
        spec = cls(
            d_model=r.u32(), n_layers=r.u32(), n_heads=r.u32(), n_kv_heads=r.u32(),
            d_ff=r.u32(), vocab_size=r.u32(), seq_len=r.u32(), rope_theta=r.u32(),
            tie_embeddings=r.boolean(), qk_norm=r.boolean(),
            norm_id=r.u16(), activation_id=r.u16(),
            n_experts=r.u32(), n_experts_active=r.u32(), n_shared_experts=r.u32(),
            d_ff_expert=r.u32(), moe_first_layer=r.u32(), moe_layer_stride=r.u32(),
            expert_shard_count=r.u32(),
        )
        spec.validate()
        return spec

    def describe(self) -> str:
        core = (f"d_model {self.d_model}, {self.n_layers} layers, {self.n_heads} heads "
                f"(head_dim {self.head_dim}), GQA {self.n_heads // self.n_kv_heads}:1, "
                f"seq_len {self.seq_len}, vocab {self.vocab_size}")
        if not self.is_moe:
            return f"{core} — dense, {self.parameter_count():,} parameters"
        return (f"{core} — MoE: {self.n_experts} experts "
                f"({self.n_experts_active} active"
                f"{f' + {self.n_shared_experts} shared' if self.n_shared_experts else ''}) "
                f"on {self.n_moe_layers} of {self.n_layers} layers, "
                f"{self.expert_shard_count} shard(s)\n"
                f"    total  {self.parameter_count():,}\n"
                f"    active {self.active_parameter_count():,} per token\n"
                f"    shard  {self.shard_parameter_count():,} per holder")
