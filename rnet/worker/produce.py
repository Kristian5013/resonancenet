"""Applying the outer step, where the weights already are.

The daemon holds no weights and no momentum. That is not an oversight — a relay
node runs on a machine with no GPU, and making it carry 3.2 GB of optimizer
state for a 400M model, or 235 GB for the mixture, would mean only machines that
could train could also relay. So the outer step happens in the worker: the
daemon supplies what consensus decided, the worker supplies the arithmetic.

WHAT MAKES THIS CHECKABLE RATHER THAN TRUSTED. Every input is public — the
aggregate, the rates, the base weights — and every operation is integer. Any
node holding the same inputs reaches the same two hashes, which is exactly what
a verifier does when it challenges a producer.

THE MOMENTUM IS LOCAL AND MUST NOT BE. Two workers that saw the same sequence of
aggregates hold identical momentum, because the recurrence is deterministic and
integer. A worker that JOINED LATE holds none, and will produce a checkpoint
that disagrees with everyone else's until it has seen a full window. That is the
optimizer-state sync problem, it is not solved here, and a node in that state
should follow rather than produce.
"""

from __future__ import annotations

import numpy as np

from ..consensus.init import float32_to_bf16, hash_of_values
from ..consensus.model_spec import ModelSpec
from ..diloco.outer import OuterOptimizer, apply_update
from ..diloco.quantize import unpack
from ..model import weights as W
from ..model.layout import layout
from . import ipc


class ProduceError(Exception):
    pass


class Producer:
    """Holds the outer optimizer across rounds and applies what it is given."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.optimizer: OuterOptimizer | None = None
        self.applied = 0

    def apply(self, model, message: ipc.Apply, contribution_format,
              base_weights: dict) -> ipc.Applied:
        """Apply the aggregate to the weights the round STARTED from.

        Not to where training left the model. A worker's inner loop has already
        moved its own copy; the outer step is defined against the checkpoint
        everyone agreed on, and applying it to a moved copy would give a result
        no other node reaches.
        """
        expected = self.spec.parameter_count()
        if message.value_count != expected:
            raise ProduceError(
                f"produce: {message.value_count} values against {expected} parameters")

        update = unpack(message.packed, expected, contribution_format)

        # Built on first use rather than at construction, because the rates are
        # policy and arrive with the work rather than with the process.
        if self.optimizer is None:
            self.optimizer = OuterOptimizer(
                momentum_q16=message.momentum_q16, lr_q16=message.lr_q16,
                nesterov=message.nesterov)
        elif (self.optimizer.momentum_q16 != message.momentum_q16
              or self.optimizer.lr_q16 != message.lr_q16
              or self.optimizer.nesterov != message.nesterov):
            # Policy changed under a running optimizer. The momentum carried
            # forward was accumulated under the old rates, so continuing would
            # apply a recurrence nobody agreed to.
            raise ProduceError(
                "produce: the outer rates changed while momentum was carried")

        step, exp = self.optimizer.step(update, message.scale_exp)

        # A TENSOR AT A TIME. The straightforward version concatenated every
        # weight into one float64 array, subtracted the update into a second,
        # and sliced that back apart — 3.18 GB each for the dense 400M, on top
        # of the momentum and the aggregate the outer step is already holding.
        # A worker peaked at 27.8 GB here and was OOM-killed.
        #
        # `apply_update` is elementwise and the exponent is given, so the
        # arithmetic per tensor is exactly the arithmetic the whole-array
        # version did on that tensor's slice. Proved by
        # ProducerTests.ApplyingATensorAtATimeIsTheSameWeights.
        current = dict(base_weights)
        at = 0
        for tensor in layout(self.spec):
            end = at + tensor.numel
            moved = apply_update(_to_float(current[tensor.name]),
                                 step[at:end], exp)
            current[tensor.name] = float32_to_bf16(moved.astype(np.float32))
            at = end

        W.load_weights(model, current)
        self.applied += 1
        return ipc.Applied(outer_step=message.outer_step,
                           weights_hash=hash_of_values(self.spec, current),
                           optimizer_state_hash=self.optimizer.state_hash())


def _to_float(words: np.ndarray) -> np.ndarray:
    from ..consensus.init import bf16_to_float32
    return bf16_to_float32(words).astype(np.float64)
