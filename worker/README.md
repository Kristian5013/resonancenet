# The ResonanceNet worker

The training client. PyTorch, Python 3.12.

**It hardcodes no model parameters.** It loads a genesis artifact, verifies it
against the anchor compiled into `rn_worker/consensus/genesis.py`, verifies the
tokenizer against the hash inside that artifact, and builds the model from what
the artifact says. An artifact that does not match its anchor is refused.

## Layout

```
rn_worker/
  consensus/
    canon.py        mirror of src/canon — parse and verify consensus objects
    genesis.py      trust anchors and genesis loading
    scheduler.py    deterministic batch derivation (mirror of src/dataset)
  model_llama.py    the model: GQA, SwiGLU, RMSNorm, RoPE, optional QK-norm
  determinism.py    deterministic mode (required for verifiable training)
  optimizer_adafactor.py  layerwise Adafactor: the single-24GB recipe
  scheduler_wsd.py  warmup-stable-decay schedule
tests/
  test_cross_language.py  proves this client agrees with the C++ node
```

## The mirror must stay exact

`consensus/` is a byte-for-byte mirror of the C++ implementation. If the two ever
disagree — about a genesis hash, or about which windows a worker should train on
— workers and verifiers would disagree about what was trained. That is a network
split, so the agreement is tested end-to-end against the real `rnet-tool` binary:

```bash
python3 worker/tests/test_cross_language.py
```

Run it after any change to either side. It is part of CI.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install numpy tokenizers
```

## Deterministic mode

Verifiable training runs deterministically: fixed seeds, TF32 disabled, and the
math attention backend (flash attention has non-deterministic reductions). This
is bit-exact run-to-run on the same GPU, which is what lets a verifier reproduce
an update.

It costs roughly 3x the memory of the fast path, because the attention matrix is
materialised. That cost is why the launch model and context length are sized the
way they are — measured on a 24 GB card, not chosen for elegance.
