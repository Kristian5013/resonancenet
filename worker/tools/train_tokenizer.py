#!/usr/bin/env python3
"""Trains the BPE the round descriptor pins by hash.

The tokenizer is a consensus artifact. Its SHA3-256 goes into the round
descriptor, so every worker and every verifier uses the same one and a mismatch
is refused at the handshake. That makes this script's output a value the whole
network agrees on, and it makes reproducibility a property worth having: anyone
should be able to run this and see why the pinned artifact is what it is.

Byte-level BPE with no normaliser, matching the construction of the 128k
artifact this replaces:

  * BYTE-LEVEL, so every byte sequence is representable and nothing is ever an
    unknown token. A corpus that contains a byte the tokenizer cannot express
    would be a corpus with windows nobody can train on.
  * NO NORMALISER. Unicode normalisation is a transformation that two
    implementations can disagree about, and the corpus is already the thing being
    committed to — changing it on the way in would mean the tokens do not describe
    the bytes the Merkle root covers.
  * NO ADDED TOKENS. Documents are separated in the corpus by a blank line, which
    is text, so there is nothing to reserve an id for.

Usage:
    train_tokenizer.py --input sample.txt --vocab 32000 --out tokenizer.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def train(input_paths: list[Path], vocab_size: int, out: Path) -> None:
    tokenizer = Tokenizer(BPE())
    # add_prefix_space=False: a leading space is part of the token, as in the
    # artifact this replaces. Changing it changes every id.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                       trim_offsets=True,
                                                       use_regex=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True,
                                           use_regex=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        # The 256 single bytes, so nothing is ever unrepresentable.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
    )

    started = time.time()
    tokenizer.train([str(p) for p in input_paths], trainer)
    elapsed = time.time() - started

    tokenizer.save(str(out))
    raw = out.read_bytes()
    digest = hashlib.sha3_256(raw).hexdigest()
    model = json.loads(raw)["model"]

    print()
    print(f"trained in {elapsed/60:.1f} min on {sum(p.stat().st_size for p in input_paths)/1e9:.2f} GB")
    print(f"vocab   {len(model['vocab']):,}   merges {len(model['merges']):,}")
    print(f"file    {out}  ({len(raw)/1e6:.1f} MB)")
    print(f"sha3    {digest}")
    print()
    print("Pin that hash as tokenizer_hash in the round descriptor. Nothing else")
    print("about this file is trusted: a worker hashes the bytes it was handed and")
    print("refuses them unless they match.")

    # A corpus this tokenizer cannot round-trip would produce windows whose text
    # nobody can reconstruct, so the property is checked here rather than assumed.
    sample = "Hello, world! Мир. 日本語 \t\n\n emoji 🙂 and \x00 control bytes."
    ids = tokenizer.encode(sample).ids
    back = tokenizer.decode(ids)
    if back != sample:
        print(f"\nWARNING: round trip differs\n  in:  {sample!r}\n  out: {back!r}", file=sys.stderr)
    else:
        print("round trip: exact, including non-ASCII, control bytes and whitespace")

    longest = max(len(t) for t in model["vocab"])
    print(f"longest token: {longest} characters — the manifest's minimum chunk size "
          f"depends on this")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, nargs="+", required=True)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in args.input:
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1
    train(args.input, args.vocab, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
