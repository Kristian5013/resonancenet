"""Command line entry point: `python -m rnet <command>`.

Everything here is read-only or writes artifacts. Nothing in this file makes a
consensus decision — it prints what the tables and anchors already say, which
is the point: an auditor should be able to check what a network claims without
building anything, without a GPU, and without trusting this program further
than the anchors it is holding it to.
"""

from __future__ import annotations

import argparse
import os
import sys

from .consensus import genesis


def default_genesis_dir() -> str:
    """Where the shipped artifacts are, from wherever this was invoked.

    The default used to be the literal relative path `share/genesis`, which is
    right in exactly one directory: the root of a source checkout. Anywhere
    else — a subdirectory of the checkout, or a machine that pip-installed the
    package rather than cloning it — `genesis-verify` printed eight REFUSED
    lines about a missing file and exited 1, which reads as the artifacts
    failing their anchors rather than as the wrong working directory.

    Order matters: an explicit `./share/genesis` wins, so somebody checking a
    directory they assembled themselves gets what they meant.
    """
    package = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(os.getcwd(), "share", "genesis"),
                      os.path.join(os.path.dirname(package), "share", "genesis"),
                      os.path.join(package, "share", "genesis")):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join("share", "genesis")


def cmd_genesis_show(args) -> int:
    for i, network in enumerate(args.networks or genesis.networks()):
        if i:
            print()
        print(genesis.describe(network))
    return 0


def cmd_genesis_emit(args) -> int:
    args.out = args.out or default_genesis_dir()
    for network in args.networks or genesis.networks():
        rnet_path, rnpol_path = genesis.emit(network, args.out)
        print(f"{network:8} {rnet_path}")
        print(f"{'':8} {rnpol_path}")
    return 0


def cmd_genesis_verify(args) -> int:
    """Check artifacts on disk against the compiled-in anchors."""
    args.dir = args.dir or default_genesis_dir()
    if not os.path.isdir(args.dir):
        print(f"rnet: no artifacts at {args.dir}.\n"
              f"      They are derivable rather than shipped — write them with\n"
              f"      `rnet genesis-emit --out {args.dir}` and verify those,\n"
              f"      or point --dir at a checkout's share/genesis.",
              file=sys.stderr)
        return 1
    failed = False
    for network in args.networks or genesis.networks():
        for suffix, loader in ((".rnet", genesis.load_round),
                               (".rnpol", genesis.load_policy)):
            path = f"{args.dir}/{network}{suffix}"
            try:
                loader(path, network)
                print(f"ok      {path}")
            except (genesis.GenesisError, OSError) as exc:
                print(f"REFUSED {path}: {exc}")
                failed = True
    return 1 if failed else 0


def cmd_daemon(args) -> int:
    """Run a node until it is told to stop."""
    import asyncio

    from .node.daemon import Daemon, DaemonConfig
    from .node.workerservice import WorkerServiceError

    config = DaemonConfig(
        network=args.network, datadir=args.datadir or "", port=args.port,
        connect=tuple(args.connect or ()),
        listen_v4=not args.no_v4, listen_v6=not args.no_v6,
        max_outbound=args.max_outbound, max_inbound=args.max_inbound,
        corpus=args.corpus or "", status_interval_s=args.status_interval)
    try:
        return asyncio.run(Daemon(config=config).run())
    except WorkerServiceError as exc:
        print(f"rnet: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # asyncio installs handlers for SIGINT where it can; this is the path
        # on platforms where it cannot, and exiting quietly beats a traceback.
        print("\nrnet: interrupted")
        return 0


def cmd_train(args) -> int:
    """Attach a worker to the local daemon and train for it."""
    try:
        from .worker.trainer import TrainerConfig, run
    except ImportError as exc:
        # The consensus half installs with numpy alone, on purpose, so the
        # training half is the one place where a missing dependency is normal
        # rather than a broken install. It arrived as a bare ModuleNotFoundError
        # traceback pointing at a line number inside diloco/inner.py, which
        # tells a new contributor nothing they can act on.
        print(f"rnet: training needs PyTorch and it is not installed "
              f"({exc.name}).\n"
              f"      pip install -e '.[train]'\n"
              f"      or, for a specific CUDA build:\n"
              f"      pip install torch --index-url "
              f"https://download.pytorch.org/whl/cu128", file=sys.stderr)
        return 1
    from .worker.client import WorkerError
    from .diloco.inner import InnerError
    try:
        return run(TrainerConfig(
            network=args.network, datadir=args.datadir or "", device=args.device,
            lr=args.lr, rounds=args.rounds, corpus=args.corpus or ""))
    except (WorkerError, InnerError) as exc:
        # InnerError is a precondition failure like any other — an unset
        # CUBLAS_WORKSPACE_CONFIG is the first thing every contributor hits, and
        # it arrived as a traceback, which reads as a broken program rather than
        # a setup step.
        print(f"rnet: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nrnet: stopped")
        return 0


def cmd_corpus_build(args) -> int:
    """Download a Hugging Face dataset into the one file a corpus is addressed
    over. Resumable — interrupt it and run it again."""
    from .dataset.build import BuildError, build, free_space

    free = free_space(os.path.dirname(os.path.abspath(args.out)) or ".")
    print(f"free space: {free / 2**40:.2f} TiB")
    try:
        state = build(args.repo, args.out, column=args.column,
                      cache_dir=args.cache, parallel=args.parallel,
                      token=args.token or os.environ.get("HF_TOKEN"),
                      limit_files=args.limit, revision=args.revision,
                      include=tuple(args.include))
    except BuildError as exc:
        print(f"rnet: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nrnet: stopped — run again to resume")
        return 0
    print(f"{args.out}: {state.bytes_written:,} bytes")
    return 0


def cmd_corpus_index(args) -> int:
    """Index a corpus file and print the root it produces.

    There was no way to learn a corpus's root short of starting a daemon
    against it and reading the exception, which is a poor way to find out that
    a two-day build produced something the network does not pin. It is also
    the command that DEFINES a pin: the root printed here is what goes into
    params.py when a network is being anchored to a new corpus.
    """
    import time

    from .dataset.corpus import CorpusError
    from .dataset.index import CorpusIndex, IndexError_, build_index

    target = args.target
    index_path = args.corpus + ".rnidx"
    started = time.monotonic()

    index = None if args.rebuild else CorpusIndex.load(index_path, args.corpus, target)
    if index is not None:
        print(f"index    {index_path} (cached)")
    else:
        size = os.path.getsize(args.corpus) if os.path.exists(args.corpus) else 0
        print(f"indexing {args.corpus} — {size / 2**40:.3f} TiB, one pass, and "
              f"it is the disk rather than the arithmetic that takes the time")

        def show(done, total, _t=started):
            print(f"\r  {done:,}/{total:,} chunks  "
                  f"{time.monotonic() - _t:5.1f}s".ljust(70), end="", flush=True)

        try:
            index = build_index(args.corpus, target, progress=show)
        except (CorpusError, IndexError_, OSError) as exc:
            print()
            print(f"rnet: {exc}", file=sys.stderr)
            return 1
        print()
        index.save(index_path)

    print(f"corpus   {args.corpus}")
    print(f"bytes    {index.n_bytes:,}")
    print(f"chunks   {index.n_chunks:,} at {target:,} bytes")
    print(f"root     {index.root.hex()}")
    print(f"took     {time.monotonic() - started:.1f}s")
    # The verdict below goes to stderr, which is not the same stream and does
    # not queue behind this one — unflushed, it printed before the figures it
    # is a verdict about.
    sys.stdout.flush()

    if not args.network:
        return 0

    # Comparing is the whole point when a network is named: "does this file
    # hold the text that network agreed on" has exactly one right answer.
    pinned = genesis.round_descriptor(args.network).dataset_root
    if pinned == bytes(32):
        print(f"\n{args.network} pins no corpus — nothing to compare against.")
        return 0
    if pinned == index.root:
        print(f"\nMATCHES {args.network}. This file is the corpus that network "
              f"pins, byte for byte.")
        return 0
    print(f"\nDIFFERENT from {args.network}, which pins {pinned.hex()}.\n"
          f"That is a different corpus, not a different copy of the same one. "
          f"Training on it\nwould produce updates no verifier holding the "
          f"pinned corpus could reproduce.", file=sys.stderr)
    return 1


def cmd_genesis_weights(args) -> int:
    """Derive the initial weights and check them against the anchor.

    The expensive one — about a minute and a half for the 29.4 billion-parameter
    mixture — and the only command that needs numpy. It is what turns "the
    network says it started here" into something a stranger can check.
    """
    import time
    failed = False
    for network in args.networks or genesis.networks():
        spec = genesis.round_descriptor(network).model
        started = time.monotonic()

        # Overwritten in place, so the finished line is the only one that stays.
        # Padded to clear whatever the previous, longer line left behind.
        def show(done, total, _n=network, _t=started):
            if done % 256 and done != total:
                return
            print(f"\r{_n:8} {done:>6}/{total} tensors  "
                  f"{time.monotonic() - _t:5.1f}s".ljust(78), end="", flush=True)

        try:
            digest = genesis.verify_weights(network, progress=show)
            line = (f"{network:8} {digest.hex()}  {spec.parameter_count():,} params  "
                    f"{time.monotonic() - started:.1f}s")
        except genesis.GenesisError as exc:
            line = f"{network:8} REFUSED: {exc}"
            failed = True
        print("\r" + line.ljust(78))
    return 1 if failed else 0


def cmd_genesis_anchors(args) -> int:
    """Print the anchors the tables currently produce.

    For regenerating genesis.py after a deliberate consensus change. It does
    NOT verify, because the whole point of running it is that the tables and
    the anchors have stopped agreeing.
    """
    from .consensus.params import NETWORKS
    print("GENESIS_HASH: dict[str, str] = {")
    for name, (r, _) in sorted(NETWORKS.items()):
        print(f'    "{name}":{" " * (10 - len(name))}"{r.id.hex()}",')
    print("}\n\nPOLICY_HASH: dict[str, str] = {")
    for name, (_, p) in sorted(NETWORKS.items()):
        print(f'    "{name}":{" " * (10 - len(name))}"{p.id.hex()}",')
    print("}")
    print("\n# Regenerate WEIGHTS_HASH with `rnet genesis-weights` — the anchors "
          "there\n# depend on these, so a change here invalidates them too.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rnet", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    daemon = sub.add_parser("daemon", help="run a node")
    daemon.add_argument("--network", default="regtest")
    daemon.add_argument("--datadir", default=None,
                        help="default: ~/.rnet/<network>")
    daemon.add_argument("--port", type=int, default=9444)
    daemon.add_argument("--connect", action="append", metavar="HOST:PORT",
                        help="dial this peer; repeatable. [::1]:9444 for IPv6")
    daemon.add_argument("--no-v4", action="store_true", help="do not listen on IPv4")
    daemon.add_argument("--no-v6", action="store_true", help="do not listen on IPv6")
    daemon.add_argument("--max-outbound", type=int, default=8)
    daemon.add_argument("--max-inbound", type=int, default=64)
    daemon.add_argument("--corpus", default=None, metavar="PATH",
                        help="serve this corpus file to peers and workers")
    daemon.add_argument("--status-interval", type=float, default=60.0,
                        metavar="SECONDS")
    daemon.set_defaults(fn=cmd_daemon)

    train = sub.add_parser("train", help="attach a worker to the local daemon")
    train.add_argument("--network", default="regtest")
    train.add_argument("--datadir", default=None)
    train.add_argument("--device", default="cuda")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--rounds", type=int, default=0,
                       help="0 means until stopped")
    train.add_argument("--corpus", default=None, metavar="PATH",
                       help="a corpus file on this machine; without it, chunks "
                            "are fetched through the daemon")
    train.set_defaults(fn=cmd_train)

    corpus = sub.add_parser("corpus-build",
                            help="download a dataset into a corpus file (resumable)")
    corpus.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    corpus.add_argument("--out", required=True, metavar="PATH")
    corpus.add_argument("--column", default="text")
    corpus.add_argument("--cache", default=None,
                        help="default: beside --out, never the home directory")
    corpus.add_argument("--parallel", type=int, default=8)
    corpus.add_argument("--token", default=None, help="or set HF_TOKEN")
    corpus.add_argument("--limit", type=int, default=0,
                        help="stop after N files; for trying it out")
    corpus.add_argument("--include", action="append", default=[], metavar="PREFIX",
                        help="keep only paths under this prefix; repeatable. "
                             "FineWeb-Edu ships sample/ copies of its own text, "
                             "so --include data/ is what avoids duplicating ~2 TB")
    corpus.add_argument("--revision", default=None, metavar="SHA",
                        help="dataset commit to build from; without it the "
                             "build tracks a moving branch and the root it "
                             "produces names no snapshot")
    corpus.set_defaults(fn=cmd_corpus_build)

    index = sub.add_parser("corpus-index",
                           help="index a corpus file and print its root")
    index.add_argument("--corpus", required=True, metavar="PATH")
    index.add_argument("--target", type=int, default=1 << 20, metavar="BYTES",
                       help="target chunk size; the default is what a node uses "
                            "and a different one is a different corpus")
    index.add_argument("--network", default=None,
                       help="compare the root against what this network pins")
    index.add_argument("--rebuild", action="store_true",
                       help="ignore any cached .rnidx and scan again")
    index.set_defaults(fn=cmd_corpus_index)

    show = sub.add_parser("genesis-show", help="what this build believes each network is")
    show.add_argument("networks", nargs="*")
    show.set_defaults(fn=cmd_genesis_show)

    emit = sub.add_parser("genesis-emit", help="write the artifacts for a network")
    emit.add_argument("networks", nargs="*")
    emit.add_argument("--out", default=None)
    emit.set_defaults(fn=cmd_genesis_emit)

    verify = sub.add_parser("genesis-verify", help="check artifacts against the anchors")
    verify.add_argument("networks", nargs="*")
    verify.add_argument("--dir", default=None)
    verify.set_defaults(fn=cmd_genesis_verify)

    weights = sub.add_parser("genesis-weights",
                             help="derive the initial weights and check the anchor (slow)")
    weights.add_argument("networks", nargs="*")
    weights.set_defaults(fn=cmd_genesis_weights)

    anchors = sub.add_parser("genesis-anchors",
                             help="print the anchors the tables produce (forks the network)")
    anchors.set_defaults(fn=cmd_genesis_anchors)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (genesis.GenesisError, KeyError) as exc:
        print(f"rnet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
