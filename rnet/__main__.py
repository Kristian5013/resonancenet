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


def cmd_genesis_show(args) -> int:
    for i, network in enumerate(args.networks or genesis.networks()):
        if i:
            print()
        print(genesis.describe(network))
    return 0


def cmd_genesis_emit(args) -> int:
    for network in args.networks or genesis.networks():
        rnet_path, rnpol_path = genesis.emit(network, args.out)
        print(f"{network:8} {rnet_path}")
        print(f"{'':8} {rnpol_path}")
    return 0


def cmd_genesis_verify(args) -> int:
    """Check artifacts on disk against the compiled-in anchors."""
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

    config = DaemonConfig(
        network=args.network, datadir=args.datadir or "", port=args.port,
        connect=tuple(args.connect or ()),
        listen_v4=not args.no_v4, listen_v6=not args.no_v6,
        max_outbound=args.max_outbound, max_inbound=args.max_inbound,
        status_interval_s=args.status_interval)
    try:
        return asyncio.run(Daemon(config=config).run())
    except KeyboardInterrupt:
        # asyncio installs handlers for SIGINT where it can; this is the path
        # on platforms where it cannot, and exiting quietly beats a traceback.
        print("\nrnet: interrupted")
        return 0


def cmd_train(args) -> int:
    """Attach a worker to the local daemon and train for it."""
    from .worker.trainer import TrainerConfig, run
    from .worker.client import WorkerError
    from .diloco.inner import InnerError
    try:
        return run(TrainerConfig(
            network=args.network, datadir=args.datadir or "", device=args.device,
            lr=args.lr, rounds=args.rounds))
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
                      limit_files=args.limit)
    except BuildError as exc:
        print(f"rnet: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nrnet: stopped — run again to resume")
        return 0
    print(f"{args.out}: {state.bytes_written:,} bytes")
    return 0


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
    corpus.set_defaults(fn=cmd_corpus_build)

    show = sub.add_parser("genesis-show", help="what this build believes each network is")
    show.add_argument("networks", nargs="*")
    show.set_defaults(fn=cmd_genesis_show)

    emit = sub.add_parser("genesis-emit", help="write the artifacts for a network")
    emit.add_argument("networks", nargs="*")
    emit.add_argument("--out", default="share/genesis")
    emit.set_defaults(fn=cmd_genesis_emit)

    verify = sub.add_parser("genesis-verify", help="check artifacts against the anchors")
    verify.add_argument("networks", nargs="*")
    verify.add_argument("--dir", default="share/genesis")
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
