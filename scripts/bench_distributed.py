#!/usr/bin/env python3
"""
Distributed benchmark runner for hybrid-LB vLLM clusters.

Splits prompts evenly across all node endpoints, runs vllm bench serve in
parallel (one subprocess per node), then aggregates and prints combined results.

Usage:
  python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt
  python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt -c 32,64,128
  python scripts/bench_distributed.py --endpoints http://10.0.0.1:8200,http://10.0.0.2:8200
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VLLM_BIN = str(REPO_ROOT / ".venv/bin/vllm")
SSH_OPTS = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"


# ── Node resolution ─────────────────────────────────────────────────────────


def get_node_ip(hostname: str) -> str:
    """Resolve a node hostname to its IP via SSH (same pattern as launch scripts)."""
    try:
        result = subprocess.run(
            ["ssh"] + SSH_OPTS.split() + ["-n", hostname, "hostname -I"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except Exception:
        pass
    return hostname


def resolve_endpoints(nodes_file: str, port: int) -> list[str]:
    """Read nodes file, resolve IPs, return list of http://IP:port URLs."""
    nodes = []
    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                nodes.append(line.split()[0])

    endpoints = []
    for node in nodes:
        ip = get_node_ip(node)
        endpoints.append(f"http://{ip}:{port}")
        print(f"  {node} -> {ip}")
    return endpoints


# ── Per-node benchmark ───────────────────────────────────────────────────────


def run_bench_on_node(
    endpoint: str,
    model: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    result_dir: str,
    label: str,
    node_idx: int,
) -> dict | None:
    """Run vllm bench serve against a single endpoint, return parsed result."""
    filename = f"{label}_node{node_idx}_{concurrency}.json"
    cmd = [
        VLLM_BIN,
        "bench",
        "serve",
        "--base-url",
        endpoint,
        "--model",
        model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "0.0",
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "128",
        "--ignore-eos",
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        filename,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--trust-remote-code",
        "--tokenizer-mode",
        "auto",
        "--disable-tqdm",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            f"  [node {node_idx}] bench FAILED (exit {proc.returncode})",
            file=sys.stderr,
        )
        stderr_tail = proc.stderr
        if stderr_tail:
            print(f"  stderr: {stderr_tail}", file=sys.stderr)
        return None

    result_path = os.path.join(result_dir, filename)
    if not os.path.exists(result_path):
        print(
            f"  [node {node_idx}] result file not found: {result_path}", file=sys.stderr
        )
        return None

    with open(result_path) as f:
        return json.load(f)


# ── Aggregation ──────────────────────────────────────────────────────────────


def aggregate_results(node_results: list[dict]) -> dict:
    """Aggregate results from multiple nodes into a single combined result.

    Throughput (tok/s, req/s): sum across nodes
    Completed/failed: sum
    Latency percentiles: weighted mean for p50, max for p99 (conservative)
    """
    valid = [r for r in node_results if r is not None]
    if not valid:
        return {}

    agg = {}

    # Sum metrics
    for key in ("completed", "failed"):
        agg[key] = sum(r.get(key, 0) for r in valid)

    for key in (
        "output_throughput",
        "request_throughput",
        "total_input_tokens",
        "total_output_tokens",
        "input_throughput",
    ):
        agg[key] = sum(r.get(key, 0) for r in valid)

    # Latency percentiles: weighted mean for p50, max for p90/p99
    # Weight by completed count so nodes with more requests contribute more
    total_completed = max(agg["completed"], 1)
    weights = [r.get("completed", 0) / total_completed for r in valid]

    for metric in ("ttft", "tpot", "itl", "e2el"):
        # Means: weighted average
        mean_key = f"mean_{metric}_ms"
        agg[mean_key] = sum(r.get(mean_key, 0) * w for r, w in zip(valid, weights))

        # p50: weighted average (reasonable when load is balanced)
        p50_key = f"p50_{metric}_ms"
        agg[p50_key] = sum(r.get(p50_key, 0) * w for r, w in zip(valid, weights))

        # p90, p99: take max across nodes (conservative upper bound —
        # the true global p99 could be any of the per-node p99s)
        for pct in ("p90", "p99"):
            pct_key = f"{pct}_{metric}_ms"
            agg[pct_key] = max(r.get(pct_key, 0) for r in valid)

    # Duration: max (wall-clock time is bounded by the slowest node)
    agg["duration"] = max(r.get("duration", 0) for r in valid)

    return agg


# ── Output formatting ────────────────────────────────────────────────────────


def print_summary(
    results: list[tuple[int, dict]],
    input_len: int,
    output_len: int,
    num_nodes: int,
):
    print()
    print(f"{'=' * 130}")
    print(
        f"  Distributed decode benchmark   input={input_len}  output={output_len}"
        f"  nodes={num_nodes}"
    )
    print(f"{'=' * 130}")
    header = (
        f"{'Conc':>5}  "
        f"{'OK':>5} {'Fail':>4}  "
        f"{'Out tok/s':>10} {'Req/s':>7}  "
        f"{'TTFT p50':>9} {'TTFT p99':>9}  "
        f"{'TPOT p50':>9} {'TPOT p99':>9}  "
        f"{'ITL p50':>9} {'ITL p99':>9}  "
        f"{'E2E p50':>10} {'E2E p99':>10}"
    )
    print(header)
    print("-" * 130)

    for conc, r in results:
        completed = r.get("completed", 0)
        failed = r.get("failed", 0)
        out_tps = r.get("output_throughput", 0)
        req_s = r.get("request_throughput", 0)
        ttft_50 = r.get("p50_ttft_ms", 0)
        ttft_99 = r.get("p99_ttft_ms", 0)
        tpot_50 = r.get("p50_tpot_ms", 0)
        tpot_99 = r.get("p99_tpot_ms", 0)
        itl_50 = r.get("p50_itl_ms", 0)
        itl_99 = r.get("p99_itl_ms", 0)
        e2e_50 = r.get("p50_e2el_ms", 0)
        e2e_99 = r.get("p99_e2el_ms", 0)

        line = (
            f"{conc:>5}  "
            f"{completed:>5} {failed:>4}  "
            f"{out_tps:>10.1f} {req_s:>7.2f}  "
            f"{ttft_50:>8.0f}ms {ttft_99:>8.0f}ms  "
            f"{tpot_50:>8.1f}ms {tpot_99:>8.1f}ms  "
            f"{itl_50:>8.1f}ms {itl_99:>8.1f}ms  "
            f"{e2e_50 / 1000:>9.2f}s {e2e_99 / 1000:>9.2f}s"
        )
        print(line)

    print(f"{'=' * 130}")
    print()


def print_per_node(
    node_results: list[dict | None],
    endpoints: list[str],
    concurrency: int,
):
    """Print a compact per-node breakdown for one concurrency level."""
    print(f"    Per-node (concurrency={concurrency}):")
    for i, (r, ep) in enumerate(zip(node_results, endpoints)):
        if r is None:
            print(f"      node {i} ({ep}): FAILED")
            continue
        ok = r.get("completed", 0)
        tps = r.get("output_throughput", 0)
        p99_itl = r.get("p99_itl_ms", 0)
        print(
            f"      node {i} ({ep}): {ok} ok  {tps:.0f} tok/s  ITL-p99={p99_itl:.1f}ms"
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Distributed benchmark runner for hybrid-LB vLLM clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Using nodes file (resolves IPs via SSH):
  python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt
  python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt -c 32,64,128

  # Using explicit endpoints:
  python scripts/bench_distributed.py \\
      --endpoints http://10.0.0.1:8200,http://10.0.0.2:8200,http://10.0.0.3:8200,http://10.0.0.4:8200

  # Custom parameters:
  python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt \\
      --input-len 2048 --output-len 512 -c 64,128,256
""",
    )
    ep_group = parser.add_mutually_exclusive_group(required=True)
    ep_group.add_argument(
        "--nodes-file",
        help="Node hostnames file (one per line); IPs resolved via SSH",
    )
    ep_group.add_argument(
        "--endpoints",
        type=lambda s: [x.strip() for x in s.split(",")],
        help="Comma-separated endpoint URLs (e.g. http://IP1:8200,http://IP2:8200)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8200,
        help="Port for node endpoints (used with --nodes-file, default: 8200)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from first endpoint if omitted)",
    )
    parser.add_argument(
        "-c",
        "--concurrencies",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 8, 32, 64, 128, 256],
        help="Total concurrency levels (split across nodes, default: 1,8,32,64,128,256)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Prompts per level is set to the concurrency level",
    )
    parser.add_argument(
        "-n",
        "--num-prompts",
        type=int,
        default=None,
        help="Total prompts per level (default: max(concurrency*2, 128); split across nodes)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Input tokens per request (default: 1024)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Output tokens per request (default: 1024)",
    )
    parser.add_argument(
        "--result-dir",
        default="results",
        help="Directory for result JSON files (default: results)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Result filename prefix (auto-generated if omitted)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-node breakdown for each concurrency level",
    )
    args = parser.parse_args()

    # Resolve endpoints
    if args.nodes_file:
        print("Resolving node IPs...")
        endpoints = resolve_endpoints(args.nodes_file, args.port)
    else:
        endpoints = args.endpoints

    num_nodes = len(endpoints)
    if num_nodes == 0:
        print("ERROR: no endpoints", file=sys.stderr)
        sys.exit(1)

    print(f"\nEndpoints ({num_nodes} nodes):")
    for i, ep in enumerate(endpoints):
        print(f"  [{i}] {ep}")

    # Auto-detect model from first endpoint
    model = args.model
    if not model:
        try:
            with urllib.request.urlopen(
                f"{endpoints[0]}/v1/models", timeout=10
            ) as resp:
                model = json.loads(resp.read())["data"][0]["id"]
        except Exception:
            print(
                "ERROR: could not detect model from server. Pass --model explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

    os.makedirs(args.result_dir, exist_ok=True)

    if not args.label:
        args.label = f"dist_in{args.input_len}_out{args.output_len}"

    print(f"\nModel         : {model}")
    print(f"Input tokens  : {args.input_len}")
    print(f"Output tokens : {args.output_len}")
    print(f"Concurrencies : {args.concurrencies}")
    print(f"Nodes         : {num_nodes}")
    print()

    all_results: list[tuple[int, dict]] = []

    for conc in args.concurrencies:
        factor = 2 if not args.single else 1
        total_prompts = args.num_prompts or max(conc * factor, 128)
        # Split prompts evenly across nodes (round up so we don't lose any)
        per_node_prompts = math.ceil(total_prompts / num_nodes)
        # Per-node concurrency: split total concurrency across nodes
        per_node_conc = max(math.ceil(conc / num_nodes), 1)

        print(
            f"  concurrency={conc:>4}  prompts={total_prompts:>5} "
            f"({per_node_prompts}/node × {num_nodes})  "
            f"conc/node={per_node_conc} ... ",
            end="",
            flush=True,
        )

        # Run benchmarks in parallel across all nodes
        node_results: list[dict | None] = [None] * num_nodes
        with ProcessPoolExecutor(max_workers=num_nodes) as pool:
            futures = {}
            for i, ep in enumerate(endpoints):
                fut = pool.submit(
                    run_bench_on_node,
                    ep,
                    model,
                    args.input_len,
                    args.output_len,
                    per_node_conc,
                    per_node_prompts,
                    args.result_dir,
                    args.label,
                    i,
                )
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    node_results[idx] = fut.result()
                except Exception as e:
                    print(f"\n  [node {idx}] exception: {e}", file=sys.stderr)

        # Aggregate
        agg = aggregate_results(node_results)
        if agg:
            ok = agg.get("completed", 0)
            fail = agg.get("failed", 0)
            tps = agg.get("output_throughput", 0)
            print(f"{ok}/{ok + fail} ok  {tps:.0f} tok/s")
            all_results.append((conc, agg))
        else:
            print("FAILED (all nodes failed)")

        if args.verbose:
            print_per_node(node_results, endpoints, conc)

        # Save per-node results for this concurrency
        per_node_path = os.path.join(
            args.result_dir, f"{args.label}_per_node_{conc}.json"
        )
        with open(per_node_path, "w") as f:
            json.dump(
                {
                    "concurrency": conc,
                    "endpoints": endpoints,
                    "per_node": [
                        {"endpoint": ep, "result": r}
                        for ep, r in zip(endpoints, node_results)
                    ],
                },
                f,
                indent=2,
            )

        # Brief pause between levels
        if conc != args.concurrencies[-1]:
            time.sleep(5)

    if all_results:
        print_summary(all_results, args.input_len, args.output_len, num_nodes)

        # Save combined results
        combined_path = os.path.join(args.result_dir, f"{args.label}_combined.json")
        out = {
            "config": {
                "endpoints": endpoints,
                "num_nodes": num_nodes,
                "input_len": args.input_len,
                "output_len": args.output_len,
            },
            "results": [
                {"concurrency": c, **{k: r[k] for k in r}} for c, r in all_results
            ],
        }
        with open(combined_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
