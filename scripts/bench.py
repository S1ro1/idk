#!/usr/bin/env python3
"""
Run decode benchmarks against a running vLLM server.

Runs vllm bench serve at each requested concurrency level and prints
a summary table.  Pair with launch_bench.sh to start the server.

Usage:
  python scripts/bench.py --base-url http://10.20.0.18:8200
  python scripts/bench.py --base-url http://10.20.0.18:8200 -c 32,64,128,256
  python scripts/bench.py --base-url http://10.20.0.18:8200 --input-len 2048 --output-len 512
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VLLM_BIN = str(REPO_ROOT / ".venv/bin/vllm")


# ── Benchmark runner ─────────────────────────────────────────────────────────


def run_bench(
    base_url: str,
    model: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_prompts: int,
    result_dir: str,
    label: str,
) -> dict | None:
    """Run vllm bench serve for one concurrency level, return parsed result."""
    filename = f"{label}_{concurrency}.json"
    cmd = [
        VLLM_BIN,
        "bench",
        "serve",
        "--base-url",
        base_url,
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
        "inf",
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
        # "--disable-tqdm",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  bench FAILED (exit {proc.returncode})", file=sys.stderr)
        stderr_tail = proc.stderr[-500:] if proc.stderr else ""
        if stderr_tail:
            print(f"  stderr: {stderr_tail}", file=sys.stderr)
        return None

    result_path = os.path.join(result_dir, filename)
    if not os.path.exists(result_path):
        print(f"  result file not found: {result_path}", file=sys.stderr)
        return None

    with open(result_path) as f:
        return json.load(f)


# ── Output formatting ────────────────────────────────────────────────────────


def print_summary(results: list[tuple[int, dict]], input_len: int, output_len: int):
    print()
    print(f"{'=' * 130}")
    print(f"  Decode benchmark   input={input_len}  output={output_len}")
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


def save_combined(results: list[tuple[int, dict]], path: str, args: argparse.Namespace):
    out = {
        "config": {
            "base_url": args.base_url,
            "input_len": args.input_len,
            "output_len": args.output_len,
        },
        "results": [{"concurrency": c, **{k: r[k] for k in r}} for c, r in results],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Combined results saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run decode benchmarks against a running vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Start server first:
  bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt

  # Then benchmark:
  python scripts/bench.py --base-url http://10.20.0.18:8200
  python scripts/bench.py --base-url http://10.20.0.18:8200 -c 32,64,128,256
  python scripts/bench.py --base-url http://10.20.0.18:8200 --input-len 4096 --output-len 2048

  # Stop server when done:
  bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt --stop
""",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server URL (e.g. http://10.20.0.18:8200)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from /v1/models if omitted)",
    )
    parser.add_argument(
        "-c",
        "--concurrencies",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 8, 32, 64, 128, 256],
        help="Concurrency levels (default: 1,8,32,64,128,256)",
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
        help="Prompts per level (default: max(concurrency*2, 128))",
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
    args = parser.parse_args()

    # Auto-detect model
    model = args.model
    if not model:
        try:
            with urllib.request.urlopen(
                f"{args.base_url}/v1/models", timeout=10
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
        args.label = f"decode_in{args.input_len}_out{args.output_len}"

    print(f"Target        : {args.base_url}")
    print(f"Model         : {model}")
    print(f"Input tokens  : {args.input_len}")
    print(f"Output tokens : {args.output_len}")
    print(f"Concurrencies : {args.concurrencies}")
    print()

    all_results: list[tuple[int, dict]] = []

    for conc in args.concurrencies:
        factor = 2 if not args.single else 1
        n_prompts = args.num_prompts or max(conc * factor, 128)
        print(
            f"  concurrency={conc:>4}  prompts={n_prompts:>5} ... ", end="", flush=True
        )

        result = run_bench(
            args.base_url,
            model,
            args.input_len,
            args.output_len,
            conc,
            n_prompts,
            args.result_dir,
            args.label,
        )

        if result:
            ok = result.get("completed", 0)
            fail = result.get("failed", 0)
            tps = result.get("output_throughput", 0)
            print(f"{ok}/{ok + fail} ok  {tps:.0f} tok/s")
            all_results.append((conc, result))
        else:
            print("FAILED")

        # brief pause between levels
        if conc != args.concurrencies[-1]:
            time.sleep(5)

    if all_results:
        print_summary(all_results, args.input_len, args.output_len)


if __name__ == "__main__":
    main()
