# Decode Benchmark — Runbook

Benchmark decode throughput of `zai-org/GLM-5-FP8` in isolation using
vLLM's `DecodeBenchConnector`. No prefill cluster, no NIXL, no proxy needed.

Two modes:
- **Internal LB** (default): `launch_bench.sh` + `bench.py` — single API server on head node, workers headless
- **Hybrid LB**: `launch_bench_hybrid.sh` + `bench_distributed.py` — every node runs its own API server, benchmark client distributes load

Last verified: **2 March 2026** on H200 HGX-8 cluster.

---

## How it works

Normal disaggregated serving requires a running prefill cluster to populate the
KV cache before decode can start. For benchmarking, we skip all of that:

```
┌─────────────────────────────────────────────────────┐
│  DecodeBenchConnector                               │
│  Fills KV cache with random values (fill_mean=0.015)│
│  No prefill cluster or NIXL needed                  │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │ Decode cluster        │
         │ N nodes, EP=N*8       │
         │ FULL_DECODE_ONLY CG   │
         │ port 8200             │
         └───────────▲───────────┘
                     │
         ┌───────────┴───────────┐
         │ vllm bench serve      │
         │ random dataset        │
         │ sweep concurrencies   │
         └───────────────────────┘
```

**Key flags on the server that make this work:**

| Flag | Value | Purpose |
|------|-------|---------|
| `--kv-transfer-config` | `DecodeBenchConnector` | Fills KV with random values so decode sees a realistic ISL without needing a prefill pass |
| `--compilation-config` | `{"cudagraph_mode": "FULL_DECODE_ONLY"}` | Full CUDA graphs for decode, no prefill graph specialization (saves GPU memory and startup time) |

The `DecodeBenchConnector` works by intercepting the scheduler: on first schedule
of a request, it claims all input tokens as "already prefilled" and fills the
corresponding KV cache blocks with `torch.full(fill_mean)`. The model then
runs decode-only forward passes against this synthetic KV.

---

## Quick start

```bash
# 1. Create a nodes file (only decode nodes, NOT prefill nodes)
cat > /tmp/bench_nodes.txt << 'EOF'
ltc-idc3-hgx8-h200-12
ltc-idc3-hgx8-h200-35
ltc-idc3-hgx8-h200-36
ltc-idc3-hgx8-h200-37
EOF

# 2. Start the server
bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt

# 3. Run benchmarks (use the URL printed by launch_bench.sh)
python scripts/bench.py --base-url http://10.20.0.18:8200

# 4. Stop the server
bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt --stop
```

---

## Server: launch_bench.sh

### Start

```bash
bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt
```

This will:
1. Kill any stale processes on all listed nodes
2. Launch the decode head on the first node
3. Launch headless workers on the remaining nodes
4. Wait for the server to become healthy (~7 min for 4 nodes)
5. Print the endpoint URL

### Start with options

```bash
# Single node (EP=8)
echo "ltc-idc3-hgx8-h200-35" > /tmp/one_node.txt
bash scripts/launch_bench.sh --nodes-file /tmp/one_node.txt

# Two nodes (EP=16)
bash scripts/launch_bench.sh --nodes-file /tmp/two_nodes.txt --decode-nodes 2

# NCCL backend instead of NVSHMEM
bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt \
    --a2a-backend deepep_high_throughput
```

### Stop

```bash
bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt --stop
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--nodes-file FILE` | (required) | Node hostnames, one per line |
| `--decode-nodes N` | all nodes in file | How many nodes to use |
| `--a2a-backend` | `deepep_low_latency` | `deepep_low_latency` or `deepep_high_throughput` |
| `--model` | `zai-org/GLM-5-FP8` | Model to serve |
| `--max-model-len` | `32768` | Max context length |
| `--stop` | — | Kill all bench processes and exit |

---

## Benchmark: bench.py

### Basic run

```bash
python scripts/bench.py --base-url http://10.20.0.18:8200
```

### Custom concurrency sweep

```bash
python scripts/bench.py --base-url http://10.20.0.18:8200 -c 32,64,128,256,512
```

### Custom input/output lengths

```bash
# Long-context decode
python scripts/bench.py --base-url http://10.20.0.18:8200 \
    --input-len 4096 --output-len 2048

# Short decode (latency-focused)
python scripts/bench.py --base-url http://10.20.0.18:8200 \
    --input-len 512 --output-len 128
```

### More requests per concurrency level

```bash
# 500 requests per level (default: max(concurrency*2, 128))
python scripts/bench.py --base-url http://10.20.0.18:8200 -n 500
```

### Custom result label

```bash
python scripts/bench.py --base-url http://10.20.0.18:8200 \
    --label "ep32_ll_run1"
# Saves results/ep32_ll_run1_1.json, results/ep32_ll_run1_8.json, etc.
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url URL` | (required) | Server endpoint |
| `--model` | auto-detected | Model name override |
| `-c`, `--concurrencies` | `1,8,32,64,128,256` | Comma-separated concurrency levels |
| `-n`, `--num-prompts` | `max(c*2, 128)` | Requests per concurrency level |
| `--input-len` | `1024` | Input tokens per request |
| `--output-len` | `1024` | Output tokens per request |
| `--result-dir` | `results` | Where to save JSON results |
| `--label` | auto-generated | Result filename prefix |

---

## Output

### Console table

```
==================================================================================================================================
  Decode benchmark   input=1024  output=1024
==================================================================================================================================
 Conc     OK Fail   Out tok/s   Req/s   TTFT p50  TTFT p99   TPOT p50  TPOT p99    ITL p50   ITL p99     E2E p50    E2E p99
----------------------------------------------------------------------------------------------------------------------------------
    1    128    0       320.5    0.31       52ms      120ms      3.1ms      4.2ms      3.0ms      3.8ms       3.20s      3.95s
    8    128    0      2450.3    2.39       80ms      200ms      3.2ms      5.1ms      3.1ms      4.5ms       3.35s      4.10s
   32    128    0      8120.7    7.93      250ms      800ms      3.5ms      8.3ms      3.3ms      7.0ms       3.62s      5.50s
   ...
==================================================================================================================================
```

### Metrics explained

| Metric | What it measures |
|--------|-----------------|
| **Out tok/s** | Total output token throughput (all requests / wall time) |
| **Req/s** | Request throughput |
| **TTFT** | Time to first token — includes KV fill time from `DecodeBenchConnector` |
| **TPOT** | Time per output token (E2E minus TTFT, divided by output tokens) |
| **ITL** | Inter-token latency (time between consecutive tokens) |
| **E2E** | End-to-end request latency |

### Result files

Each concurrency level saves a JSON file in `results/`:
```
results/decode_in1024_out1024_1.json
results/decode_in1024_out1024_8.json
results/decode_in1024_out1024_32.json
...
```

---

## Hybrid LB mode

Internal LB sends all traffic through a single head node API server, which can
bottleneck at high concurrency. **Hybrid LB** (`--data-parallel-hybrid-lb`)
gives each node its own API server scheduling only its 8 co-located DP ranks.
The distributed benchmark client splits load across all endpoints.

### Quick start (hybrid)

```bash
# 1. Start hybrid server (all 4 nodes serve on :8200)
bash scripts/launch_bench_hybrid.sh --nodes-file /tmp/bench_nodes.txt

# 2. Run distributed benchmarks (splits load across all nodes)
python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt

# 3. Stop
bash scripts/launch_bench_hybrid.sh --nodes-file /tmp/bench_nodes.txt --stop
```

### Server: launch_bench_hybrid.sh

Same options as `launch_bench.sh`. Key differences:
- Every node (including workers) runs an API server on port 8200
- `--data-parallel-hybrid-lb` flag on all nodes
- Workers are NOT `--headless`
- Health check waits for ALL nodes, not just head

### Distributed benchmark: bench_distributed.py

Accepts either `--nodes-file` (resolves IPs) or `--endpoints` (explicit URLs).

```bash
# Default sweep
python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt

# Custom concurrency
python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt -c 32,64,128,256

# Explicit endpoints
python scripts/bench_distributed.py \
    --endpoints http://10.0.0.1:8200,http://10.0.0.2:8200,http://10.0.0.3:8200,http://10.0.0.4:8200

# Verbose per-node breakdown
python scripts/bench_distributed.py --nodes-file /tmp/bench_nodes.txt --verbose
```

**How it works:**
- Splits total prompts evenly across N nodes
- Splits total concurrency evenly across N nodes
- Runs `vllm bench serve` in parallel (one subprocess per node)
- Aggregates: sum throughput, weighted-mean p50, max p99 (conservative)
- Saves per-node results for debugging (`results/*_per_node_*.json`)

| Flag | Default | Description |
|------|---------|-------------|
| `--nodes-file FILE` | — | Node hostnames (resolves IPs via SSH) |
| `--endpoints URLs` | — | Comma-separated endpoint URLs |
| `--port` | `8200` | Port (used with --nodes-file) |
| `--model` | auto-detected | Model name override |
| `-c`, `--concurrencies` | `1,8,32,64,128,256` | Total concurrency levels |
| `-n`, `--num-prompts` | `max(c*2, 128)` | Total requests per level (split across nodes) |
| `--input-len` | `1024` | Input tokens per request |
| `--output-len` | `1024` | Output tokens per request |
| `--result-dir` | `results` | Where to save JSON results |
| `--label` | auto-generated | Result filename prefix |
| `--verbose` | off | Print per-node breakdown |

### When to use hybrid vs internal LB

| | Internal LB | Hybrid LB |
|---|---|---|
| Script | `launch_bench.sh` + `bench.py` | `launch_bench_hybrid.sh` + `bench_distributed.py` |
| API servers | 1 (head only) | N (one per node) |
| Scheduling bottleneck | Head node serializes all 32 ranks | Each node schedules its 8 ranks |
| Best for | Low-mid concurrency, simple setup | High concurrency, max throughput |
| Client | Single endpoint | Distributed across nodes |

---

## Running alongside the disagg cluster

The bench server uses a **different RPC port** (29560 vs 29550) so it won't
interfere with a running disagg cluster — but it **cannot share GPU nodes**.

| | Disagg (`launch_disagg.sh`) | Bench (`launch_bench.sh`) |
|---|---|---|
| HTTP port | 8200 | 8200 |
| RPC port | 29550 | 29560 |
| Log dir | `/tmp/vllm_disagg_logs/` | `/tmp/vllm_bench_logs/` |
| KV connector | `NixlConnector` | `DecodeBenchConnector` |
| CUDA graphs | `PIECEWISE` (default) | `FULL_DECODE_ONLY` |

Allocate separate SLURM jobs with separate nodes for each.

---

## Troubleshooting

### Server won't start (TIMEOUT)

Check the head log:
```bash
ssh <head_node> 'tail -50 /tmp/vllm_bench_logs/bench_head.log'
```

Common causes:
- Stale processes from a prior run — `launch_bench.sh` cleans automatically, but if you
  ctrl-C'd a previous launch, run `--stop` first
- Wrong nodes file (nodes not in SLURM allocation)
- GPU OOM

### All requests fail

```bash
# Check server is healthy
curl http://<head_ip>:8200/health

# Check model name matches
curl http://<head_ip>:8200/v1/models
```

### Logs

```bash
# Bench server head
ssh <node> 'tail -50 /tmp/vllm_bench_logs/bench_head.log'

# Bench server worker N
ssh <node> 'tail -50 /tmp/vllm_bench_logs/bench_worker_N.log'
```

---

## File reference

| File | Purpose |
|------|---------|
| `scripts/launch_bench.sh` | Start/stop decode-bench server (internal LB) |
| `scripts/bench.py` | Run benchmarks against a single endpoint |
| `scripts/launch_bench_hybrid.sh` | Start/stop decode-bench server (hybrid LB) |
| `scripts/bench_distributed.py` | Distributed benchmarks across multiple endpoints |
| `results/*.json` | Per-concurrency result files from `vllm bench serve` |
| `/tmp/vllm_bench_logs/` | Server logs on remote nodes |

---

## Quick reference

```bash
# Allocate nodes
salloc -N 4 -p cluster -w node1,node2,node3,node4

# Create nodes file
cat > /tmp/bench_nodes.txt << 'EOF'
node1
node2
node3
node4
EOF

# Start server
bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt

# Run benchmarks (use URL from launch output)
python scripts/bench.py --base-url http://<HEAD_IP>:8200
python scripts/bench.py --base-url http://<HEAD_IP>:8200 -c 32,64,128,256

# Stop server
bash scripts/launch_bench.sh --nodes-file /tmp/bench_nodes.txt --stop
```
