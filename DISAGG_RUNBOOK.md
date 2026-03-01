# Disaggregated Prefill/Decode Serving — Runbook

Production deployment of `zai-org/GLM-5-FP8` on vLLM 0.16 with Expert Parallelism,
disaggregated prefill/decode, and KV transfer over NIXL/UCX/InfiniBand.

Last verified: **2 March 2026** on H200 HGX-8 cluster.

---

## Architecture

```
                   ┌────────────────────────┐
                   │  Proxy (localhost:8000) │
                   └──────┬─────────┬───────┘
                     step 1│         │step 3
              ┌────────────▼──┐  ┌───▼──────────────┐
              │ Prefill (P)   │  │ Decode (D)        │
              │ 2 nodes EP=16 │  │ 4 nodes EP=32     │
              │ port 8100     │  │ port 8200         │
              │ HT backend    │  │ LL backend        │
              └───────┬───────┘  └───────▲───────────┘
                      │    step 2: RDMA   │
                      └───────────────────┘
                        NIXL / UCX / IB
```

**Request flow:**

1. Proxy sends request to P with `max_tokens=1` + `kv_transfer_params: {do_remote_decode: true}`
2. P runs full prefill, returns `kv_transfer_params` (block IDs, engine ID, host:port)
3. Proxy injects those params into the original request, sends to D
4. D's NixlConnector RDMA-fetches the KV cache from P's GPU memory
5. D runs decode, streams response back through proxy

### Modes

| Mode | Nodes | Model | Prefill | Decode | A2A |
|------|-------|-------|---------|--------|-----|
| `dev`  | 2 (1P + 1D) | `samsja/mini-glm-moe` | EP=8 | EP=8 | both `deepep_high_throughput` |
| `prod` | 6 (2P + 4D) | `zai-org/GLM-5-FP8`   | EP=16 | EP=32 | P: `deepep_high_throughput` / D: `deepep_low_latency` |

### Node roles (prod)

| Role | Count | GPUs | Flags | Notes |
|------|-------|------|-------|-------|
| Prefill head | 1 node | 8 | `--host 0.0.0.0 --port 8100` | Serves HTTP, DP ranks 0-7 |
| Prefill worker | 1 node | 8 | `--headless --data-parallel-start-rank 8` | No HTTP server |
| Decode head | 1 node | 8 | `--host 0.0.0.0 --port 8200` | Serves HTTP, DP ranks 0-7, spawns 32 API servers |
| Decode workers | 3 nodes | 24 | `--headless --data-parallel-start-rank {8,16,24}` | No HTTP server |
| Proxy | local | — | — | aiohttp, routes P→D |

---

## Prerequisites

### Hardware
- 6 H200 HGX-8 nodes (SM90, 8 GPUs x 141 GB each)
- InfiniBand between nodes (only `mlx5_0` device works)
- Nodes must have `NVreg_EnableStreamMemOPs=1` (listed in `valid_nodes.txt`)

### Software (shared NFS)
- vLLM 0.16.0 in `.venv/`
- `deep_ep` with NVSHMEM (`ep_kernels_workspace/nvshmem/lib/libnvshmem_host.so.3`)
- `nixl==0.10.0` (`nixl-cu12` package)
- `aiohttp` (for proxy)
- Model weights at `/shared/huggingface`

```bash
bash scripts/install_disagg_deps.sh
```

---

## Launch procedure

### 1. Allocate nodes via SLURM

```bash
salloc -N 6 --partition=cluster --job-name=disagg_prod \
  -w ltc-idc3-hgx8-h200-10,ltc-idc3-hgx8-h200-11,ltc-idc3-hgx8-h200-12,ltc-idc3-hgx8-h200-35,ltc-idc3-hgx8-h200-36,ltc-idc3-hgx8-h200-37
```

Pick any 6 nodes from `valid_nodes.txt`.

### 2. Create a nodes file matching your allocation

**This is critical.** The launch script reads nodes from a file. If you use
`valid_nodes.txt` directly, it picks the first N nodes alphabetically — these
may not be in your SLURM allocation, and the launch will silently start processes
on unallocated nodes that have no GPU access.

```bash
cat > /tmp/disagg_nodes.txt << 'EOF'
ltc-idc3-hgx8-h200-10
ltc-idc3-hgx8-h200-11
ltc-idc3-hgx8-h200-12
ltc-idc3-hgx8-h200-35
ltc-idc3-hgx8-h200-36
ltc-idc3-hgx8-h200-37
EOF
```

**Order matters:** lines 1-2 = prefill, lines 3-6 = decode.

### 3. Kill stale processes (MANDATORY before every launch)

**This is the single most important step.** Stale vLLM processes from prior runs
poison the torch.distributed Gloo process group: new EngineCore processes
accidentally form groups with dead peers, then crash during CUDA graph capture.
The head then waits 10 minutes before timing out. There is no graceful recovery.

```bash
for node in $(cat /tmp/disagg_nodes.txt); do
  echo "=== $node ==="
  ssh -o StrictHostKeyChecking=no "$node" \
    'pkill -9 -f "vllm.entrypoints" 2>/dev/null
     pkill -9 -f EngineCore 2>/dev/null
     pkill -9 -f ApiServer 2>/dev/null
     pkill -9 -f DPCoordinator 2>/dev/null
     rm -rf /tmp/vllm_disagg_logs/
     echo cleaned' 2>/dev/null || echo "  (unreachable)"
done
pkill -f disagg_proxy.py 2>/dev/null || true
```

**Do not use `pkill -f vllm`** — the SSH command string contains the path
`/home/.../vllm-benchmarking/.venv/bin/vllm`, so `pkill -f vllm` matches and
kills the SSH session itself. Always use the specific patterns above.

### 4. Launch

```bash
bash scripts/launch_disagg.sh --mode prod --nodes-file /tmp/disagg_nodes.txt
```

Startup takes **~7 minutes**:
- Prefill: ~3 min (model load + DeepGEMM warmup + torch.compile + CUDA graphs)
- Decode: ~6-7 min (same pipeline, but 32-rank Gloo group + 32 API servers)

### 5. Verify

```bash
# Health checks
curl -s http://<PREFILL_IP>:8100/health
curl -s http://<DECODE_IP>:8200/health

# Test full P→D flow through proxy
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-5-FP8","prompt":"Hello","max_tokens":50}' \
  | python3 -m json.tool
```

---

## Chat

```bash
# Interactive chat (auto-detects model from server)
python scripts/chat.py

# One-shot
python scripts/chat.py "What is quantum computing?"

# Raw completion mode (text continuation, not chat)
python scripts/chat.py --completions "The meaning of life is"

# System prompt + shorter output
python scripts/chat.py --system "Answer in one sentence." --max-tokens 64 "What is gravity?"

# Direct to a specific endpoint (bypass proxy)
python scripts/chat.py --host 10.20.0.18 --port 8200 "Hello"
```

---

## Stop

```bash
bash scripts/launch_disagg.sh --stop --nodes-file /tmp/disagg_nodes.txt
```

---

## Environment variables

Every vLLM process needs these. They are set inside the SSH heredocs in `launch_disagg.sh`.

| Variable | Value | Why |
|----------|-------|-----|
| `NCCL_NET_PLUGIN=none` | Prevents NCCL from loading HPC-X UCX plugin | NCCL's UCX plugin installs UCM hooks globally, blocking NIXL's bundled UCX from registering RDMA memory. Without this: `NIXL_ERR_BACKEND`, `rcache failed to install UCM event handler`. NCCL still works via its built-in IB verbs transport. |
| `UCX_NET_DEVICES=mlx5_0:1` | Only working IB device | Other mlx5 ports are not connected on this cluster. |
| `UCX_TLS=rc,dc,self,sm,cma,cuda_ipc,cuda_copy` | Transport selection | Ensures reliable connected + direct-connect for RDMA. |
| `UCX_RCACHE_MAX_UNRELEASED=1024` | Registration cache limit | Must be set before importing nixl to prevent memory leak. |
| `GLOO_SOCKET_IFNAME=bond0` | Ethernet interface for gloo | These nodes use `bond0`, **not** `eth0`. Without this: `RuntimeError: Unable to find address for: eth0`. |
| `DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache` | Local DeepGEMM JIT cache | NFS-shared `~/.cache/vllm/deep_gemm/` gets `Stale file handle` when 16+ ranks JIT-compile simultaneously. |
| `TOKENIZERS_PARALLELISM=false` | Suppress HuggingFace warning | Cosmetic. |
| `HF_HUB_OFFLINE=1` | Don't hit HuggingFace Hub | Model is cached at `/shared/huggingface`. |
| `LD_LIBRARY_PATH` | Must include `ep_kernels_workspace/nvshmem/lib` | `deep_ep_cpp.so` dynamically links `libnvshmem_host.so.3`. |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | Node's own IP | NIXL side-channel for KV metadata exchange. |
| `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` | Base port | Each DP rank offsets from this (5600, 5601, ...). |

---

## Tricks and caveats

### DeepGEMM cache seeding

First-ever run JIT-compiles ~4600 CUDA kernels → **~25 minutes**. After that,
compiled kernels live in `~/.cache/vllm/deep_gemm/` (NFS, ~39 MB).

The launch script copies this cache to `/tmp` before starting each vLLM process:
```bash
rsync -a --ignore-existing ~/.cache/vllm/deep_gemm/ /tmp/deep_gemm_cache/
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache
```

With the seeded cache, warmup takes **~60s**. The very first launch ever on a
fresh cluster will be slow; all subsequent ones are fast.

### The `--headless` flag

Worker nodes must use `--headless`. This means:
- No HTTP server (no `--host`/`--port` flags)
- EngineCore processes connect to the head node via `--data-parallel-address`
- Head sends config (including Gloo ports) during ZMQ handshake

Without `--headless`: `RuntimeError: Remote engine N must use --headless`

### 32 API servers on the decode head

The decode head spawns **32 API server processes** — one per DP rank — for
internal load balancing. This is by design. Don't be alarmed by 32 `ApiServer_N`
entries in the decode head log. They are lightweight HTTP+ZMQ forwarders.

### Proxy kv_transfer_params injection

The prefill server only returns `kv_transfer_params` in its response if the
**request** includes `kv_transfer_params: {"do_remote_decode": true}`. The proxy
does this automatically. If you query prefill directly for testing:

```bash
curl ... -d '{"prompt":"...","max_tokens":1,"kv_transfer_params":{"do_remote_decode":true}}'
```

### Gloo port synchronization

Each node independently generates random ports for the Gloo DP process group.
The head sends the correct ports to all workers during the ZMQ startup handshake
(`EngineHandshakeMetadata` includes `data_parallel_master_port` and
`_data_parallel_master_port_list`). This works automatically — no manual port
configuration needed.

### GPU memory utilization

- Dev mode: `0.80` (safe default)
- Prod mode: `0.95` (needed for GLM-5-FP8 to fit; lower values OOM)

---

## Troubleshooting

### Checking logs

Logs are on each remote node at `/tmp/vllm_disagg_logs/`:

```bash
ssh <node> 'tail -50 /tmp/vllm_disagg_logs/vllm_prefill.log'
ssh <node> 'tail -50 /tmp/vllm_disagg_logs/vllm_decode_head.log'
ssh <node> 'tail -50 /tmp/vllm_disagg_logs/vllm_decode_worker_1.log'
tail -50 /tmp/vllm_disagg_logs/disagg_proxy.log  # local
```

**Healthy startup sequence in the log:**
```
Loading safetensors checkpoint shards: 100% ...     # model load (~18s)
Loading weights took 17.96 seconds
Using DeepEPLLPrepareAndFinalize                    # EP backend selected
NIXL INFO  Initialized NIXL agent: <uuid>           # one per DP rank
Registering KV_Caches. use_mla: True                # KV cache registered with NIXL
DeepGEMM warmup: 100%|██████████| 4628/4628         # JIT warmup (~60s with cache)
Capturing CUDA graphs ... 100%                      # CUDA graph capture
torch.compile took 93.11s                           # compilation
Application startup complete.                       # READY
```

### Error reference

| Error | Cause | Fix |
|-------|-------|-----|
| `DistStoreError: wait timeout after 600000ms` | Gloo DP group can't form — stale processes hold the port or not all ranks joined | Kill ALL stale processes on ALL nodes, relaunch |
| `Connection closed by peer [wrong_ip]:port` | Stale Gloo connection to a dead process from a prior run | Kill stale processes, relaunch |
| `Unable to find address for: eth0` | Wrong `GLOO_SOCKET_IFNAME` | Must be `bond0` on this cluster |
| `NVCC compilation failed: Stale file handle` | NFS race in DeepGEMM JIT | Set `DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache` |
| `NIXL_ERR_BACKEND` / `rcache failed to install UCM` | NCCL loaded HPC-X UCX plugin | Set `NCCL_NET_PLUGIN=none` |
| `Remote engine N must use --headless` | Worker missing `--headless` flag | Already fixed in launch script |
| `Did not receive response from front-end within 5 min` | Worker can't reach head's ZMQ ROUTER | Check `--data-parallel-address`, network connectivity |
| `Timed out waiting for engines to send initial message` | API server waiting for EngineCore that crashed | Check EngineCore logs for the real error |
| GPU OOM | `--gpu-memory-utilization` too low | Use `0.95` for prod |
| `P returned no kv_transfer_params` (proxy log) | Prefill request missing `do_remote_decode` flag | Already fixed in proxy |

---

## File reference

| File | Purpose |
|------|---------|
| `scripts/launch_disagg.sh` | SSH orchestrator — launches all P/D nodes + proxy |
| `scripts/disagg_proxy.py` | aiohttp proxy — routes P→D, injects kv_transfer_params |
| `scripts/chat.py` | Interactive chat / one-shot query client |
| `scripts/install_disagg_deps.sh` | Installs nixl, aiohttp, verifies deep_ep + NVSHMEM |
| `valid_nodes.txt` | All 22 cluster nodes with correct GPU config |
| `/tmp/disagg_nodes.txt` | Your SLURM-allocated subset (create before each launch) |
| `DISAGG_RUNBOOK.md` | This file |

---

## Quick reference

```bash
# 1. Allocate
salloc -N 6 -p cluster -w node1,node2,node3,node4,node5,node6

# 2. Create nodes file (first 2 = prefill, next 4 = decode)
cat > /tmp/disagg_nodes.txt << 'EOF'
node1
node2
node3
node4
node5
node6
EOF

# 3. Clean (MANDATORY)
for n in $(cat /tmp/disagg_nodes.txt); do
  ssh -o StrictHostKeyChecking=no "$n" \
    'pkill -9 -f "vllm.entrypoints" 2>/dev/null
     pkill -9 -f EngineCore 2>/dev/null
     pkill -9 -f ApiServer 2>/dev/null
     rm -rf /tmp/vllm_disagg_logs/
     echo ok' 2>/dev/null
done

# 4. Launch
bash scripts/launch_disagg.sh --mode prod --nodes-file /tmp/disagg_nodes.txt

# 5. Chat
python scripts/chat.py

# 6. Stop
bash scripts/launch_disagg.sh --stop --nodes-file /tmp/disagg_nodes.txt
```
