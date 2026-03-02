#!/usr/bin/env bash
# =============================================================================
# launch_bench_hybrid.sh — Launch decode-only vLLM cluster with hybrid LB
# =============================================================================
#
# Like launch_bench.sh, but uses --data-parallel-hybrid-lb so every node runs
# its own API server (scheduling only its co-located 8 DP ranks). An external
# load balancer or the distributed benchmark client distributes requests across
# all per-node endpoints.
#
# Usage
# ─────
#   bash scripts/launch_bench_hybrid.sh [OPTIONS]
#
# Options:
#   --nodes-file FILE     Node list (required)
#   --decode-nodes N      How many nodes to use (default: all nodes in file)
#   --a2a-backend BACKEND deepep_low_latency (default) | deepep_high_throughput
#   --model MODEL_ID      Override model (default: zai-org/GLM-5-FP8)
#   --max-model-len N     Override max context length (default: 32768)
#   --stop                Kill bench server processes on all nodes and exit
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── defaults ─────────────────────────────────────────────────────────────────
DO_STOP=0
NODES_FILE=""
DECODE_NODES_OVERRIDE=""
A2A_BACKEND="deepep_low_latency"
MODEL="zai-org/GLM-5-FP8"
MAX_MODEL_LEN=2048
GPU_MEM_UTIL=0.90
MAX_NUM_SEQS=2048

while [[ $# -gt 0 ]]; do
    case $1 in
        --nodes-file)     NODES_FILE="$2";            shift 2 ;;
        --decode-nodes)   DECODE_NODES_OVERRIDE="$2"; shift 2 ;;
        --a2a-backend)    A2A_BACKEND="$2";           shift 2 ;;
        --model)          MODEL="$2";                 shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2";         shift 2 ;;
        --max-num-seqs)   MAX_NUM_SEQS="$2";          shift 2 ;;
        --stop)           DO_STOP=1;                  shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$NODES_FILE" ]]; then
    echo "ERROR: --nodes-file is required" >&2
    echo "Usage: bash scripts/launch_bench_hybrid.sh --nodes-file /tmp/bench_nodes.txt" >&2
    exit 1
fi

# ── constants ────────────────────────────────────────────────────────────────
VLLM_BIN="$REPO_ROOT/.venv/bin/vllm"
NVSHMEM_LIB_DIR="$REPO_ROOT/ep_kernels_workspace/nvshmem/lib"
LOG_DIR="/tmp/vllm_bench_logs"
DECODE_PORT=8200
DP_RPC_PORT=29560   # different from disagg (29550)
DP_LOCAL=8

HF_HOME_VAL="/shared/huggingface"
HF_HUB_OFFLINE_VAL=1

BENCH_KV_CFG='{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}'
COMPILATION_CFG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

# ── helpers ──────────────────────────────────────────────────────────────────

read_all_nodes() {
    grep -v '^\s*$' "$NODES_FILE" | grep -v '^\s*#' | awk '{print $1}'
}

read_nodes() {
    local count="$1"
    read_all_nodes | head -n "$count"
}

get_node_ip() {
    ssh $SSH_OPTS -n "$1" \
        "hostname -I 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "$1"
}

wait_for_server() {
    local host="$1" port="$2" label="${3:-$1:$2}" timeout="${4:-900}"
    local elapsed=0
    echo -n "    Waiting for $label ."
    until curl -sf "http://$host:$port/health" >/dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed+5)); echo -n "."
        if (( elapsed >= timeout )); then
            echo " TIMEOUT (${timeout}s)"
            return 1
        fi
    done
    echo " ready (${elapsed}s)"
}

launch_remote() {
    local node="$1"
    local log_file="$2"
    local script_body
    script_body="$(cat)"

    echo "" >&2
    echo "--- Launching on $node  (log: $log_file) ---" >&2

    local remote_script="/tmp/vllm_bench_launch_$(date +%s%N).sh"
    ssh $SSH_OPTS "$node" \
        "mkdir -p '${LOG_DIR}'; cat > '${remote_script}'; chmod +x '${remote_script}'" \
        <<< "$script_body"

    ssh $SSH_OPTS -n "$node" \
        "nohup '${remote_script}' > '${log_file}' 2>&1 & echo \$!"
}

# =============================================================================
# --stop
# =============================================================================
if (( DO_STOP )); then
    echo "=== Stopping bench server processes ==="
    mapfile -t nodes < <(read_all_nodes)
    for node in "${nodes[@]}"; do
        echo -n "  [$node] "
        ssh $SSH_OPTS -n "$node" \
            'pids=$(ps aux | grep -E "VLLM::|vllm serve|vllm.entrypoints|multiprocessing.resource_tracker" | grep -v grep | awk "{print \$2}")
             if [ -n "$pids" ]; then echo "$pids" | xargs kill -9 2>/dev/null; fi
             echo done' \
            2>/dev/null || echo "(unreachable)"
    done
    echo "Done."
    exit 0
fi

# =============================================================================
# Node selection
# =============================================================================
TOTAL_AVAILABLE=$(read_all_nodes | wc -l)
if [[ -n "$DECODE_NODES_OVERRIDE" ]]; then
    NUM_DECODE_NODES="$DECODE_NODES_OVERRIDE"
else
    NUM_DECODE_NODES="$TOTAL_AVAILABLE"
fi

mapfile -t ALL_NODES < <(read_nodes "$NUM_DECODE_NODES")
if (( ${#ALL_NODES[@]} < NUM_DECODE_NODES )); then
    echo "ERROR: need $NUM_DECODE_NODES nodes, only ${#ALL_NODES[@]} in $NODES_FILE" >&2
    exit 1
fi

DECODE_EP=$(( NUM_DECODE_NODES * DP_LOCAL ))
HEAD="${ALL_NODES[0]}"

echo "================================================================"
echo " Decode benchmark server  [Hybrid LB — DecodeBenchConnector]"
echo " Model          : $MODEL"
echo " Nodes          : ${ALL_NODES[*]}"
echo " EP             : $DECODE_EP  (${NUM_DECODE_NODES} nodes × $DP_LOCAL GPUs)"
echo " A2A backend    : $A2A_BACKEND"
echo " max_model_len  : $MAX_MODEL_LEN"
echo " CUDAGraph mode : FULL_DECODE_ONLY"
echo " LB mode        : hybrid (each node serves on :$DECODE_PORT)"
echo "================================================================"

# ── Clean stale processes ────────────────────────────────────────────────────
echo ""
echo "=== Cleaning stale processes ==="
for node in "${ALL_NODES[@]}"; do
    echo -n "  [$node] "
    ssh $SSH_OPTS -n "$node" \
        'pids=$(ps aux | grep -E "VLLM::|vllm serve|vllm.entrypoints|multiprocessing.resource_tracker" | grep -v grep | awk "{print \$2}")
         if [ -n "$pids" ]; then echo "$pids" | xargs kill -9 2>/dev/null; fi
         rm -rf /tmp/vllm_bench_logs/ 2>/dev/null
         echo cleaned' \
        2>/dev/null || echo "(unreachable)"
done
sleep 2

# ── Resolve all node IPs ────────────────────────────────────────────────────
echo ""
echo "--- Resolving node IPs ---"
declare -a NODE_IPS
for i in "${!ALL_NODES[@]}"; do
    NODE_IPS[$i]="$(get_node_ip "${ALL_NODES[$i]}")"
    echo "  Node $i : ${ALL_NODES[$i]} (${NODE_IPS[$i]})"
done
HEAD_IP="${NODE_IPS[0]}"

# =============================================================================
# Launch head (rank 0)
# =============================================================================
echo ""
echo "=== Starting decode-bench server (hybrid LB) ==="

HEAD_LOG="$LOG_DIR/bench_head.log"
HEAD_PID=$(launch_remote "$HEAD" "$HEAD_LOG" <<HEAD_SCRIPT
export HF_HOME='${HF_HOME_VAL}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE_VAL}'
export LD_LIBRARY_PATH='${NVSHMEM_LIB_DIR}':\${LD_LIBRARY_PATH:-}
ulimit -n 65536
export NCCL_NET_PLUGIN=none
export GLOO_SOCKET_IFNAME=bond0
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache
rsync -a --ignore-existing /home/matej/.cache/vllm/deep_gemm/ /tmp/deep_gemm_cache/ 2>/dev/null || true
D_IP=\$(hostname -I | awk '{print \$1}')
exec '${VLLM_BIN}' serve '${MODEL}' \
    --host 0.0.0.0 \
    --port ${DECODE_PORT} \
    --tensor-parallel-size 1 \
    --data-parallel-size ${DECODE_EP} \
    --data-parallel-size-local ${DP_LOCAL} \
    --data-parallel-address \$D_IP \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --data-parallel-hybrid-lb \
    --enable-expert-parallel \
    --all2all-backend ${A2A_BACKEND} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --disable-log-requests \
    --kv-transfer-config '${BENCH_KV_CFG}' \
    --compilation-config '${COMPILATION_CFG}'
HEAD_SCRIPT
)
echo "  Head PID: $HEAD_PID"

# =============================================================================
# Launch workers (NOT headless — each runs its own API server)
# =============================================================================
for i in "${!ALL_NODES[@]}"; do
    (( i == 0 )) && continue
    WORKER_NODE="${ALL_NODES[$i]}"
    START_RANK=$(( i * DP_LOCAL ))
    WORKER_LOG="$LOG_DIR/bench_worker_${i}.log"

    WORKER_PID=$(launch_remote "$WORKER_NODE" "$WORKER_LOG" <<WORKER_SCRIPT
export HF_HOME='${HF_HOME_VAL}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE_VAL}'
export LD_LIBRARY_PATH='${NVSHMEM_LIB_DIR}':\${LD_LIBRARY_PATH:-}
ulimit -n 65536
export NCCL_NET_PLUGIN=none
export GLOO_SOCKET_IFNAME=bond0
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache
rsync -a --ignore-existing /home/matej/.cache/vllm/deep_gemm/ /tmp/deep_gemm_cache/ 2>/dev/null || true
exec '${VLLM_BIN}' serve '${MODEL}' \
    --host 0.0.0.0 \
    --port ${DECODE_PORT} \
    --tensor-parallel-size 1 \
    --data-parallel-size ${DECODE_EP} \
    --data-parallel-size-local ${DP_LOCAL} \
    --data-parallel-start-rank ${START_RANK} \
    --data-parallel-address ${HEAD_IP} \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --data-parallel-hybrid-lb \
    --enable-expert-parallel \
    --all2all-backend ${A2A_BACKEND} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --disable-log-requests \
    --kv-transfer-config '${BENCH_KV_CFG}' \
    --compilation-config '${COMPILATION_CFG}'
WORKER_SCRIPT
    )
    echo "  Worker $i PID: $WORKER_PID  start_rank=$START_RANK"
done

# =============================================================================
# Wait for ALL nodes to be healthy
# =============================================================================
echo ""
echo "=== Waiting for all nodes ==="
FAILED=0
for i in "${!ALL_NODES[@]}"; do
    if ! wait_for_server "${NODE_IPS[$i]}" "$DECODE_PORT" "Node $i (${ALL_NODES[$i]})" 900; then
        FAILED=1
    fi
done

if (( FAILED )); then
    echo ""
    echo "ERROR: one or more nodes failed health check" >&2
    echo "Check logs: ssh <node> 'tail -50 $LOG_DIR/bench_*.log'" >&2
    exit 1
fi

echo ""
echo "================================================================"
echo " Decode benchmark server is UP  [Hybrid LB]"
echo ""
echo "  Endpoints (one per node):"
for i in "${!ALL_NODES[@]}"; do
    echo "    Node $i : http://${NODE_IPS[$i]}:${DECODE_PORT}"
done
echo ""
echo "  Model    : $MODEL"
echo "  EP       : $DECODE_EP  A2A=$A2A_BACKEND"
echo "  Logs     : ssh <node> 'tail -f $LOG_DIR/bench_*.log'"
echo ""
echo "  Run distributed benchmarks:"
echo "    python scripts/bench_distributed.py --nodes-file $NODES_FILE"
echo ""
echo "  Or single-node benchmark (one endpoint only):"
echo "    python scripts/bench.py --base-url http://${HEAD_IP}:${DECODE_PORT}"
echo ""
echo "  Stop:"
echo "    bash scripts/launch_bench_hybrid.sh --nodes-file $NODES_FILE --stop"
echo "================================================================"
