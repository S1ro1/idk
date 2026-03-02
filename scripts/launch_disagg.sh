#!/usr/bin/env bash
# =============================================================================
# launch_disagg.sh — Disaggregated prefill/decode serving on vLLM 0.16+
# =============================================================================
#
# Architecture
# ─────────────
#   • N Prefill (P) nodes  : EP=N*8, TP=1, deepep_high_throughput A2A
#   • M Decode  (D) nodes  : EP=M*8, TP=1, deepep_low_latency A2A (prod)
#   • KV transfer          : NixlConnector over UCX / InfiniBand
#   • Proxy                : scripts/disagg_proxy.py on port 8000
#
# Modes
# ─────
#   dev   2 nodes (1P + 1D), samsja/mini-glm-moe (cached), EP=8 both sides.
#   prod  6 nodes (2P + 4D), zai-org/GLM-5-FP8, EP=16 prefill / EP=32 decode.
#
# Usage
# ─────
#   bash scripts/launch_disagg.sh [OPTIONS]
#
# Options:
#   --mode  dev|prod          (default: dev)
#   --stop                    Kill all remote vllm processes and exit
#   --nodes-file FILE         Node list (default: valid_nodes.txt)
#   --model MODEL_ID          Override model for this run
#   --max-model-len N         Override max context length
#   --proxy-only              (Re-)start the proxy only, leave servers alone
#   --no-proxy                Start servers only, skip the proxy
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── defaults ──────────────────────────────────────────────────────────────────
MODE="dev"
DO_STOP=0
PROXY_ONLY=0
NO_PROXY=0
NODES_FILE="$REPO_ROOT/valid_nodes.txt"
MODEL_OVERRIDE=""
MAX_MODEL_LEN_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)          MODE="$2";              shift 2 ;;
        --stop)          DO_STOP=1;              shift   ;;
        --nodes-file)    NODES_FILE="$2";        shift 2 ;;
        --model)         MODEL_OVERRIDE="$2";    shift 2 ;;
        --max-model-len) MAX_MODEL_LEN_OVERRIDE="$2"; shift 2 ;;
        --proxy-only)    PROXY_ONLY=1;           shift   ;;
        --no-proxy)      NO_PROXY=1;             shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── mode config ───────────────────────────────────────────────────────────────
case "$MODE" in
    dev)
        MODEL="${MODEL_OVERRIDE:-samsja/mini-glm-moe}"
        NUM_PREFILL_NODES=1
        NUM_DECODE_NODES=1
        PREFILL_EP=8
        PREFILL_DP_LOCAL=8
        DECODE_EP=8
        DECODE_DP_LOCAL=8
        MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-4096}"
        GPU_MEM_UTIL=0.80
        HF_HUB_OFFLINE_VAL=1      # mini-glm-moe is cached
        HF_HOME_VAL="/shared/huggingface"
        PREFILL_A2A="deepep_high_throughput"
        DECODE_A2A="deepep_high_throughput"
        ;;
    prod)
        MODEL="${MODEL_OVERRIDE:-zai-org/GLM-5-FP8}"
        NUM_PREFILL_NODES=2
        NUM_DECODE_NODES=4
        PREFILL_EP=16
        PREFILL_DP_LOCAL=8
        DECODE_EP=32
        DECODE_DP_LOCAL=8
        MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-32768}"
        GPU_MEM_UTIL=0.95
        HF_HUB_OFFLINE_VAL=1
        HF_HOME_VAL="/shared/huggingface"
        PREFILL_A2A="deepep_high_throughput"
        DECODE_A2A="deepep_low_latency"
        ;;
    *)
        echo "Unknown mode: $MODE  (use dev|prod)" >&2; exit 1 ;;
esac

TOTAL_NODES=$((NUM_PREFILL_NODES + NUM_DECODE_NODES))

# ── ports ─────────────────────────────────────────────────────────────────────
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8000
NIXL_PORT=5600
DP_RPC_PORT=29550

# ── paths ─────────────────────────────────────────────────────────────────────
VLLM_BIN="$REPO_ROOT/.venv/bin/vllm"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
NVSHMEM_LIB_DIR="$REPO_ROOT/ep_kernels_workspace/nvshmem/lib"
LOG_DIR="/tmp/vllm_disagg_logs"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

# ── KV transfer config ────────────────────────────────────────────────────────
# Both P and D use kv_role=kv_both with NixlConnector.
# P listens on its own NIXL side-channel and includes remote_host/port in
# every response; D connects to that address to fetch KV via UCX/RDMA.
KV_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_threads":1}}'

# =============================================================================
# Helpers
# =============================================================================

read_nodes() {
    grep -v '^\s*$' "$NODES_FILE" | awk '{print $1}' | head -n "$TOTAL_NODES"
}

get_node_ip() {
    ssh $SSH_OPTS -n "$1" \
        "hostname -I 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo "$1"
}

wait_for_server() {
    local host="$1" port="$2" label="${3:-$1:$2}" timeout="${4:-600}"
    local elapsed=0
    echo -n "    Waiting for $label ."
    until curl -sf "http://$host:$port/health" >/dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed+5)); echo -n "."
        if (( elapsed >= timeout )); then
            echo " TIMEOUT (${timeout}s) — check $LOG_DIR/"
            return 1
        fi
    done
    echo " ready (${elapsed}s)"
}

# =============================================================================
# --stop
# =============================================================================
if (( DO_STOP )); then
    echo "=== Stopping all remote vllm processes ==="
    # Read ALL nodes from the file (not just TOTAL_NODES) so --stop works
    # without needing --mode to match the running deployment.
    mapfile -t nodes < <(grep -v '^\s*$' "$NODES_FILE" | awk '{print $1}')
    for node in "${nodes[@]}"; do
        echo -n "  [$node] "
        # Use pgrep + kill instead of pkill -f to avoid killing the SSH
        # session (pkill -f matches the bash process running the command).
        ssh $SSH_OPTS -n "$node" \
            'pids=$(ps aux | grep -E "VLLM::|vllm serve|vllm.entrypoints|multiprocessing.resource_tracker" | grep -v grep | awk "{print \$2}")
             if [ -n "$pids" ]; then echo "$pids" | xargs kill -9 2>/dev/null; fi
             echo done' \
            2>/dev/null || echo "(unreachable)"
    done
    pkill -f "disagg_proxy.py" 2>/dev/null && echo "  [local] proxy stopped" || true
    echo "Done."
    exit 0
fi

# =============================================================================
# Node selection
# =============================================================================
mapfile -t ALL_NODES < <(read_nodes)
if (( ${#ALL_NODES[@]} < TOTAL_NODES )); then
    echo "ERROR: need $TOTAL_NODES nodes, only ${#ALL_NODES[@]} in $NODES_FILE" >&2
    exit 1
fi
PREFILL_NODES=("${ALL_NODES[@]:0:$NUM_PREFILL_NODES}")
PREFILL_HEAD="${PREFILL_NODES[0]}"
DECODE_NODES=("${ALL_NODES[@]:$NUM_PREFILL_NODES:$NUM_DECODE_NODES}")
DECODE_HEAD="${DECODE_NODES[0]}"

echo "================================================================"
echo " Disaggregated prefill/decode  [mode=$MODE]"
echo " Model          : $MODEL"
echo " Prefill (P)    : ${PREFILL_NODES[*]}  EP=$PREFILL_EP  A2A=$PREFILL_A2A"
echo " Decode  (D)    : ${DECODE_NODES[*]}"
echo "                  EP=$DECODE_EP  A2A=$DECODE_A2A"
echo " max_model_len  : $MAX_MODEL_LEN"
echo "================================================================"

echo ""
echo "--- Resolving node IPs ---"
PREFILL_HEAD_IP="$(get_node_ip "$PREFILL_HEAD")"
DECODE_HEAD_IP="$(get_node_ip "$DECODE_HEAD")"
echo "  Prefill head IP : $PREFILL_HEAD_IP"
echo "  Decode  head IP : $DECODE_HEAD_IP"

mkdir -p "$LOG_DIR"

# =============================================================================
# Launch helpers
# =============================================================================

# Run a script on a remote node via SSH stdin (avoids all quoting issues with
# embedded JSON, single quotes, etc.).  The script is a here-doc so local
# variables ($MODEL, $PREFILL_EP …) are expanded here; remote-side variables
# (\$PREFILL_IP, \$!) are escaped and evaluated on the remote shell.
#
# Usage: launch_remote <node> <log_file> <<'SCRIPT'
#           ... bash script ...
#        SCRIPT
#
# Returns the remote PID via stdout.
launch_remote() {
    # Reads the script body from stdin, SCP-less uploads it to the remote node
    # (via SSH stdin + cat), then executes it detached under nohup.
    # Prints only the remote PID to stdout; all messages go to stderr so callers
    # can capture the PID cleanly:
    #   PID=$(launch_remote node logfile <<'SCRIPT' ... SCRIPT)
    local node="$1"
    local log_file="$2"
    local script_body
    script_body="$(cat)"

    echo "" >&2
    echo "--- Launching on $node  (log: $log_file) ---" >&2

    # Step 1: Upload script via SSH stdin (no -n so stdin is available).
    local remote_script="/tmp/vllm_launch_$(date +%s%N).sh"
    # shellcheck disable=SC2029
    ssh $SSH_OPTS "$node" \
        "mkdir -p '${LOG_DIR}'; cat > '${remote_script}'; chmod +x '${remote_script}'" \
        <<< "$script_body"

    # Step 2: Execute the remote script with nohup (-n safe here: no stdin needed).
    # shellcheck disable=SC2029
    ssh $SSH_OPTS -n "$node" \
        "nohup '${remote_script}' > '${log_file}' 2>&1 & echo \$!"
}

# =============================================================================
# PREFILL INSTANCE
# =============================================================================
if (( ! PROXY_ONLY )); then
    echo ""
    echo "=== Starting Prefill (P) instances (${#PREFILL_NODES[@]} node(s)) ==="

    # Prefill head — serves HTTP and coordinates all prefill DP ranks
    PREFILL_LOG="$LOG_DIR/vllm_prefill.log"
    PREFILL_HEAD_PID=$(launch_remote "$PREFILL_HEAD" "$PREFILL_LOG" <<PREFILL_SCRIPT
export HF_HOME='${HF_HOME_VAL}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE_VAL}'
export LD_LIBRARY_PATH='${NVSHMEM_LIB_DIR}':\${LD_LIBRARY_PATH:-}
ulimit -n 65536
export UCX_RCACHE_MAX_UNRELEASED=1024
# Restrict nixl's UCX to mlx5_0 only (other ports not connected/working).
export UCX_NET_DEVICES='mlx5_0:1'
export UCX_TLS='rc,dc,self,sm,cma,cuda_ipc,cuda_copy'
# NCCL_NET_PLUGIN=none prevents NCCL from loading the HPC-X UCX plugin
# (libnccl-net.so), which would install UCM hooks and prevent nixl's bundled
# UCX from installing its own UCM hooks → NIXL_ERR_BACKEND.
# NCCL still uses its built-in IB verbs transport for collectives.
export NCCL_NET_PLUGIN=none
# Force torch.distributed gloo (DP group discovery) to use Ethernet, not IB.
# Without this, gloo hangs or fails on IB-attached nodes.
export GLOO_SOCKET_IFNAME=bond0
# Use local /tmp for DeepGEMM JIT cache to avoid NFS stale-file-handle races
# when many DP ranks compile simultaneously across nodes.  Seed it from the
# NFS cache so JIT warmup reuses previously compiled kernels.
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache
rsync -a --ignore-existing /home/matej/.cache/vllm/deep_gemm/ /tmp/deep_gemm_cache/ 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
PREFILL_IP=\$(hostname -I | awk '{print \$1}')
export VLLM_NIXL_SIDE_CHANNEL_HOST=\$PREFILL_IP
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PORT}
exec '${VLLM_BIN}' serve '${MODEL}' \
    --host 0.0.0.0 \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size 1 \
    --data-parallel-size ${PREFILL_EP} \
    --data-parallel-size-local ${PREFILL_DP_LOCAL} \
    --data-parallel-address \$PREFILL_IP \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --enable-expert-parallel \
    --all2all-backend ${PREFILL_A2A} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --disable-log-requests \
    --kv-transfer-config '${KV_CFG}'
PREFILL_SCRIPT
    )
    echo "  Prefill head PID: $PREFILL_HEAD_PID"

    # Prefill workers (nodes 1..NUM_PREFILL_NODES-1) — headless, connect to prefill head
    for i in "${!PREFILL_NODES[@]}"; do
        (( i == 0 )) && continue
        P_WORKER_NODE="${PREFILL_NODES[$i]}"
        P_START_RANK=$(( i * PREFILL_DP_LOCAL ))
        P_WORKER_LOG="$LOG_DIR/vllm_prefill_worker_${i}.log"

        P_WORKER_PID=$(launch_remote "$P_WORKER_NODE" "$P_WORKER_LOG" <<PWORKER_SCRIPT
export HF_HOME='${HF_HOME_VAL}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE_VAL}'
export LD_LIBRARY_PATH='${NVSHMEM_LIB_DIR}':\${LD_LIBRARY_PATH:-}
ulimit -n 65536
export UCX_RCACHE_MAX_UNRELEASED=1024
export UCX_NET_DEVICES='mlx5_0:1'
export UCX_TLS='rc,dc,self,sm,cma,cuda_ipc,cuda_copy'
export NCCL_NET_PLUGIN=none
# Force torch.distributed gloo (DP group discovery) to use Ethernet, not IB.
# Without this, gloo hangs or fails on IB-attached nodes.
export GLOO_SOCKET_IFNAME=bond0
# Use local /tmp for DeepGEMM JIT cache to avoid NFS stale-file-handle races
# when many DP ranks compile simultaneously across nodes.  Seed it from the
# NFS cache so JIT warmup reuses previously compiled kernels.
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache
rsync -a --ignore-existing /home/matej/.cache/vllm/deep_gemm/ /tmp/deep_gemm_cache/ 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
P_IP=\$(hostname -I | awk '{print \$1}')
export VLLM_NIXL_SIDE_CHANNEL_HOST=\$P_IP
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PORT}
exec '${VLLM_BIN}' serve '${MODEL}' \
    --headless \
    --tensor-parallel-size 1 \
    --data-parallel-size ${PREFILL_EP} \
    --data-parallel-size-local ${PREFILL_DP_LOCAL} \
    --data-parallel-start-rank ${P_START_RANK} \
    --data-parallel-address ${PREFILL_HEAD_IP} \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --enable-expert-parallel \
    --all2all-backend ${PREFILL_A2A} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --disable-log-requests \
    --kv-transfer-config '${KV_CFG}'
PWORKER_SCRIPT
        )
        echo "  Prefill worker $i PID: $P_WORKER_PID"
    done

    # ==========================================================================
    # DECODE INSTANCES
    # ==========================================================================
    echo ""
    echo "=== Starting Decode (D) instances ==="

    # Decode head — serves HTTP, first node in DECODE_NODES
    DECODE_HEAD_LOG="$LOG_DIR/vllm_decode_head.log"
    DECODE_HEAD_PID=$(launch_remote "$DECODE_HEAD" "$DECODE_HEAD_LOG" <<DECODE_HEAD_SCRIPT
export HF_HOME='${HF_HOME_VAL}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE_VAL}'
export LD_LIBRARY_PATH='${NVSHMEM_LIB_DIR}':\${LD_LIBRARY_PATH:-}
ulimit -n 65536
export UCX_RCACHE_MAX_UNRELEASED=1024
export UCX_NET_DEVICES='mlx5_0:1'
export UCX_TLS='rc,dc,self,sm,cma,cuda_ipc,cuda_copy'
export NCCL_NET_PLUGIN=none
# Force torch.distributed gloo (DP group discovery) to use Ethernet, not IB.
# Without this, gloo hangs or fails on IB-attached nodes.
export GLOO_SOCKET_IFNAME=bond0
# Use local /tmp for DeepGEMM JIT cache to avoid NFS stale-file-handle races
# when many DP ranks compile simultaneously across nodes.  Seed it from the
# NFS cache so JIT warmup reuses previously compiled kernels.
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache
rsync -a --ignore-existing /home/matej/.cache/vllm/deep_gemm/ /tmp/deep_gemm_cache/ 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
D_IP=\$(hostname -I | awk '{print \$1}')
export VLLM_NIXL_SIDE_CHANNEL_HOST=\$D_IP
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PORT}
exec '${VLLM_BIN}' serve '${MODEL}' \
    --host 0.0.0.0 \
    --port ${DECODE_PORT} \
    --tensor-parallel-size 1 \
    --data-parallel-size ${DECODE_EP} \
    --data-parallel-size-local ${DECODE_DP_LOCAL} \
    --data-parallel-address \$D_IP \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --enable-expert-parallel \
    --all2all-backend ${DECODE_A2A} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --disable-log-requests \
    --kv-transfer-config '${KV_CFG}'
DECODE_HEAD_SCRIPT
    )
    echo "  PID: $DECODE_HEAD_PID"

    # Decode workers (nodes 1..N-1) — headless, connect to decode head
    for i in "${!DECODE_NODES[@]}"; do
        (( i == 0 )) && continue
        WORKER_NODE="${DECODE_NODES[$i]}"
        START_RANK=$(( i * DECODE_DP_LOCAL ))
        WORKER_LOG="$LOG_DIR/vllm_decode_worker_${i}.log"

        WORKER_PID=$(launch_remote "$WORKER_NODE" "$WORKER_LOG" <<WORKER_SCRIPT
export HF_HOME='${HF_HOME_VAL}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE_VAL}'
export LD_LIBRARY_PATH='${NVSHMEM_LIB_DIR}':\${LD_LIBRARY_PATH:-}
ulimit -n 65536
export UCX_RCACHE_MAX_UNRELEASED=1024
export UCX_NET_DEVICES='mlx5_0:1'
export UCX_TLS='rc,dc,self,sm,cma,cuda_ipc,cuda_copy'
export NCCL_NET_PLUGIN=none
# Force torch.distributed gloo (DP group discovery) to use Ethernet, not IB.
# Without this, gloo hangs or fails on IB-attached nodes.
export GLOO_SOCKET_IFNAME=bond0
# Use local /tmp for DeepGEMM JIT cache to avoid NFS stale-file-handle races
# when many DP ranks compile simultaneously across nodes.  Seed it from the
# NFS cache so JIT warmup reuses previously compiled kernels.
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache
rsync -a --ignore-existing /home/matej/.cache/vllm/deep_gemm/ /tmp/deep_gemm_cache/ 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false
D_IP=\$(hostname -I | awk '{print \$1}')
export VLLM_NIXL_SIDE_CHANNEL_HOST=\$D_IP
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PORT}
exec '${VLLM_BIN}' serve '${MODEL}' \
    --headless \
    --tensor-parallel-size 1 \
    --data-parallel-size ${DECODE_EP} \
    --data-parallel-size-local ${DECODE_DP_LOCAL} \
    --data-parallel-start-rank ${START_RANK} \
    --data-parallel-address ${DECODE_HEAD_IP} \
    --data-parallel-rpc-port ${DP_RPC_PORT} \
    --enable-expert-parallel \
    --all2all-backend ${DECODE_A2A} \
    --trust-remote-code \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --disable-log-requests \
    --kv-transfer-config '${KV_CFG}'
WORKER_SCRIPT
        )
        echo "  Worker $i PID: $WORKER_PID"
    done

    # ── Wait for servers ──────────────────────────────────────────────────────
    echo ""
    echo "=== Waiting for servers ==="
    wait_for_server "$PREFILL_HEAD_IP" "$PREFILL_PORT" "Prefill($PREFILL_HEAD)" 600
    wait_for_server "$DECODE_HEAD_IP"  "$DECODE_PORT"  "Decode($DECODE_HEAD)"   600
fi

# =============================================================================
# PROXY
# =============================================================================
if (( ! NO_PROXY )); then
    echo ""
    echo "=== Starting proxy (local, port $PROXY_PORT) ==="
    pkill -f "disagg_proxy.py" 2>/dev/null && sleep 1 || true

    PROXY_LOG="$LOG_DIR/disagg_proxy.log"
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/disagg_proxy.py" \
        --host 0.0.0.0 \
        --port "$PROXY_PORT" \
        --prefill-host "$PREFILL_HEAD_IP" \
        --prefill-port "$PREFILL_PORT" \
        --decode-host  "$DECODE_HEAD_IP" \
        --decode-port  "$DECODE_PORT" \
        > "$PROXY_LOG" 2>&1 &
    echo "  Proxy PID=$!  log=$PROXY_LOG"
    sleep 2
fi

# =============================================================================
# Smoke test
# =============================================================================
echo ""
echo "=== Smoke test ==="

if (( NO_PROXY )); then
    TARGET="http://${PREFILL_HEAD_IP}:${PREFILL_PORT}"
else
    TARGET="http://localhost:${PROXY_PORT}"
    # Wait for proxy health
    for _ in $(seq 1 15); do
        curl -sf "${TARGET}/health" >/dev/null 2>&1 && break || sleep 2
    done
fi

echo "  POST ${TARGET}/v1/completions"
RESP=$(curl -sf -X POST "${TARGET}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 8,
        \"temperature\": 0
    }" 2>&1) || RESP="(curl failed)"

if echo "$RESP" | grep -q '"text"'; then
    TEXT=$("$PYTHON_BIN" -c \
        "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())" \
        <<< "$RESP" 2>/dev/null || echo "$RESP")
    echo "  SUCCESS → \"$TEXT\""
    KV_PRESENT=$("$PYTHON_BIN" -c \
        "import sys,json; d=json.load(sys.stdin); print('kv_transfer_params' in str(d))" \
        <<< "$RESP" 2>/dev/null || echo "unknown")
    echo "  kv_transfer_params present: $KV_PRESENT"
else
    echo "  WARNING: unexpected response — $RESP"
    echo "  Logs: $LOG_DIR/"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo " Disaggregated serving is UP"
echo ""
echo "  Proxy  (OpenAI API) : http://localhost:${PROXY_PORT}"
echo "  Prefill head        : http://${PREFILL_HEAD_IP}:${PREFILL_PORT}  [$PREFILL_HEAD]"
echo "  Decode head         : http://${DECODE_HEAD_IP}:${DECODE_PORT} [$DECODE_HEAD]"
echo "  Logs                : $LOG_DIR/"
echo ""
echo "  Model  : $MODEL"
echo "  Mode   : $MODE  (P ${#PREFILL_NODES[@]}×node EP=$PREFILL_EP / D ${#DECODE_NODES[@]}×node EP=$DECODE_EP)"
echo ""
echo "  To stop: bash scripts/launch_disagg.sh --stop"
echo "================================================================"
