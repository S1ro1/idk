#!/usr/bin/env python3
"""
Test approaches to fix: dist.barrier() → HPC-X UCX UCM hook conflict → NIXL_ERR_BACKEND

Root cause: dist.barrier() causes NCCL to load HPC-X UCX (/opt/hpcx/ucx/lib/libucs.so.0),
which installs UCM (memory event) hooks as a global singleton. When nixl's bundled UCX
subsequently tries to install its own UCM hooks, it gets "Unsupported operation", which
prevents RDMA memory domain setup, making rc/dc transports unavailable → NIXL_ERR_BACKEND.

Usage:
  python test_nixl_ucm.py <approach>
  Approaches: A B C D E

  A = Pre-warm nixl BEFORE dist.init_process_group (baseline - works, no NCCL yet)
  B = Pre-warm nixl AFTER dist.init_process_group but BEFORE dist.barrier()
  C = UCX_MEM_EVENTS=no (prevent HPC-X UCX from installing UCM hooks)
  D = NCCL_NET_PLUGIN=none (prevent HPC-X NCCL plugin from loading HPC-X UCX)
  E = Baseline: no pre-warm, dist.barrier() first (expected to FAIL)
"""

import os, sys, time

APPROACH = sys.argv[1] if len(sys.argv) > 1 else 'E'
PORT = 29550 + ord(APPROACH)

# Base UCX settings - restrict to mlx5_0 only
os.environ['UCX_RCACHE_MAX_UNRELEASED'] = '1024'
os.environ['UCX_NET_DEVICES'] = 'mlx5_0:1'
os.environ['UCX_TLS'] = 'rc,self,sm,cma,cuda_ipc,cuda_copy'
os.environ['UCX_LOG_LEVEL'] = 'warn'  # quiet unless something goes wrong

print(f"\n=== Approach {APPROACH} ===")
print(f"UCX_NET_DEVICES={os.environ['UCX_NET_DEVICES']}")
print(f"UCX_TLS={os.environ['UCX_TLS']}")

if APPROACH == 'C':
    os.environ['UCX_MEM_EVENTS'] = 'no'
    print("UCX_MEM_EVENTS=no (prevent UCM hook installation by any UCX instance)")
elif APPROACH == 'D':
    os.environ['NCCL_NET_PLUGIN'] = 'none'
    print("NCCL_NET_PLUGIN=none (prevent HPC-X NCCL plugin from loading)")


def try_nixl(label):
    import nixl._api as nixl_api
    cfg = nixl_api.nixl_agent_config(num_threads=1)
    try:
        t0 = time.time()
        agent = nixl_api.nixl_agent(f'test_{label}', cfg)
        dt = (time.time() - t0) * 1000
        print(f"  [PASS] nixl_agent('{label}') created in {dt:.1f}ms")
        del agent
        return True
    except Exception as e:
        print(f"  [FAIL] nixl_agent('{label}'): {e}")
        return False


def do_nccl_and_barrier():
    import torch
    import torch.distributed as dist
    print("  init_process_group(nccl)...")
    dist.init_process_group(
        backend='nccl',
        world_size=1, rank=0,
        init_method=f'tcp://127.0.0.1:{PORT}'
    )
    print("  dist.barrier()...")
    t0 = time.time()
    dist.barrier()
    dt = (time.time() - t0) * 1000
    print(f"  [ok] barrier done in {dt:.1f}ms")
    return dist


if APPROACH == 'A':
    # Pre-warm nixl BEFORE any NCCL (safest)
    print("\n[Step 1] Pre-warm nixl (before NCCL init)...")
    ok = try_nixl('pre_nccl')
    if not ok:
        print("Pre-warm failed, aborting")
        sys.exit(1)
    print("\n[Step 2] NCCL init + barrier...")
    d = do_nccl_and_barrier()
    print("\n[Step 3] nixl after barrier...")
    ok = try_nixl('post_barrier')
    d.destroy_process_group()

elif APPROACH == 'B':
    # Pre-warm nixl AFTER NCCL init but BEFORE barrier
    print("\n[Step 1] NCCL init (no barrier yet)...")
    import torch
    import torch.distributed as dist
    dist.init_process_group(
        backend='nccl', world_size=1, rank=0,
        init_method=f'tcp://127.0.0.1:{PORT}'
    )
    print("  [ok] NCCL initialized (no barrier yet)")
    print("\n[Step 2] Pre-warm nixl (after NCCL init, before barrier)...")
    ok = try_nixl('pre_barrier')
    if not ok:
        print("Pre-warm failed after NCCL init. NCCL init alone already corrupts UCM?")
        dist.destroy_process_group()
        sys.exit(1)
    print("\n[Step 3] dist.barrier()...")
    t0 = time.time()
    dist.barrier()
    dt = (time.time() - t0) * 1000
    print(f"  [ok] barrier done in {dt:.1f}ms")
    print("\n[Step 4] nixl after barrier...")
    ok = try_nixl('post_barrier')
    dist.destroy_process_group()

elif APPROACH == 'C':
    # UCX_MEM_EVENTS=no - prevents UCM hook installation
    print("\n[Step 1] NCCL init + barrier...")
    d = do_nccl_and_barrier()
    print("\n[Step 2] nixl after barrier (with UCX_MEM_EVENTS=no)...")
    ok = try_nixl('post_barrier')
    d.destroy_process_group()

elif APPROACH == 'D':
    # NCCL_NET_PLUGIN=none - prevents HPC-X NCCL plugin
    print("\n[Step 1] NCCL init + barrier...")
    d = do_nccl_and_barrier()
    print("\n[Step 2] nixl after barrier (with NCCL_NET_PLUGIN=none)...")
    ok = try_nixl('post_barrier')
    d.destroy_process_group()

elif APPROACH == 'E':
    # Baseline: no pre-warm, expected to FAIL
    print("\n[Step 1] NCCL init + barrier (no pre-warm)...")
    d = do_nccl_and_barrier()
    print("\n[Step 2] nixl after barrier (baseline - expected FAIL)...")
    ok = try_nixl('post_barrier')
    d.destroy_process_group()

print("\nDone.")
