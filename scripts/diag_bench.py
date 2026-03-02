#!/usr/bin/env python3
"""
Diagnostic benchmark — sends requests and tracks exactly which complete/hang.
Bypasses vllm bench serve to isolate the issue.
"""

import argparse
import asyncio
import json
import time
import aiohttp


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    req_id: int,
    stream: bool = True,
) -> dict:
    """Send one completion request, return timing info."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": stream,
    }

    t_start = time.monotonic()
    t_first_token = None
    output_tokens = 0
    status = "unknown"
    error = ""

    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                status = f"http_{resp.status}"
                error = (await resp.text())[:200]
            elif stream:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        if chunk.get("choices"):
                            text = chunk["choices"][0].get("text", "")
                            if text:
                                output_tokens += 1
                    except json.JSONDecodeError:
                        pass
                status = "ok"
            else:
                body = await resp.json()
                t_first_token = time.monotonic()
                if body.get("choices"):
                    usage = body.get("usage", {})
                    output_tokens = usage.get("completion_tokens", 0)
                status = "ok"
    except asyncio.TimeoutError:
        status = "timeout"
    except Exception as e:
        status = "error"
        error = str(e)[:200]

    t_end = time.monotonic()
    return {
        "req_id": req_id,
        "status": status,
        "output_tokens": output_tokens,
        "ttft_ms": (t_first_token - t_start) * 1000 if t_first_token else None,
        "total_ms": (t_end - t_start) * 1000,
        "error": error,
    }


async def run_bench(
    url: str,
    model: str,
    num_prompts: int,
    max_concurrency: int,
    input_tokens: int,
    output_tokens: int,
    stream: bool,
):
    # Use a simple repeated token as prompt
    prompt = "hello " * max(input_tokens // 2, 1)

    sem = asyncio.Semaphore(max_concurrency)
    results = []

    async def bounded_request(session, req_id):
        async with sem:
            r = await send_request(session, url, model, prompt, output_tokens, req_id, stream)
            results.append(r)
            # Live progress
            ok = sum(1 for x in results if x["status"] == "ok")
            fail = sum(1 for x in results if x["status"] != "ok" and x["status"] != "unknown")
            print(
                f"\r  [{len(results):>4}/{num_prompts}]  ok={ok}  timeout={sum(1 for x in results if x['status']=='timeout')}  error={sum(1 for x in results if x['status'] not in ('ok','timeout','unknown'))}",
                end="",
                flush=True,
            )
            return r

    print(f"Sending {num_prompts} requests (max_concurrency={max_concurrency}, stream={stream})")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Input ~{input_tokens} tokens, Output {output_tokens} tokens")
    print()

    connector = aiohttp.TCPConnector(limit=max_concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.monotonic()
        tasks = [bounded_request(session, i) for i in range(num_prompts)]
        await asyncio.gather(*tasks)
        wall = time.monotonic() - t0

    print()
    print()

    # Analyze
    ok = [r for r in results if r["status"] == "ok"]
    timeout = [r for r in results if r["status"] == "timeout"]
    errors = [r for r in results if r["status"] not in ("ok", "timeout")]

    print(f"=== Results ({wall:.1f}s wall time) ===")
    print(f"  OK:      {len(ok)}")
    print(f"  Timeout: {len(timeout)}")
    print(f"  Errors:  {len(errors)}")
    print(f"  Total:   {len(results)}")
    print(f"  OK ratio: {len(ok)/len(results)*100:.1f}%")
    print()

    if ok:
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
        totals = [r["total_ms"] for r in ok]
        out_toks = [r["output_tokens"] for r in ok]
        total_out = sum(out_toks)
        throughput = total_out / wall if wall > 0 else 0

        ttfts.sort()
        totals.sort()

        def pct(arr, p):
            idx = min(int(len(arr) * p / 100), len(arr) - 1)
            return arr[idx]

        print(f"  Output throughput: {throughput:.0f} tok/s")
        print(f"  Request throughput: {len(ok)/wall:.1f} req/s")
        print(f"  Avg output tokens: {total_out/len(ok):.0f}")
        print()
        if ttfts:
            print(f"  TTFT   p50={pct(ttfts,50):.0f}ms  p90={pct(ttfts,90):.0f}ms  p99={pct(ttfts,99):.0f}ms")
        print(f"  E2E    p50={pct(totals,50):.0f}ms  p90={pct(totals,90):.0f}ms  p99={pct(totals,99):.0f}ms")

    if errors:
        print()
        print("  First 5 errors:")
        for r in errors[:5]:
            print(f"    req {r['req_id']}: {r['status']}  {r['error'][:100]}")

    if timeout:
        print()
        print(f"  Timeout requests: {[r['req_id'] for r in timeout[:20]]}{'...' if len(timeout) > 20 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic vLLM benchmark")
    parser.add_argument("--url", default="http://10.20.0.18:8200", help="Server URL")
    parser.add_argument("--model", default="zai-org/GLM-5-FP8")
    parser.add_argument("-n", "--num-prompts", type=int, default=64)
    parser.add_argument("-c", "--max-concurrency", type=int, default=64)
    parser.add_argument("--input-tokens", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run_bench(
            args.url,
            args.model,
            args.num_prompts,
            args.max_concurrency,
            args.input_tokens,
            args.output_tokens,
            stream=not args.no_stream,
        )
    )


if __name__ == "__main__":
    main()
