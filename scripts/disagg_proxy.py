#!/usr/bin/env python3
"""
Disaggregated prefill/decode proxy for vLLM 0.16+ with NixlConnector.

Request flow per inference call:
  1.  Proxy receives an OpenAI-compatible /v1/completions or
      /v1/chat/completions request from the client.
  2.  Proxy forwards it to the Prefill (P) instance with max_tokens=1.
      P runs the full prefill pass and returns a response whose body
      contains a `kv_transfer_params` field (do_remote_prefill=True,
      remote_block_ids, remote_engine_id, remote_host, remote_port …).
  3.  Proxy injects those params into the original request and sends it
      to the Decode (D) head node.  D's NixlConnector fetches the KV
      from P via NIXL/UCX and runs the full decode.
  4.  Proxy streams D's response back to the client.

Usage:
  python3 scripts/disagg_proxy.py \
      --prefill-host <P_NODE_IP> --prefill-port 8100 \
      --decode-host  <D_HEAD_IP> --decode-port  8200 \
      [--host 0.0.0.0] [--port 8000]
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

import aiohttp
from aiohttp import web

LOG = logging.getLogger("disagg_proxy")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    timeout: float = 300.0,
) -> dict[str, Any]:
    """POST JSON, raise on HTTP error, return parsed response body."""
    async with session.post(
        url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise web.HTTPBadGateway(
                reason=f"upstream {resp.status}: {text[:200]}"
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise web.HTTPBadGateway(reason=f"upstream JSON parse error: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Core P→D routing logic (shared by completions and chat endpoints)
# ─────────────────────────────────────────────────────────────────────────────

async def _route_pd(
    request: web.Request,
    p_url: str,
    d_url: str,
) -> web.Response:
    """Route a single inference request through P then D."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Bad JSON: {exc}")

    stream: bool = body.get("stream", False)

    async with aiohttp.ClientSession() as session:
        # ── Step 1: Prefill on P (max_tokens=1) ─────────────────────────────
        # kv_transfer_params={"do_remote_decode": true} tells the NixlConnector
        # on P to treat this as a disaggregated prefill request and include
        # remote_block_ids / remote_host / remote_port in the response.
        p_payload = {
            **body,
            "stream": False,
            "max_tokens": 1,
            "kv_transfer_params": {"do_remote_decode": True},
        }
        try:
            p_resp = await _post_json(session, p_url, p_payload)
        except web.HTTPException as exc:
            LOG.error("Prefill request to %s failed: %s", p_url, exc.reason)
            return web.Response(status=502, text=f"Prefill error: {exc.reason}")

        kv_params: dict[str, Any] | None = p_resp.get("kv_transfer_params")
        if kv_params:
            LOG.debug(
                "P→D KV transfer: %d block(s) from %s:%s",
                len(kv_params.get("remote_block_ids") or []),
                kv_params.get("remote_host"),
                kv_params.get("remote_port"),
            )
        else:
            LOG.warning(
                "P returned no kv_transfer_params — "
                "NixlConnector may not be configured on P, or model has no KV."
            )

        # ── Step 2: Decode on D ──────────────────────────────────────────────
        d_payload = {**body}
        if kv_params:
            # vLLM's OpenAI serving reads kv_transfer_params from the request
            # body extra_args and sets do_remote_prefill=True on the request.
            d_payload["kv_transfer_params"] = kv_params

        if stream:
            return await _stream_from_d(session, d_url, d_payload)
        else:
            try:
                d_resp = await _post_json(session, d_url, d_payload)
            except web.HTTPException as exc:
                LOG.error("Decode request to %s failed: %s", d_url, exc.reason)
                return web.Response(status=502, text=f"Decode error: {exc.reason}")
            return web.json_response(d_resp)


async def _stream_from_d(
    session: aiohttp.ClientSession,
    d_url: str,
    payload: dict[str, Any],
) -> web.Response:
    """Collect streaming SSE chunks from D and return them as a single response.

    True chunked forwarding requires aiohttp's StreamResponse to be prepared
    against the incoming request object, which isn't available here.  For the
    disagg use-case the latency added by buffering is dominated by the prefill
    round-trip, so this is acceptable for now.
    """
    chunks: list[bytes] = []
    try:
        async with session.post(
            d_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600.0),
        ) as d_resp:
            async for chunk in d_resp.content.iter_any():
                chunks.append(chunk)
    except Exception as exc:
        return web.Response(status=502, text=f"Decode stream error: {exc}")
    return web.Response(
        status=200,
        body=b"".join(chunks),
        content_type="text/event-stream",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_completions(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    p_url = f"http://{cfg.prefill_host}:{cfg.prefill_port}/v1/completions"
    d_url = f"http://{cfg.decode_host}:{cfg.decode_port}/v1/completions"
    return await _route_pd(request, p_url, d_url)


async def handle_chat_completions(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    p_url = f"http://{cfg.prefill_host}:{cfg.prefill_port}/v1/chat/completions"
    d_url = f"http://{cfg.decode_host}:{cfg.decode_port}/v1/chat/completions"
    return await _route_pd(request, p_url, d_url)


async def handle_models(request: web.Request) -> web.Response:
    """Proxy /v1/models to the decode side (it holds the canonical model list)."""
    cfg = request.app["cfg"]
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://{cfg.decode_host}:{cfg.decode_port}/v1/models",
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as resp:
            body = await resp.json()
    return web.json_response(body)


async def handle_health(request: web.Request) -> web.Response:
    """Simple proxy-level health check (does NOT probe P/D)."""
    return web.Response(text="OK")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def build_app(cfg: argparse.Namespace) -> web.Application:
    app = web.Application()
    app["cfg"] = cfg
    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disaggregated prefill/decode proxy (vLLM NixlConnector)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Proxy listen address")
    parser.add_argument("--port", type=int, default=8000, help="Proxy listen port")
    parser.add_argument(
        "--prefill-host", required=True,
        help="Hostname/IP of the Prefill (P) vLLM instance"
    )
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument(
        "--decode-host", required=True,
        help="Hostname/IP of the Decode (D) head vLLM instance"
    )
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        stream=sys.stdout,
    )
    LOG.info(
        "Proxy  P=%s:%d  D=%s:%d  listening=%s:%d",
        args.prefill_host, args.prefill_port,
        args.decode_host, args.decode_port,
        args.host, args.port,
    )

    app = build_app(args)
    web.run_app(app, host=args.host, port=args.port, access_log=LOG)


if __name__ == "__main__":
    main()
