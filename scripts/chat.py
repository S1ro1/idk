#!/usr/bin/env python3
"""
Chat with the disagg proxy (or direct vLLM endpoint).

Usage:
  python scripts/chat.py                           # interactive chat via proxy
  python scripts/chat.py "What is 2+2?"            # one-shot
  python scripts/chat.py --completions "Once upon"  # raw completion mode
  python scripts/chat.py --host 10.20.0.18 --port 8200  # direct to decode
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTP {e.code}: {body[:500]}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def _get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def detect_model(base: str) -> str:
    """Fetch model name from /v1/models endpoint."""
    try:
        data = _get(f"{base}/v1/models")
        return data["data"][0]["id"]
    except Exception:
        return "zai-org/GLM-5-FP8"


def chat(base: str, model: str, messages: list[dict], max_tokens: int,
         temperature: float) -> str:
    data = _post(f"{base}/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    return data["choices"][0]["message"]["content"]


def complete(base: str, model: str, prompt: str, max_tokens: int,
             temperature: float) -> str:
    data = _post(f"{base}/v1/completions", {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    return data["choices"][0]["text"]


def main():
    parser = argparse.ArgumentParser(description="Chat with the disagg proxy")
    parser.add_argument("prompt", nargs="?",
                        help="Prompt text (omit for interactive mode)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected from server if omitted)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--completions", action="store_true",
                        help="Use raw /v1/completions instead of chat")
    parser.add_argument("--system", default=None,
                        help="System prompt for chat mode")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    model = args.model or detect_model(base)

    if args.prompt:
        # One-shot mode
        if args.completions:
            text = complete(base, model, args.prompt,
                            args.max_tokens, args.temperature)
            print(args.prompt + text)
        else:
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": args.prompt})
            text = chat(base, model, messages,
                        args.max_tokens, args.temperature)
            print(text)
    else:
        # Interactive mode
        print(f"Connected to {base}  model={model}  "
              f"max_tokens={args.max_tokens}  Ctrl-C to quit\n")
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt.strip():
                continue
            messages.append({"role": "user", "content": prompt})
            if args.completions:
                text = complete(base, model, prompt,
                                args.max_tokens, args.temperature)
                print(f"\n{prompt}{text}\n")
                messages = messages[:-1]  # completions are stateless
            else:
                text = chat(base, model, messages,
                            args.max_tokens, args.temperature)
                print(f"\nAssistant: {text}\n")
                messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
