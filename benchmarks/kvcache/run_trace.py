"""Replay an OpenAI-compatible JSONL trace and save latency samples."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, payload: dict, timeout: float) -> tuple[float, int, str]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read()
            return time.perf_counter() - started, response.status, ""
    except urllib.error.HTTPError as error:
        error.read()
        return time.perf_counter() - started, error.code, "http_error"
    except (OSError, TimeoutError) as error:
        return time.perf_counter() - started, 0, type(error).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="JSONL OpenAI request bodies")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--output", type=Path, default=Path("latency.jsonl"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    samples: list[dict] = []
    with args.trace.open(encoding="utf-8") as trace:
        payloads = [json.loads(line) for line in trace if line.strip()]
    for repeat in range(args.repeat):
        for index, payload in enumerate(payloads):
            elapsed, status, error = request(args.url, payload, args.timeout)
            sample = {
                "repeat": repeat,
                "index": index,
                "elapsed_seconds": elapsed,
                "status": status,
            }
            if error:
                sample["error"] = error
            samples.append(sample)
            print(json.dumps(sample, ensure_ascii=False), flush=True)
    with args.output.open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
