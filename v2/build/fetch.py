"""Network fetch layer for CulturaQuant-v2 build.

When the local environment blocks direct outbound HTTP, set CQ_FETCH_SSH_HOST to a
network-capable host and requests are tunneled through `ssh <host> curl ...`; otherwise
curl runs locally. Every raw response is cached to v2/cache/ keyed by a hash of the
request, so the build is fully reproducible and resumable: a cache hit never touches the
network. Throttling (sleep) is applied only on cache misses.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
import urllib.parse
from pathlib import Path

SSH_HOST = os.environ.get("CQ_FETCH_SSH_HOST", "")
UA = "CulturaQuant/1.0 (research)"
WDQS = "https://query.wikidata.org/sparql"
PV_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"

ROOT = Path(__file__).resolve().parent.parent
CACHE_SPARQL = ROOT / "cache" / "sparql"
CACHE_PV = ROOT / "cache" / "pageviews"
for _d in (CACHE_SPARQL, CACHE_PV):
    _d.mkdir(parents=True, exist_ok=True)

# Throttle between live network calls (seconds). Cache hits skip this entirely.
SPARQL_SLEEP = 2.0
PV_SLEEP = 0.3


class FetchError(RuntimeError):
    pass


def _remote_cmd(args: list[str]) -> list[str]:
    """Build the curl command, tunneled through ssh only when CQ_FETCH_SSH_HOST is set.

    With no host, curl runs locally. With a host, ssh concatenates its trailing args with
    spaces and the remote shell re-parses them, so each curl token must be shell-quoted.
    """
    if not SSH_HOST:
        return ["curl", *args]
    remote = "curl " + " ".join(shlex.quote(a) for a in args)
    return ["ssh", "-o", "ConnectTimeout=20", SSH_HOST, remote]


def _ssh_curl(args: list[str], timeout: int = 180) -> str:
    """Run `curl <args>` locally, or on a network-capable host over ssh; return stdout."""
    cmd = _remote_cmd(args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise FetchError(f"ssh curl failed (rc={proc.returncode}): {proc.stderr[:500]}")
    if not proc.stdout.strip():
        raise FetchError(f"empty response; stderr={proc.stderr[:300]}")
    return proc.stdout


def sparql(query: str, *, tag: str = "q", retries: int = 4) -> dict:
    """Run a SPARQL query against WDQS, cached by query hash. Returns parsed JSON.

    Retries with exponential backoff on transient failures (WDQS "upstream request
    timeout", empty body, ssh hiccups). Only successful JSON is cached, so a transient
    error never poisons the cache.
    """
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:20]
    cache_file = CACHE_SPARQL / f"{tag}__{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    args = [
        "-s",
        "--max-time",
        "150",
        "-G",
        WDQS,
        "--data-urlencode",
        "query@-",
        "-H",
        "Accept: application/sparql-results+json",
        "-A",
        UA,
    ]
    cmd = _remote_cmd(args)
    last_err = ""
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                cmd, input=query, capture_output=True, text=True, timeout=200
            )
        except subprocess.TimeoutExpired:
            last_err = "local timeout"
            time.sleep(5 * (attempt + 1))
            continue
        out = proc.stdout.strip()
        if proc.returncode == 0 and out.startswith("{"):
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last_err = f"non-JSON: {out[:120]}"
            else:
                cache_file.write_text(json.dumps(data, ensure_ascii=False))
                time.sleep(SPARQL_SLEEP)
                return data
        else:
            last_err = f"rc={proc.returncode} body={out[:120]!r} err={proc.stderr[:120]}"
        # Transient (timeout / 429 / 5xx): back off and retry.
        time.sleep(5 * (attempt + 1))
    raise FetchError(f"SPARQL failed after {retries} tries [{tag}]: {last_err}")


def pageviews(title: str, start: str = "2024010100", end: str = "2024123100") -> dict | None:
    """Fetch pt.wikipedia monthly pageviews for an article title. Cached.

    Returns the parsed JSON, or None if the article has no pageview data (404).
    """
    enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
    key = hashlib.sha256(f"{title}|{start}|{end}".encode()).hexdigest()[:20]
    cache_file = CACHE_PV / f"{key}.json"
    if cache_file.exists():
        txt = cache_file.read_text()
        return json.loads(txt) if txt.strip() else None

    url = f"{PV_API}/pt.wikipedia/all-access/user/{enc}/monthly/{start}/{end}"
    args = ["-s", "--max-time", "30", url, "-A", UA]
    try:
        out = _ssh_curl(args, timeout=60)
    except FetchError:
        cache_file.write_text("")  # negative cache
        time.sleep(PV_SLEEP)
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        cache_file.write_text("")
        time.sleep(PV_SLEEP)
        return None
    if "items" not in data:
        # 404 / not-found payload; negative-cache it.
        cache_file.write_text("")
        time.sleep(PV_SLEEP)
        return None
    cache_file.write_text(json.dumps(data, ensure_ascii=False))
    time.sleep(PV_SLEEP)
    return data
