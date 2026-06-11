#!/usr/bin/env bash
# Pull per-seed prediction dirs from the configured GPU hosts into a local staging tree,
# then merge per seed across hosts into one analyzable grid dir per seed. Set the HOSTS_*
# variables to your own hosts (any user@host reachable by ssh). Idempotent: re-run any
# time to refresh with whatever units have finished so far.
#
# Layout produced:
#   $STAGE/raw/<host>/s<SEED>/preds__*.jsonl   (pulled, per host)
#   $STAGE/seeds/s<SEED>/{manifest.json,preds__*.jsonl}  (merged across hosts)
set -uo pipefail

ART="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="${STAGE:-$ART/data/results/multiseed}"
# Configure your own ssh-reachable hosts here (or via the environment); blank by default.
HOSTS_QWEN="${HOSTS_QWEN:-}"
HOSTS_TUCANO="${HOSTS_TUCANO:-}"
HOSTS_EXTRA="${HOSTS_EXTRA:-}"

mkdir -p "$STAGE/raw" "$STAGE/seeds"

pull() {  # host
  local host="$1"
  echo "[pull] $host"
  mkdir -p "$STAGE/raw/$host"
  rsync -az -e 'ssh -o ConnectTimeout=10' \
    "$host:~/cq_data/out/multiseed/" "$STAGE/raw/$host/" 2>/dev/null || \
    echo "  (nothing yet on $host)"
}

for h in $HOSTS_QWEN $HOSTS_TUCANO $HOSTS_EXTRA; do pull "$h"; done

# Discover all seeds seen on any host
seeds=$(find "$STAGE/raw" -maxdepth 2 -type d -name 's*' -printf '%f\n' 2>/dev/null \
        | sort -u)

# Synthesize a manifest.json for a host-seed dir that has preds but no manifest yet
# (the runner writes the per-seed manifest only after all its models finish, so an
# in-progress seed lacks one). We rebuild it from each preds file's _meta line and
# the shipped item strata so merge/aggregation can proceed on completed units.
synth_manifest() {  # dir
  python3 - "$1" "$ART/data/cultural_strata.jsonl" "$ART/data/control_generic.jsonl" <<'PY'
import sys, json, glob, os
d, cult_f, ctrl_f = sys.argv[1], sys.argv[2], sys.argv[3]
mp = os.path.join(d, "manifest.json")
# Always rebuild from the CURRENT preds on disk. A host dir can accrue preds from
# several model-disjoint runs into the same seed dir (e.g. Tucano then heavy-Qwen),
# and even the runner's own manifest lists only that run's models, so any cached
# manifest may undercount. The preds _meta lines are the source of truth.
runs, models, precs, seed, n_items = [], [], [], None, None
for pf in sorted(glob.glob(os.path.join(d, "preds__*.jsonl"))):
    with open(pf, encoding="utf-8") as f:
        first = f.readline()
    try:
        meta = json.loads(first)
    except Exception:
        continue
    if not meta.get("_meta"):
        continue
    seed = meta.get("seed", seed); n_items = meta.get("n_items", n_items)
    m, p = meta["model"], meta["precision"]
    if m not in models: models.append(m)
    if p not in precs: precs.append(p)
    runs.append({"model": m, "precision": p, "preds_file": os.path.basename(pf)})
if not runs:
    sys.exit(1)
# strata counts from shipped data (proverbs added at runtime are tagged in preds)
import collections
by_group = collections.Counter(); by_stratum = collections.Counter(); by_region = collections.Counter()
# derive strata from the first preds file's records (authoritative for this run)
with open(runs[0]["preds_file"] and os.path.join(d, runs[0]["preds_file"]), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("_meta"): continue
        by_group[r["group"]] += 1; by_stratum[r["stratum"]] += 1
        if r["group"] == "cultural": by_region[r["region"]] += 1
man = {"seed": seed, "n_items": n_items, "n_proverbs": None, "scoring": "perm_options_v2",
       "models": models, "precisions": precs,
       "reference_machine": {"gpu": "?", "cuda": "?", "torch": "?"},
       "runs": runs,
       "item_strata": {"by_group": dict(by_group), "by_stratum": dict(by_stratum),
                       "by_region": dict(by_region)}}
json.dump(man, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
}

for s in $seeds; do
  # collect every host's preds for this seed
  indirs=()
  for hostdir in "$STAGE/raw"/*/"$s"; do
    [ -d "$hostdir" ] && [ -n "$(ls "$hostdir"/preds__*.jsonl 2>/dev/null)" ] || continue
    synth_manifest "$hostdir"
    [ -f "$hostdir/manifest.json" ] && indirs+=("$hostdir")
  done
  [ ${#indirs[@]} -eq 0 ] && continue
  out="$STAGE/seeds/$s"
  python3 "$ART/scripts/merge_grid.py" --in "${indirs[@]}" --out "$out" \
    >/dev/null 2>&1 && \
    echo "[merge] $s <- ${#indirs[@]} host(s) -> $(ls "$out"/preds__*.jsonl 2>/dev/null | wc -l) preds files" || \
    echo "[merge] $s FAILED"
done

echo "[collect] staged at $STAGE/seeds"
