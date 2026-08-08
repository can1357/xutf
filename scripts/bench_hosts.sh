#!/usr/bin/env bash
# Runs the benches on any mix of the local machine and remote hosts, collects
# the raw JSONL under benches/data/, and re-renders the README benchmark
# section from everything committed there.
#
# Usage:
#   scripts/bench_hosts.sh [--local "Label"] [ssh-host[:"Remote Label"]...]
# Examples:
#   scripts/bench_hosts.sh --local "Apple M4 Max" xeon.internal:"Xeon x86-64"
#   scripts/bench_hosts.sh xeon.internal:"Xeon x86-64"   # refresh xeon only
#
# Remote hosts need rustup + the repo toolchain; sources are rsynced to
# /tmp/xutf-bench and results copied back, nothing else is left behind.
set -eu

cd "$(dirname "$0")/.."
BENCHES=(--bench graphemes --bench width --bench truncate --bench wrap --bench throughput)
RUSTFLAGS="-C target-cpu=native"
export RUSTFLAGS

slug() {
	printf '%s' "$1" | tr '[:upper:] ' '[:lower:]-'
}

mkdir -p benches/data

local_label=""
if [ "${1:-}" = "--local" ]; then
	local_label="${2:?--local needs a label}"
	shift 2
fi
[ -n "$local_label" ] || [ $# -gt 0 ] ||
	{ echo "usage: bench_hosts.sh [--local \"Label\"] [host[:\"Label\"]...]" >&2; exit 2; }

if [ -n "$local_label" ]; then
	out="benches/data/$(slug "$local_label").jsonl"
	rm -f "$out"
	echo "== $local_label (local) -> $out"
	BENCH_JSON="$out" BENCH_HOST="$local_label" cargo bench -q "${BENCHES[@]}"
fi

for spec in "$@"; do
	host="${spec%%:*}"
	label="${spec#*:}"
	[ "$label" = "$spec" ] && label="$host"
	out="benches/data/$(slug "$label").jsonl"
	echo "== $label ($host) -> $out"
	rsync -a --delete --exclude target --exclude scripts/ucd --exclude .git \
		--exclude benches/data ./ "$host":/tmp/xutf-bench/
	printf '%s\0%s\0' "$RUSTFLAGS" "$label" |
		ssh "$host" 'bash -c '"'"'
IFS= read -r -d "" rustflags
IFS= read -r -d "" label
cd /tmp/xutf-bench
rm -f /tmp/xutf-bench.jsonl
RUSTFLAGS="$rustflags" BENCH_JSON=/tmp/xutf-bench.jsonl BENCH_HOST="$label" \
	cargo bench -q --bench graphemes --bench width --bench truncate --bench wrap --bench throughput
'"'"''
	scp -q "$host":/tmp/xutf-bench.jsonl "$out"
done

uv run scripts/bench_viz.py benches/data/*.jsonl --readme README.md
