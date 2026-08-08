#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "seaborn>=0.13", "pandas>=2"]
# ///
"""Renders committed benchmark data into seaborn figures and README tables.

Reads the JSONL files produced by the benches when `BENCH_JSON` is set (see
`benches/common.rs`), keeps the latest record per `(host, bench)`, draws one
PNG per bench under `benches/data/` — bars colored by implementation (xutf
vibrant, reference crates muted), rows labelled `input / arch`, absolute
GB/s (or time per call for truncation) as labels — and emits per-host tables.
With `--readme` the markdown (image grid + tables) replaces the section
between the `<!-- benches:begin -->` / `<!-- benches:end -->` markers.

Capture data (per machine, then commit): see `scripts/bench_hosts.sh`.

Render:
    uv run scripts/bench_viz.py benches/data/*.jsonl --readme README.md
"""

import argparse
import json
import sys
from pathlib import Path

BEGIN = "<!-- benches:begin -->"
END = "<!-- benches:end -->"

# All bench inputs are built to this size (`TARGET` in the benches); converts
# the truncation panel from GB/s of the line to time per call.
INPUT_BYTES = 1024 * 1024

# The transcode/equality tables cover many inputs; the figures show ascii plus
# one representative multi-byte input per byte-length class (no cyrillic/syn-*
# rows), the tables keep everything.
TRANSCODE_INPUTS = frozenset(("ascii", "cjk-3b", "emoji-4b"))

# Impl columns kept out of the figures AND the README tables; simdutf-valid
# skips validation, so it isn't an apples-to-apples comparison anywhere.
HIDDEN_IMPLS = frozenset(("simdutf-valid",))


def keep_cut(name: str) -> bool:
    return name.endswith(("/80", "/half"))


def keep_transcode(name: str) -> bool:
    return name in TRANSCODE_INPUTS


# bench title -> (figure title, note, time-based, input filter); the filter
# applies to figures and tables alike. None -> dropped from the README
# entirely. Unlisted benches get a default GB/s figure, unfiltered.
META = {
    "Grapheme clusters": ("grapheme iteration", "GB/s, higher is better", False, None),
    "Visible width": ("visible width", "GB/s, higher is better", False, None),
    "Width truncation": (
        "truncate to width",
        "time per cut of a 1 MiB line, lower is better",
        True,
        keep_cut,
    ),
    "word wrap at 80 columns": (
        "word wrap, 80 columns",
        "GB/s, higher is better",
        False,
        None,
    ),
    "word wrap at 40 columns": None,
    "UTF-8 -> UTF-16": ("transcode UTF-8 → UTF-16", "GB/s of source, higher is better", False, keep_transcode),
    "UTF-16 -> UTF-8": ("transcode UTF-16 → UTF-8", "GB/s of source, higher is better", False, keep_transcode),
    "UTF-8 -> UTF-32": ("transcode UTF-8 → UTF-32", "GB/s of source, higher is better", False, keep_transcode),
    "UTF-32 -> UTF-8": ("transcode UTF-32 → UTF-8", "GB/s of source, higher is better", False, keep_transcode),
    "equality": ("equality", "GB/s, higher is better", False, keep_transcode),
}

VIBRANT = ["#f4511e", "#fb8c00", "#fdd835"]
MUTED = ["#78909c", "#a5b6c0", "#546e7a", "#cfd8dc", "#90a4ae"]

# GitHub dark-mode canvas — the figures sit in the README.
GH = {"bg": "#0d1117", "grid": "#21262d", "text": "#e6edf3", "muted": "#9198a1"}
GH_RC = {
    "figure.facecolor": GH["bg"],
    "savefig.facecolor": GH["bg"],
    "axes.facecolor": GH["bg"],
    "axes.edgecolor": GH["grid"],
    "grid.color": GH["grid"],
    "text.color": GH["text"],
    "axes.labelcolor": GH["text"],
    "xtick.color": GH["text"],
    "ytick.color": GH["muted"],
}


def load(paths: list[str]) -> list[dict]:
    """Latest record per (host, bench), in first-seen order."""
    latest: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["host"], rec["bench"])
            if key not in latest:
                order.append(key)
            latest[key] = rec
    return [latest[key] for key in order]


def hosts_of(records: list[dict]) -> list[str]:
    seen: list[str] = []
    for rec in records:
        if rec["host"] not in seen:
            seen.append(rec["host"])
    return seen


def benches_of(records: list[dict]) -> list[str]:
    seen: list[str] = []
    for rec in records:
        if rec["bench"] not in seen:
            seen.append(rec["bench"])
    return seen


def arch_of(host: str) -> str:
    low = host.lower()
    if "x86" in low or "xeon" in low or "amd" in low:
        return "x86"
    if any(k in low for k in ("arm", "aarch", "apple", "m1", "m2", "m3", "m4")):
        return "arm64"
    return low


def slug(text: str) -> str:
    out = []
    for c in text.lower():
        out.append(c if c.isalnum() else "-")
    joined = "".join(out)
    while "--" in joined:
        joined = joined.replace("--", "-")
    return joined.strip("-")


def fmt_gbps(v: float) -> str:
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def fmt_time(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.3g}µs"
    return f"{ns / 1_000_000:.3g}ms"


def impl_palette(impls: list[str]) -> dict[str, str]:
    palette = {}
    vibrant, muted = iter(VIBRANT), iter(MUTED)
    for name in impls:
        if name.startswith("xutf"):
            palette[name] = next(vibrant, VIBRANT[0])
        else:
            palette[name] = next(muted, MUTED[-1])
    return palette


def plot_bench(bench: str, records: list[dict], png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    title, note, time_based, keep = META.get(bench) or (
        bench,
        "GB/s, higher is better",
        False,
        None,
    )
    hosts = hosts_of(records)
    recs = [r for r in records if r["bench"] == bench]
    impls = [i for i in recs[0]["impls"] if i not in HIDDEN_IMPLS]
    palette = impl_palette(impls)

    tidy = []
    for rec in recs:
        arch = arch_of(rec["host"])
        for row in rec["rows"]:
            if keep and not keep(row["input"]):
                continue
            for name, gbps in zip(recs[0]["impls"], row["values"]):
                if name in HIDDEN_IMPLS:
                    continue
                if gbps <= 0:
                    continue
                value = INPUT_BYTES / gbps if time_based else gbps
                tidy.append(
                    {"row": f"{row['input']} / {arch}", "impl": name, "value": value}
                )
    frame = pd.DataFrame(tidy)
    inputs = [r["input"] for r in recs[0]["rows"] if not keep or keep(r["input"])]
    order = [f"{i} / {arch_of(h)}" for i in inputs for h in hosts]

    sns.set_theme(style="whitegrid", context="notebook", rc=GH_RC)
    width = max(7.5, 1.8 + len(order) * (0.30 + 0.34 * len(impls)))
    fig, ax = plt.subplots(figsize=(width, 5.4))
    sns.barplot(
        frame,
        x="row",
        y="value",
        hue="impl",
        hue_order=impls,
        order=order,
        palette=palette,
        saturation=1.0,
        ax=ax,
    )
    ax.set_yscale("log")
    fmt = fmt_time if time_based else fmt_gbps
    # Time labels are wide ("5.33ms"); stagger alternating impl columns so
    # near-equal neighbors don't collide.
    for i, container in enumerate(ax.containers):
        ax.bar_label(
            container,
            fmt=lambda v, f=fmt: f(v),
            padding=3 + (11 if time_based and i % 2 else 0),
            fontsize=8,
            color=GH["text"],
        )
    # Headroom for the value labels (log scale).
    ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
    ax.set_title(
        f"{title}  ·  {note}",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=30,
        color=GH["text"],
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(range(len(order)), [o.replace(" / ", "\n") for o in order], fontsize=10)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=len(impls),
        frameon=False,
        fontsize=10,
        columnspacing=1.4,
        handlelength=1.2,
    )
    ax.grid(False, axis="x")
    ax.tick_params(axis="y", labelsize=9)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.savefig(png, dpi=170, bbox_inches="tight", facecolor=GH["bg"])
    plt.close(fig)


def bar_count(bench: str, records: list[dict]) -> int:
    """Total bars the figure will draw; sparser figures sort first."""
    _, _, _, keep = META.get(bench) or (bench, "", False, None)
    recs = [r for r in records if r["bench"] == bench]
    impls = [i for i in recs[0]["impls"] if i not in HIDDEN_IMPLS]
    inputs = [r for r in recs[0]["rows"] if not keep or keep(r["input"])]
    return len(inputs) * len(recs) * len(impls)


def cell(value: float, base: float, is_base: bool, is_best: bool) -> str:
    """One table cell, mirroring `benches/common.rs::print_ratio_table`:
    `▼ +X%` = slower than xutf, `▲ -X%` = ahead of xutf. The fastest
    implementation(s) in the row are bolded."""
    if value <= 0:
        return "N/A"
    if is_base or abs(value / base - 1.0) < 0.01:
        text = f"{value:.1f} GB/s (1.00x)"
    elif value > base:
        text = f"{value:.1f} GB/s (▲ -{(value / base - 1.0) * 100:.1f}%)"
    else:
        text = f"{value:.1f} GB/s (▼ +{(base / value - 1.0) * 100:.1f}%)"
    return f"**{text}**" if is_best else text


def host_tables(host: str, records: list[dict]) -> str:
    out = [f"### {host}", ""]
    for rec in (r for r in records if r["host"] == host):
        meta = META.get(rec["bench"], ...)
        if meta is None:
            continue
        keep = meta[3] if meta is not ... else None
        shown = [i for i, name in enumerate(rec["impls"]) if name not in HIDDEN_IMPLS]
        impls = [rec["impls"][i] for i in shown]
        out.append(f"**{rec['bench']}** ({rec['unit']})")
        out.append("")
        out.append("| input | " + " | ".join(impls) + " |")
        out.append("|" + "---|" * (len(impls) + 1))
        for row in rec["rows"]:
            if keep and not keep(row["input"]):
                continue
            values = [row["values"][i] for i in shown]
            base, best = values[0], max(values)
            cells = [
                cell(v, base, i == 0, v >= best * 0.99)
                for i, v in enumerate(values)
            ]
            out.append(f"| {row['input']} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def render(records: list[dict], images: list[tuple[str, str]]) -> str:
    parts = [BEGIN, ""]
    for title, path in images:
        parts.append(f"![{title}]({path})")
        parts.append("")
    parts.append("<details><summary>Raw numbers per host</summary>")
    parts.append("")
    for host in hosts_of(records):
        parts.append(host_tables(host, records))
    parts.append("</details>")
    parts.append(END)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", nargs="+", help="JSONL files from BENCH_JSON runs")
    ap.add_argument("--out-dir", default="benches/data")
    ap.add_argument("--readme", help="splice output between the README markers")
    args = ap.parse_args()

    records = load(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images: list[tuple[str, str]] = []
    # Sparsest figures first (fewest bars), META order as tiebreak; benches
    # missing from META come last.
    ordered = [b for b in META if b in benches_of(records)]
    ordered += [b for b in benches_of(records) if b not in META]
    ordered.sort(key=lambda b: bar_count(b, records))
    for bench in ordered:
        if META.get(bench, ...) is None:
            continue
        png = out_dir / f"{slug(bench)}.png"
        plot_bench(bench, records, png)
        title = (META.get(bench) or (bench,))[0]
        images.append((title, png.as_posix()))
        print(f"wrote {png}")

    section = render(records, images)
    if not args.readme:
        print(section)
        return 0

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    begin, end = text.find(BEGIN), text.find(END)
    if begin < 0 or end < 0:
        sys.exit(f"{args.readme}: missing {BEGIN} / {END} markers")
    readme.write_text(
        text[:begin] + section + text[end + len(END):], encoding="utf-8"
    )
    print(f"updated {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
