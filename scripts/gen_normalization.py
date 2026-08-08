#!/usr/bin/env python3
"""Generates compact Unicode 17 NFC/NFD lookup tables."""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Iterable

UCD = "https://www.unicode.org/Public/UCD/latest/ucd/"
FILES = {
    "UnicodeData.txt": UCD + "UnicodeData.txt",
    "DerivedNormalizationProps.txt": UCD + "DerivedNormalizationProps.txt",
}

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ucd")
OUT = os.path.join(HERE, "..", "src", "normalize_data.rs")
NUM_CP = 0x110000

S_BASE = 0xAC00
L_BASE = 0x1100
V_BASE = 0x1161
T_BASE = 0x11A7
L_COUNT = 19
V_COUNT = 21
T_COUNT = 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT

CCC_MASK = 0xFF
NFD_NO_BIT = 1 << 8
NFC_MAYBE_BIT = 1 << 9
NFC_NO_BIT = 1 << 10
DECOMPOSITION_SHIFT = 11
DECOMPOSITION_BITS = 11
DECOMPOSITION_ORDER_BIT = 1 << (DECOMPOSITION_SHIFT + DECOMPOSITION_BITS)
COMPOSITION_SHIFT = DECOMPOSITION_SHIFT + DECOMPOSITION_BITS + 1
COMPOSITION_BITS = 32 - COMPOSITION_SHIFT


def fetch(name: str) -> list[str]:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        with urllib.request.urlopen(FILES[name]) as response, open(path, "wb") as output:
            output.write(response.read())
    with open(path, encoding="utf-8") as source:
        return source.readlines()


def parse_range(raw: str) -> tuple[int, int]:
    bounds = raw.strip().split("..")
    lo = int(bounds[0], 16)
    return lo, int(bounds[-1], 16)


def parse_unicode_data() -> tuple[list[int], dict[int, tuple[int, ...]]]:
    ccc = [0] * NUM_CP
    mappings: dict[int, tuple[int, ...]] = {}
    pending_range: tuple[int, int] | None = None

    for raw in fetch("UnicodeData.txt"):
        fields = raw.rstrip("\n").split(";")
        cp = int(fields[0], 16)
        name = fields[1]
        combining = int(fields[3])
        if name.endswith(", First>"):
            pending_range = (cp, combining)
            continue
        if name.endswith(", Last>"):
            assert pending_range is not None
            start, expected = pending_range
            assert expected == combining
            ccc[start : cp + 1] = [combining] * (cp - start + 1)
            pending_range = None
            continue

        ccc[cp] = combining
        decomposition = fields[5]
        if decomposition and not decomposition.startswith("<"):
            mappings[cp] = tuple(int(part, 16) for part in decomposition.split())

    assert pending_range is None
    return ccc, mappings


def parse_normalization_properties() -> tuple[set[int], set[int], set[int]]:
    nfd_no: set[int] = set()
    nfc_maybe: set[int] = set()
    nfc_no: set[int] = set()

    for raw in fetch("DerivedNormalizationProps.txt"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(";")]
        lo, hi = parse_range(fields[0])
        prop = fields[1]
        value = fields[2] if len(fields) > 2 else "Y"
        target: set[int] | None = None
        if prop == "NFD_QC" and value == "N":
            target = nfd_no
        elif prop == "NFC_QC" and value == "M":
            target = nfc_maybe
        elif prop == "NFC_QC" and value == "N":
            target = nfc_no
        if target is not None:
            target.update(range(lo, hi + 1))

    return nfd_no, nfc_maybe, nfc_no


def hangul_decomposition(cp: int) -> tuple[int, ...] | None:
    if not S_BASE <= cp < S_BASE + S_COUNT:
        return None
    index = cp - S_BASE
    result = [L_BASE + index // N_COUNT, V_BASE + index % N_COUNT // T_COUNT]
    trailing = index % T_COUNT
    if trailing:
        result.append(T_BASE + trailing)
    return tuple(result)


def best_trie(values: list[int]) -> tuple[int, int, list[int], list[tuple[int, ...]]]:
    best: tuple[int, int, list[int], list[tuple[int, ...]]] | None = None
    for shift in range(4, 10):
        width = 1 << shift
        indexes: dict[tuple[int, ...], int] = {}
        stage1: list[int] = []
        blocks: list[tuple[int, ...]] = []
        for at in range(0, len(values), width):
            block = tuple(values[at : at + width])
            index = indexes.get(block)
            if index is None:
                index = len(blocks)
                indexes[block] = index
                blocks.append(block)
            stage1.append(index)
        assert len(blocks) <= 0x10000
        size = len(stage1) * 2 + len(blocks) * width * 4
        candidate = (size, shift, stage1, blocks)
        if best is None or size < best[0]:
            best = candidate
    assert best is not None
    return best


def emit_array(
    lines: list[str], name: str, rust_type: str, values: Iterable[int], formatter, per_line: int
) -> None:
    values = list(values)
    lines.append(f"pub(super) static {name}: [{rust_type}; {len(values)}] = [")
    for at in range(0, len(values), per_line):
        row = ", ".join(formatter(value) for value in values[at : at + per_line])
        lines.append(f"\t{row},")
    lines.append("];")
    lines.append("")


def emit() -> tuple[str, str]:
    ccc, mappings = parse_unicode_data()
    nfd_no, nfc_maybe, nfc_no = parse_normalization_properties()

    @functools.cache
    def fully_decompose(cp: int) -> tuple[int, ...]:
        hangul = hangul_decomposition(cp)
        if hangul is not None:
            return hangul
        mapping = mappings.get(cp)
        if mapping is None:
            return (cp,)
        return tuple(part for child in mapping for part in fully_decompose(child))

    decomposition_ids: dict[tuple[int, ...], int] = {}
    decomposition_by_cp: dict[int, int] = {}
    for cp in sorted(mappings):
        decomposition = fully_decompose(cp)
        index = decomposition_ids.setdefault(decomposition, len(decomposition_ids) + 1)
        decomposition_by_cp[cp] = index

    assert len(decomposition_ids) < 1 << DECOMPOSITION_BITS
    ordered_decompositions: list[tuple[int, ...] | None] = [None] * (len(decomposition_ids) + 1)
    for decomposition, index in decomposition_ids.items():
        ordered_decompositions[index] = decomposition

    records = [0]
    decomposition_chars: list[int] = []
    max_decomposition = 0
    for decomposition in ordered_decompositions[1:]:
        assert decomposition is not None
        max_decomposition = max(max_decomposition, len(decomposition))
        assert len(decomposition) < 8
        records.append(len(decomposition_chars) << 3 | len(decomposition))
        decomposition_chars.extend(decomposition)

    compositions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for composite, mapping in mappings.items():
        if len(mapping) == 2 and composite not in nfc_no:
            first, second = mapping
            compositions[first].append((second, composite))

    composition_ids = {starter: index + 1 for index, starter in enumerate(sorted(compositions))}
    assert len(composition_ids) < 1 << COMPOSITION_BITS
    composition_offsets = [0, 0]
    composition_pairs: list[int] = []
    max_compositions = 0
    for starter in sorted(compositions):
        pairs = sorted(compositions[starter])
        assert len({second for second, _ in pairs}) == len(pairs)
        max_compositions = max(max_compositions, len(pairs))
        composition_pairs.extend((second << 21) | composite for second, composite in pairs)
        composition_offsets.append(len(composition_pairs))
    assert len(composition_pairs) < 0x10000

    properties = [0] * NUM_CP
    for cp in range(NUM_CP):
        word = ccc[cp] & CCC_MASK
        if cp in nfd_no:
            word |= NFD_NO_BIT
        if cp in nfc_maybe:
            word |= NFC_MAYBE_BIT
        elif cp in nfc_no:
            word |= NFC_NO_BIT
        decomposition_id = decomposition_by_cp.get(cp, 0)
        word |= decomposition_id << DECOMPOSITION_SHIFT
        if decomposition_id and any(ccc[part] for part in ordered_decompositions[decomposition_id]):
            word |= DECOMPOSITION_ORDER_BIT
        word |= composition_ids.get(cp, 0) << COMPOSITION_SHIFT
        properties[cp] = word

    size, shift, stage1, blocks = best_trie(properties)
    stage2 = [value for block in blocks for value in block]

    version_match = re.search(r"-(\d+)\.(\d+)\.(\d+)\.txt", fetch("DerivedNormalizationProps.txt")[0])
    assert version_match is not None
    version = tuple(int(part) for part in version_match.groups())

    lines = [
        f"// Generated from Unicode {version[0]}.{version[1]}.{version[2]}; run scripts/gen_normalization.py.",
        "",
        f"pub(super) const NORMALIZATION_SHIFT: usize = {shift};",
        f"pub(super) const NORMALIZATION_MASK: usize = 0x{(1 << shift) - 1:x};",
        f"pub(super) const NORMALIZATION_UNICODE_VERSION: (u8, u8, u8) = {version};",
        "",
    ]
    emit_array(lines, "NORMALIZATION_STAGE1", "u16", stage1, lambda value: str(value), 16)
    emit_array(
        lines,
        "NORMALIZATION_STAGE2",
        "u32",
        stage2,
        lambda value: f"0x{value:08x}",
        8,
    )
    emit_array(
        lines,
        "DECOMPOSITION_RECORDS",
        "u16",
        records,
        lambda value: f"0x{value:04x}",
        12,
    )
    emit_array(
        lines,
        "DECOMPOSITION_CHARS",
        "u32",
        decomposition_chars,
        lambda value: f"0x{value:06x}",
        10,
    )
    emit_array(
        lines,
        "COMPOSITION_OFFSETS",
        "u16",
        composition_offsets,
        lambda value: str(value),
        16,
    )
    emit_array(
        lines,
        "COMPOSITION_PAIRS",
        "u64",
        composition_pairs,
        lambda value: f"0x{value:012x}",
        6,
    )

    stats = (
        f"UCD {version[0]}.{version[1]}.{version[2]}: trie block {1 << shift}, "
        f"{len(blocks)} unique blocks, {size} bytes; {len(decomposition_ids)} decompositions "
        f"(max {max_decomposition}), {len(composition_pairs)} compositions "
        f"(max {max_compositions}/starter)"
    )
    return "\n".join(lines), stats


def main() -> int:
    source, stats = emit()
    with open(OUT, "w", encoding="utf-8") as output:
        output.write(source)
    subprocess.run(["rustfmt", OUT], check=True)
    print(f"{stats} -> {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
