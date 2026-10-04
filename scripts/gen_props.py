#!/usr/bin/env python3
"""Generates src/props.rs: packed Unicode properties for terminal text.

Fetches UCD data files (cached in scripts/ucd/), packs per-codepoint
properties into one byte, and emits a deduplicated two-level trie.

Packed byte layout (must match src/grapheme.rs):
  bits 0..=3  grapheme cluster break class (CB_*)
  bits 4..=5  standalone cell width per kitty's rules (0, 1 or 2)
  bit  6      Extended_Pictographic (UAX #29 GB11)
  bit  7      InCB=Extend (UAX #29 GB9c)

Run: uv run scripts/gen_props.py
"""

import os
import re
import struct
import subprocess
import sys
import urllib.request

UCD = "https://www.unicode.org/Public/UCD/latest/"
# Paths within a Unicode release root. `None` as a version means "latest".
FILES = {
    "UnicodeData.txt": "ucd/UnicodeData.txt",
    "EastAsianWidth.txt": "ucd/EastAsianWidth.txt",
    "GraphemeBreakProperty.txt": "ucd/auxiliary/GraphemeBreakProperty.txt",
    "DerivedCoreProperties.txt": "ucd/DerivedCoreProperties.txt",
    "emoji-data.txt": "ucd/emoji/emoji-data.txt",
    "emoji-sequences.txt": "emoji/emoji-sequences.txt",
    "PropList.txt": "ucd/PropList.txt",
    "Scripts.txt": "ucd/Scripts.txt",
}

# `General_Category`/`Script` tables are additionally emitted for this older
# release, because consumers that must byte-match a regex engine or tokenizer
# built against it cannot use the newest assignments. See `PINNED` in
# `src/ucd.rs`.
UCD_PINNED = "16.0.0"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ucd")
OUT = os.path.join(HERE, "..", "src", "props.rs")
OUT_BIN = os.path.join(HERE, "..", "src", "props_bmp.bin")


def ucd_out(version) -> tuple[str, str]:
    """Generated module and blob for a UCD generation, keyed by its major
    version: `src/ucd/<major>/{data.rs,bmp.bin}`."""
    out_dir = os.path.join(HERE, "..", "src", "ucd", str(version[0]))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "data.rs"), os.path.join(out_dir, "bmp.bin")

NUM_CP = 0x110000

# Grapheme cluster break classes; InCB Linker/Consonant folded in as variants
# of their break class so GB9c costs no extra bits.
CB = {
    "Other": 0,
    "CR": 1,
    "LF": 2,
    "Control": 3,
    "Extend": 4,
    "ExtendInCbLinker": 5,
    "ZWJ": 6,
    "Regional_Indicator": 7,
    "Prepend": 8,
    "SpacingMark": 9,
    "L": 10,
    "V": 11,
    "T": 12,
    "LV": 13,
    "LVT": 14,
    "OtherInCbConsonant": 15,
}

WIDTH_SHIFT = 4
EPIC_BIT = 1 << 6
INCB_EXTEND_BIT = 1 << 7

# General_Category values in UAX #44 table order. This is the discriminant
# contract with the hand-written `GeneralCategory` enum in src/ucd.rs; the
# differential tests in src/ucd_tests.rs pin it against an oracle crate.
GC = [
    "Lu", "Ll", "Lt", "Lm", "Lo",
    "Mn", "Mc", "Me",
    "Nd", "Nl", "No",
    "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
    "Sm", "Sc", "Sk", "So",
    "Zs", "Zl", "Zp",
    "Cc", "Cf", "Cs", "Co", "Cn",
]
# UCD word layout: bits 0..=4 general-category index, bits 5.. script index.
SCRIPT_SHIFT = 5


def fetch(name: str, version: str | None = None) -> list[str]:
    """Read a UCD data file, downloading it into the cache on first use.

    `version` is a release like `"16.0.0"`; `None` reads the `latest` tree,
    whose cache lives directly in `scripts/ucd/`.
    """
    base = UCD if version is None else f"https://www.unicode.org/Public/{version}/"
    cache = CACHE if version is None else os.path.join(CACHE, version)
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        print(f"fetching {base + FILES[name]}")
        urllib.request.urlretrieve(base + FILES[name], path)
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def parse_ranges(lines: list[str], prop_field: int = 1):
    """Yields (lo, hi, value) from `XXXX[..YYYY] ; value` data lines."""
    for line in lines:
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        fields = [f.strip() for f in body.split(";")]
        if len(fields) <= prop_field:
            continue
        cps = fields[0]
        if ".." in cps:
            lo, hi = (int(x, 16) for x in cps.split(".."))
        else:
            lo = hi = int(cps, 16)
        yield lo, hi, fields[prop_field]


def parse_missing(lines: list[str]):
    """Yields (lo, hi, value) from `# @missing: XXXX..YYYY; value` lines."""
    for line in lines:
        m = re.match(r"#\s*@missing:\s*([0-9A-F]+)\.\.([0-9A-F]+);\s*([^;]+)", line)
        if m:
            yield int(m.group(1), 16), int(m.group(2), 16), m.group(3).strip()


def unicode_version(lines: list[str]) -> tuple[int, int, int]:
    m = re.search(r"-(\d+)\.(\d+)\.(\d+)\.txt", lines[0])
    assert m, "no version in header"
    return tuple(int(g) for g in m.groups())


def load_categories(version: str | None = None) -> list[str]:
    cats = ["Cn"] * NUM_CP
    lines = fetch("UnicodeData.txt", version)
    first = None
    for line in lines:
        fields = line.split(";")
        cp, name, cat = int(fields[0], 16), fields[1], fields[2]
        if name.endswith(", First>"):
            first = cp
        elif name.endswith(", Last>"):
            assert first is not None
            for c in range(first, cp + 1):
                cats[c] = cat
            first = None
        else:
            cats[cp] = cat
    return cats


def load_ranged(
    name: str, prop_field: int = 1, default=None, version: str | None = None
) -> list:
    values = [default] * NUM_CP
    lines = fetch(name, version)
    # @missing defaults first (later data lines override).
    for lo, hi, v in parse_missing(lines):
        for cp in range(lo, min(hi, NUM_CP - 1) + 1):
            values[cp] = v
    for lo, hi, v in parse_ranges(lines, prop_field):
        for cp in range(lo, hi + 1):
            values[cp] = v
    return values


def load_set(name: str, prop: str, prop_field: int = 1) -> set[int]:
    out = set()
    for lo, hi, v in parse_ranges(fetch(name), prop_field):
        if v == prop:
            out.update(range(lo, hi + 1))
    return out


def load_emoji_sequences() -> tuple[set[int], set[int]]:
    """Wide emoji and emoji presentation bases from emoji-sequences.txt.

    Wide: `Basic_Emoji` without a `FE0F` in its entry (emoji presentation by
    default), every code point of an `RGI_Emoji_Flag_Sequence` and the
    leading code point of every `RGI_Emoji_Tag_Sequence` and
    `RGI_Emoji_Modifier_Sequence`. Presentation bases: all of those plus the
    `Basic_Emoji` that need `FE0F` and the keycap bases, i.e. every code point
    a variation selector can switch between text and emoji presentation.
    """
    wide, bases = set(), set()
    for line in fetch("emoji-sequences.txt"):
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        seq, kind = (f.strip() for f in body.split(";")[:2])
        parts = seq.split()
        if ".." in parts[0]:
            lo, hi = (int(x, 16) for x in parts[0].split(".."))
            first = set(range(lo, hi + 1))
        else:
            first = {int(parts[0], 16)}
        if kind == "Basic_Emoji":
            if len(parts) == 1:
                wide |= first
        elif kind == "RGI_Emoji_Flag_Sequence":
            first = {int(p, 16) for p in parts}
            wide |= first
        elif kind in ("RGI_Emoji_Tag_Sequence", "RGI_Emoji_Modifier_Sequence"):
            wide |= first
        elif kind != "Emoji_Keycap_Sequence":
            continue
        bases |= first
    return wide, bases


def load_incb() -> dict[int, str]:
    """InCB from DerivedCoreProperties: `cp ; InCB; Extend|Linker|Consonant`."""
    out = {}
    for line in fetch("DerivedCoreProperties.txt"):
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        fields = [f.strip() for f in body.split(";")]
        if len(fields) < 3 or fields[1] != "InCB":
            continue
        cps = fields[0]
        if ".." in cps:
            lo, hi = (int(x, 16) for x in cps.split(".."))
        else:
            lo = hi = int(cps, 16)
        for cp in range(lo, hi + 1):
            out[cp] = fields[2]
    return out


def build_props() -> tuple[list[int], tuple[int, int, int]]:
    version = unicode_version(fetch("EastAsianWidth.txt"))
    cats = load_categories()
    eaw = load_ranged("EastAsianWidth.txt", default="N")
    gcb = load_ranged("GraphemeBreakProperty.txt", default="Other")
    incb = load_incb()
    other_ignorable = load_set("PropList.txt", "Other_Default_Ignorable_Code_Point")
    epic = load_set("emoji-data.txt", "Extended_Pictographic")
    wide_emoji, _ = load_emoji_sequences()

    props = [0] * NUM_CP
    for cp in range(NUM_CP):
        cat = cats[cp]
        g = gcb[cp]

        # --- grapheme break class -----------------------------------------
        cls = CB.get(g, CB["Other"])
        v = incb.get(cp)
        if v == "Linker":
            assert g == "Extend", f"InCB Linker U+{cp:04X} has GCB {g}"
            cls = CB["ExtendInCbLinker"]
        elif v == "Consonant":
            assert g == "Other", f"InCB Consonant U+{cp:04X} has GCB {g}"
            cls = CB["OtherInCbConsonant"]

        # --- cell width ---------------------------------------------------
        # kitty's rules ("The algorithm for splitting text into cells" in its
        # text sizing protocol), first match wins:
        #   2  regional indicators, East Asian Wide/Fullwidth (the `@missing`
        #      defaults make unassigned CJK ideographs wide) and wide emoji;
        #   0  marks (M*), format characters (Cf),
        #      Other_Default_Ignorable_Code_Point, and the code points kitty
        #      rejects as invalid: controls (Cc), surrogates (Cs) and
        #      noncharacters;
        #   1  everything else, ambiguous, private use and unassigned alike.
        if g == "Regional_Indicator" or eaw[cp] in ("W", "F") or cp in wide_emoji:
            width = 2
        elif (
            cat[0] == "M"
            or cat in ("Cf", "Cc", "Cs")
            or cp in other_ignorable
            or (cp & 0xFFFE) == 0xFFFE
            or 0xFDD0 <= cp <= 0xFDEF
        ):
            width = 0
        else:
            width = 1

        b = cls | (width << WIDTH_SHIFT)
        if cp in epic:
            b |= EPIC_BIT
        if v == "Extend":
            b |= INCB_EXTEND_BIT
        props[cp] = b

    return props, version


BMP_END = 0x10000


def best_trie(astral: list[int], elem_size: int = 1):
    """Two-level trie over the astral planes only; the BMP is direct-indexed.

    `elem_size` is the leaf element width in bytes (u8 props, u16 ucd words).
    """
    best = None
    for shift in (6, 7, 8):
        block = 1 << shift
        blocks: dict[tuple, int] = {}
        stage1 = []
        for i in range(0, len(astral), block):
            key = tuple(astral[i : i + block])
            idx = blocks.setdefault(key, len(blocks))
            stage1.append(idx)
        wide = len(blocks) > 256
        size = len(stage1) * (2 if wide else 1) + len(blocks) * block * elem_size
        if best is None or size < best[0]:
            best = (size, shift, stage1, list(blocks), wide)
    return best


def to_ranges(cps: set[int]) -> list[tuple[int, int]]:
    out = []
    for cp in sorted(cps):
        if out and cp == out[-1][1] + 1:
            out[-1] = (out[-1][0], cp)
        else:
            out.append((cp, cp))
    return out


def emit(props: list[int], version) -> str:
    size, shift, stage1, blocks, wide = best_trie(props[BMP_END:])
    epb = to_ranges(load_emoji_sequences()[1])
    leaves = bytearray()
    for key in blocks:
        leaves.extend(key)
    with open(OUT_BIN, "wb") as f:
        f.write(bytes(props[:BMP_END]))

    idx_ty = "u16" if wide else "u8"
    lines = []
    w = lines.append
    w("//! Generated by `scripts/gen_props.py` — do not edit.")
    w("//!")
    w("//! Packed per-codepoint properties for grapheme segmentation and")
    w("//! terminal cell width, from UCD %d.%d.%d. See the generator for" % version)
    w("//! the byte layout contract shared with `grapheme.rs`.")
    w("")
    w("// The class constants document the full packed-byte contract; not")
    w("// every consumer references every class.")
    w("#![allow(")
    w('    dead_code,')
    w('    reason = "the generated table documents packed-byte classes used by consumers"')
    w(")]")
    w("")
    v = "(%d, %d, %d)" % version
    w("/// Unicode version of the generated property tables.")
    w(f"pub const UNICODE_VERSION: (u8, u8, u8) = {v};")
    w("")
    w("// Grapheme cluster break classes (bits 0..=3 of a props byte).")
    order = sorted(CB.items(), key=lambda kv: kv[1])
    names = {
        "Regional_Indicator": "CB_RI",
        "ExtendInCbLinker": "CB_EXTEND_INCB_LINKER",
        "OtherInCbConsonant": "CB_OTHER_INCB_CONSONANT",
        "SpacingMark": "CB_SPACING_MARK",
    }
    for name, val in order:
        rust = names.get(name, "CB_" + name.upper())
        w(f"pub const {rust}: u8 = {val};")
    w("")
    w("pub const CB_MASK: u8 = 0x0f;")
    w(f"pub const WIDTH_SHIFT: u8 = {WIDTH_SHIFT};")
    w(f"pub const EPIC_BIT: u8 = 0x{EPIC_BIT:02x};")
    w(f"pub const INCB_EXTEND_BIT: u8 = 0x{INCB_EXTEND_BIT:02x};")
    w("")
    w("/// Props byte for codepoints outside the Unicode range (permissive")
    w("/// decoding of garbage UTF-32): break class Other, width 1.")
    w(f"pub const DEFAULT_PROPS: u8 = 0x{1 << WIDTH_SHIFT:02x};")
    w("")
    w("/// Packed properties of `cp`: one direct load for the BMP, a two-level")
    w("/// trie for the astral planes.")
    w("#[inline(always)]")
    w("pub fn props(cp: u32) -> u8 {")
    w("    if cp < 0x1_0000 {")
    w("        return PROPS_BMP[cp as usize];")
    w("    }")
    w("    if cp >= 0x11_0000 {")
    w("        return DEFAULT_PROPS;")
    w("    }")
    w("    let astral = (cp - 0x1_0000) as usize;")
    w(f"    let block = STAGE1[astral >> {shift}] as usize;")
    w("    // SAFETY: every STAGE1 entry indexes a whole LEAVES block by")
    w("    // construction, so the largest in-block offset stays in bounds.")
    w(
        f"    unsafe {{ core::hint::assert_unchecked(block * {1 << shift}"
        f" + 0x{(1 << shift) - 1:x} < LEAVES.len()) }};"
    )
    w(f"    LEAVES[block * {1 << shift} + (astral & 0x{(1 << shift) - 1:x})]")
    w("}")
    w("")
    w("/// `true` when a variation selector switches `cp` between text and")
    w("/// emoji presentation: the code points of `Basic_Emoji`, keycap, flag,")
    w("/// tag and modifier sequences in emoji-sequences.txt.")
    w("#[inline]")
    w("pub fn is_emoji_presentation_base(cp: u32) -> bool {")
    w(f"    if !(0x{epb[0][0]:x}..=0x{epb[-1][1]:x}).contains(&cp) {{")
    w("        return false;")
    w("    }")
    w("    EMOJI_PRESENTATION_BASE")
    w("        .binary_search_by(|&(lo, hi)| {")
    w("            if cp < lo {")
    w("                core::cmp::Ordering::Greater")
    w("            } else if cp > hi {")
    w("                core::cmp::Ordering::Less")
    w("            } else {")
    w("                core::cmp::Ordering::Equal")
    w("            }")
    w("        })")
    w("        .is_ok()")
    w("}")
    w("")
    w("/// Sorted emoji presentation base ranges (inclusive).")
    w(f"static EMOJI_PRESENTATION_BASE: [(u32, u32); {len(epb)}] = [")
    for i in range(0, len(epb), 6):
        row = ", ".join(f"(0x{lo:x}, 0x{hi:x})" for lo, hi in epb[i : i + 6])
        w(f"    {row},")
    w("];")
    w("")

    w("/// Direct-indexed properties of the Basic Multilingual Plane.")
    w('static PROPS_BMP: [u8; 0x1_0000] = *include_bytes!("props_bmp.bin");')
    w("")

    def table(name, ty, vals, per_line):
        w(f"static {name}: [{ty}; {len(vals)}] = [")
        for i in range(0, len(vals), per_line):
            row = ", ".join(str(x) for x in vals[i : i + per_line])
            w(f"    {row},")
        w("];")

    table("STAGE1", idx_ty, stage1, 24)
    w("")
    table("LEAVES", "u8", list(leaves), 24)
    return "\n".join(lines) + "\n"


def build_ucd(version: str | None = None) -> tuple[list[int], list[str]]:
    """Packed General_Category + Script word per codepoint, and the script
    index order (Unknown/Common/Inherited first, then alphabetical)."""
    cats = load_categories(version)
    scripts = load_ranged("Scripts.txt", default="Unknown", version=version)
    order = ["Unknown", "Common", "Inherited"] + sorted(
        set(scripts) - {"Unknown", "Common", "Inherited"}
    )
    assert len(order) <= 256, "script index no longer fits 8 bits"
    sidx = {n: i for i, n in enumerate(order)}
    gidx = {n: i for i, n in enumerate(GC)}
    words = [
        gidx[cats[cp]] | (sidx[scripts[cp]] << SCRIPT_SHIFT) for cp in range(NUM_CP)
    ]
    return words, order


def emit_ucd(words: list[int], order: list[str], version, out_bin: str) -> str:
    size, shift, stage1, blocks, wide = best_trie(words[BMP_END:], elem_size=2)
    leaves = []
    for key in blocks:
        leaves.extend(key)
    with open(out_bin, "wb") as f:
        f.write(struct.pack(f"<{BMP_END}H", *words[:BMP_END]))

    idx_ty = "u16" if wide else "u8"
    default_word = GC.index("Cn")  # script Unknown is index 0
    lines = []
    w = lines.append
    w("//! Generated by `scripts/gen_props.py` — do not edit.")
    w("//!")
    w("//! Per-codepoint `General_Category` and `Script` from UCD %d.%d.%d," % version)
    w("//! behind the public API in `ucd.rs`. Word layout: bits 0..=4 the")
    w("//! general-category index (UAX #44 order, the `GeneralCategory`")
    w(f"//! discriminants), bits {SCRIPT_SHIFT}.. the index into `SCRIPT_BY_INDEX`.")
    w("")
    w("/// Script property values (UAX #24) of UCD %d.%d.%d." % version)
    w("///")
    w("/// Values not yet encoded map to [`Script::Unknown`]. New Unicode")
    w("/// versions add variants, hence `#[non_exhaustive]`.")
    w("#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]")
    w("#[non_exhaustive]")
    w("#[repr(u8)]")
    w("pub enum Script {")
    for i, name in enumerate(order):
        w(f"    /// Script `{name}`.")
        w(f"    {name.replace('_', '')} = {i},")
    w("}")
    w("")
    w("/// Script of each packed script index, padded to 256 with `Unknown`")
    w("/// so the 8-bit index in a ucd word never bounds-checks.")
    w(f"pub(super) static SCRIPT_BY_INDEX: [Script; 256] = [")
    padded = [n.replace("_", "") for n in order] + ["Unknown"] * (256 - len(order))
    for i in range(0, 256, 8):
        w("    " + ", ".join(f"Script::{n}" for n in padded[i : i + 8]) + ",")
    w("];")
    w("")
    w("/// Unicode version of these `General_Category`/`Script` tables.")
    w("pub const UCD_VERSION: (u8, u8, u8) = (%d, %d, %d);" % version)
    w("")
    w("/// Word for codepoints outside the Unicode range (permissive decoding")
    w("/// of garbage UTF-32): `General_Category` `Cn`, script `Unknown`.")
    w(f"pub(super) const DEFAULT_WORD: u16 = {default_word};")
    w("")
    w("/// Packed category+script word of `cp`: one direct load for the BMP,")
    w("/// a two-level trie for the astral planes.")
    w("#[inline(always)]")
    w("pub(super) fn word(cp: u32) -> u16 {")
    w("    if cp < 0x1_0000 {")
    w("        let i = (cp as usize) * 2;")
    w("        return u16::from_le_bytes([UCD_BMP[i], UCD_BMP[i + 1]]);")
    w("    }")
    w("    if cp >= 0x11_0000 {")
    w("        return DEFAULT_WORD;")
    w("    }")
    w("    let astral = (cp - 0x1_0000) as usize;")
    w(f"    let block = STAGE1[astral >> {shift}] as usize;")
    w("    // SAFETY: every STAGE1 entry indexes a whole LEAVES block by")
    w("    // construction, so the largest in-block offset stays in bounds.")
    w(
        f"    unsafe {{ core::hint::assert_unchecked(block * {1 << shift}"
        f" + 0x{(1 << shift) - 1:x} < LEAVES.len()) }};"
    )
    w(f"    LEAVES[block * {1 << shift} + (astral & 0x{(1 << shift) - 1:x})]")
    w("}")
    w("")
    w("/// Direct-indexed words of the Basic Multilingual Plane (u16 LE).")
    w(
        'static UCD_BMP: [u8; 0x2_0000] = *include_bytes!("'
        + os.path.basename(out_bin)
        + '");'
    )
    w("")

    def table(name, ty, vals, per_line):
        w(f"static {name}: [{ty}; {len(vals)}] = [")
        for i in range(0, len(vals), per_line):
            row = ", ".join(str(x) for x in vals[i : i + per_line])
            w(f"    {row},")
        w("];")

    table("STAGE1", idx_ty, stage1, 24)
    w("")
    table("LEAVES", "u16", leaves, 16)
    return "\n".join(lines) + "\n"


def main():
    props, version = build_props()
    src = emit(props, version)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(src)

    pinned_version = unicode_version(fetch("Scripts.txt", UCD_PINNED))
    generations = []
    for release, ver in ((None, version), (UCD_PINNED, pinned_version)):
        out_rs, out_bin = ucd_out(ver)
        words, order = build_ucd(release)
        with open(out_rs, "w", encoding="utf-8") as f:
            f.write(emit_ucd(words, order, ver, out_bin))
        generations.append((ver, words, order, out_rs))

    # Normalize to the repo's rustfmt style so regeneration never drifts.
    subprocess.run(["rustfmt", OUT, *(g[3] for g in generations)], check=True)

    size, shift, stage1, blocks, wide = best_trie(props[BMP_END:])
    print(
        f"UCD {version[0]}.{version[1]}.{version[2]}: {BMP_END} bytes direct BMP + "
        f"astral trie (block {1 << shift}, {len(blocks)} unique blocks, "
        f"{'u16' if wide else 'u8'} stage1, {size} bytes) -> {os.path.relpath(OUT)}"
    )
    for ver, words, order, out_rs in generations:
        size, shift, stage1, blocks, wide = best_trie(words[BMP_END:], elem_size=2)
        print(
            f"ucd {ver[0]}.{ver[1]}.{ver[2]}: {len(order)} scripts, "
            f"{2 * BMP_END} bytes direct BMP + astral trie (block {1 << shift}, "
            f"{len(blocks)} unique blocks, {'u16' if wide else 'u8'} stage1, "
            f"{size} bytes) -> {os.path.relpath(out_rs)}"
        )


if __name__ == "__main__":
    sys.exit(main())
