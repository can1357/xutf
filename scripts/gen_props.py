#!/usr/bin/env python3
"""Generates src/props.rs: packed Unicode properties for terminal text.

Fetches UCD data files (cached in scripts/ucd/), packs per-codepoint
properties into one byte, and emits a deduplicated two-level trie.

Packed byte layout (must match src/grapheme.rs):
  bits 0..=3  grapheme cluster break class (CB_*)
  bits 4..=5  cell width class (0, 1, 2; 3 = width 1, promotes to 2 when the
              cluster carries U+FE0F VS16 or U+20E3 COMBINING ENCLOSING KEYCAP)
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

UCD = "https://www.unicode.org/Public/UCD/latest/ucd/"
FILES = {
    "UnicodeData.txt": UCD + "UnicodeData.txt",
    "EastAsianWidth.txt": UCD + "EastAsianWidth.txt",
    "GraphemeBreakProperty.txt": UCD + "auxiliary/GraphemeBreakProperty.txt",
    "DerivedCoreProperties.txt": UCD + "DerivedCoreProperties.txt",
    "emoji-data.txt": UCD + "emoji/emoji-data.txt",
    "PropList.txt": UCD + "PropList.txt",
    "Scripts.txt": UCD + "Scripts.txt",
}

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ucd")
OUT = os.path.join(HERE, "..", "src", "props.rs")
OUT_BIN = os.path.join(HERE, "..", "src", "props_bmp.bin")
OUT_UCD = os.path.join(HERE, "..", "src", "ucd_data.rs")
OUT_UCD_BIN = os.path.join(HERE, "..", "src", "ucd_bmp.bin")

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


def fetch(name: str) -> list[str]:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print(f"fetching {FILES[name]}")
        urllib.request.urlretrieve(FILES[name], path)
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


def load_categories() -> list[str]:
    cats = ["Cn"] * NUM_CP
    lines = fetch("UnicodeData.txt")
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


def load_ranged(name: str, prop_field: int = 1, default=None) -> list:
    values = [default] * NUM_CP
    lines = fetch(name)
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
    default_ignorable = load_set(
        "DerivedCoreProperties.txt", "Default_Ignorable_Code_Point"
    )
    grapheme_extend = load_set("DerivedCoreProperties.txt", "Grapheme_Extend")
    prepended_concat = load_set("PropList.txt", "Prepended_Concatenation_Mark")
    emoji = load_set("emoji-data.txt", "Emoji")
    epic = load_set("emoji-data.txt", "Extended_Pictographic")

    # Zero-width prepended concatenation marks (Arabic and Syriac; see the
    # Unicode core spec ch. 9) and DEVANAGARI CARET.
    special_zero = {0x0605, 0x070F, 0x0890, 0x0891, 0x08E2, 0xA8FA}
    # U+115F HANGUL CHOSEONG FILLER carries the syllable's width (2);
    # U+2D7F TIFINAGH CONSONANT JOINER is visible in isolation; the Kirat
    # Rai vowel signs are Grapheme_Extend but render standalone at 1 cell
    # (all per unicode-width).
    never_zero = {0x115F, 0x2D7F, 0x16D63, 0x16D67, 0x16D68, 0x16D69, 0x16D6A}
    # U+17A4 KHMER INDEPENDENT VOWEL QAA renders 2 cells despite EAW N
    # (unicode-width override). U+17D8 KHMER SIGN BEYYAL is 3 cells there;
    # our 2-bit width class keeps it at 1 like wcwidth.
    force_wide = {0x17A4}

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

        # --- cell width class ---------------------------------------------
        # Mirrors unicode-width's `load_zero_widths`: zero for
        # Default_Ignorable_Code_Point, Grapheme_Extend (which pulls in the
        # canonically-combining Mc oddballs), Hangul jungseong/jongseong
        # (GCB V/T ≡ Hangul_Syllable_Type V/T), zero-width prepended
        # concatenation marks, and GCB Prepend letters that are not
        # concatenation marks (Indic consonant prefixes). Controls are 0
        # (unicode-width reports None). 2 for East Asian Wide/Fullwidth.
        # Class 3 marks default-text-presentation emoji: width 1, promoted
        # to 2 by VS16/keycap at the cluster level.
        zero = (
            cp in default_ignorable
            or cp in grapheme_extend
            or g in ("V", "T")
            or cp in special_zero
            or (g == "Prepend" and cp not in prepended_concat)
        ) and cp not in never_zero
        if cat == "Cc" or zero:
            width = 0
        elif eaw[cp] in ("W", "F") or cp in force_wide:
            width = 2
        elif cp in emoji and g != "Regional_Indicator":
            width = 3  # text presentation by default; VS16/keycap promotes
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
    emb = to_ranges(load_set("emoji-data.txt", "Emoji_Modifier_Base"))
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
    w("/// Width class 3: default text presentation emoji — 1 cell, promoted")
    w("/// to 2 by VS16 (U+FE0F) or COMBINING ENCLOSING KEYCAP (U+20E3).")
    w("pub const WIDTH_EMOJI_TEXT: u8 = 3;")
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
    w("/// `true` when `cp` accepts an emoji skin-tone modifier")
    w("/// (`Emoji_Modifier_Base`).")
    w("#[inline]")
    w("pub fn is_emoji_modifier_base(cp: u32) -> bool {")
    w(f"    if !(0x{emb[0][0]:x}..=0x{emb[-1][1]:x}).contains(&cp) {{")
    w("        return false;")
    w("    }")
    w("    EMOJI_MODIFIER_BASE")
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
    w("/// Sorted `Emoji_Modifier_Base` ranges (inclusive).")
    w(f"static EMOJI_MODIFIER_BASE: [(u32, u32); {len(emb)}] = [")
    for i in range(0, len(emb), 6):
        row = ", ".join(f"(0x{lo:x}, 0x{hi:x})" for lo, hi in emb[i : i + 6])
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


def build_ucd() -> tuple[list[int], list[str]]:
    """Packed General_Category + Script word per codepoint, and the script
    index order (Unknown/Common/Inherited first, then alphabetical)."""
    cats = load_categories()
    scripts = load_ranged("Scripts.txt", default="Unknown")
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


def emit_ucd(words: list[int], order: list[str], version) -> str:
    size, shift, stage1, blocks, wide = best_trie(words[BMP_END:], elem_size=2)
    leaves = []
    for key in blocks:
        leaves.extend(key)
    with open(OUT_UCD_BIN, "wb") as f:
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
    w('static UCD_BMP: [u8; 0x2_0000] = *include_bytes!("ucd_bmp.bin");')
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
    words, order = build_ucd()
    ucd_src = emit_ucd(words, order, version)
    with open(OUT_UCD, "w", encoding="utf-8") as f:
        f.write(ucd_src)
    # Normalize to the repo's rustfmt style so regeneration never drifts.
    subprocess.run(["rustfmt", OUT, OUT_UCD], check=True)
    size, shift, stage1, blocks, wide = best_trie(props[BMP_END:])
    print(
        f"UCD {version[0]}.{version[1]}.{version[2]}: {BMP_END} bytes direct BMP + "
        f"astral trie (block {1 << shift}, {len(blocks)} unique blocks, "
        f"{'u16' if wide else 'u8'} stage1, {size} bytes) -> {os.path.relpath(OUT)}"
    )
    size, shift, stage1, blocks, wide = best_trie(words[BMP_END:], elem_size=2)
    print(
        f"ucd: {len(order)} scripts, {2 * BMP_END} bytes direct BMP + astral trie "
        f"(block {1 << shift}, {len(blocks)} unique blocks, "
        f"{'u16' if wide else 'u8'} stage1, {size} bytes) -> {os.path.relpath(OUT_UCD)}"
    )


if __name__ == "__main__":
    sys.exit(main())
