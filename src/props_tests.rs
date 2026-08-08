//! Differential validation of the generated property tables against the
//! `unicode-width` crate (same Unicode version, see [`UNICODE_VERSION`]).
//!
//! String-level width/segmentation semantics are covered by the integration
//! tests; this module pins the raw table data.

use std::{fmt::Write as _, string::String, vec::Vec};

use unicode_width::UnicodeWidthChar;

use crate::props::*;

/// Standalone cell width of one codepoint per the generated table
/// (no cluster context, so width class 3 reads as 1).
fn table_width(cp: u32) -> usize {
	let w = (props(cp) >> WIDTH_SHIFT) & 3;
	if w == WIDTH_EMOJI_TEXT { 1 } else { w as usize }
}

fn class(cp: u32) -> u8 {
	props(cp) & CB_MASK
}

/// Every scalar's width must match `unicode-width`, except a short list of
/// deliberate divergences.
#[test]
fn char_width_matches_unicode_width() {
	// Any entry here must be triaged against unicode-width's tables; the
	// only accepted divergences are documented in `text.rs`.
	let mut diffs: Vec<(u32, usize, usize)> = Vec::new();

	for cp in 0..0x11_0000u32 {
		let Some(c) = char::from_u32(cp) else {
			continue; // surrogates: unrepresentable as char; ours is 1
		};
		let theirs = UnicodeWidthChar::width(c).unwrap_or(0);
		let ours = table_width(cp);
		// U+17D8 KHMER SIGN BEYYAL: unicode-width says 3, our 2-bit width
		// class saturates at 2 cells and reports 1 like wcwidth.
		if ours != theirs && cp != 0x17d8 {
			diffs.push((cp, ours, theirs));
		}
	}

	// Group contiguous codepoints with identical (ours, theirs) for triage.
	let mut msg = String::new();
	let mut groups = 0usize;
	let mut i = 0usize;
	while i < diffs.len() {
		let (start, ours, theirs) = diffs[i];
		let mut end = start;
		while i + 1 < diffs.len() {
			let (cp, o, t) = diffs[i + 1];
			if cp != end + 1 && !(end == 0xd7ff && cp == 0xe000) || (o, t) != (ours, theirs) {
				break;
			}
			end = cp;
			i += 1;
		}
		end = end.max(diffs[i].0);
		writeln!(msg, "U+{start:04X}..U+{end:04X}: ours {ours} theirs {theirs}")
			.expect("writing to a String cannot fail");
		groups += 1;
		i += 1;
	}
	assert!(diffs.is_empty(), "{} diffs in {groups} ranges:\n{msg}", diffs.len());
}

/// Spot checks of grapheme break classes and flag bits for codepoints the
/// cluster rules depend on.
#[test]
fn break_classes_and_flags() {
	assert_eq!(class(0x0d), CB_CR);
	assert_eq!(class(0x0a), CB_LF);
	assert_eq!(class(0x09), CB_CONTROL);
	assert_eq!(class(0x7f), CB_CONTROL);
	assert_eq!(class(0x200d), CB_ZWJ);
	assert_eq!(class(0x0300), CB_EXTEND); // combining grave
	assert_eq!(class(0xfe0f), CB_EXTEND); // VS16
	assert_eq!(class(0x20e3), CB_EXTEND); // enclosing keycap
	assert_eq!(class(0x1f3fb), CB_EXTEND); // emoji modifier (skin tone)
	assert_eq!(class(0xe0067), CB_EXTEND); // tag latin small g
	assert_eq!(class(0x1f1e6), CB_RI); // regional indicator A
	assert_eq!(class(0x0600), CB_PREPEND); // arabic number sign
	assert_eq!(class(0x0903), CB_SPACING_MARK); // devanagari visarga
	assert_eq!(class(0x1100), CB_L);
	assert_eq!(class(0x1160), CB_V);
	assert_eq!(class(0x11a8), CB_T);
	assert_eq!(class(0xac00), CB_LV); // 가
	assert_eq!(class(0xac01), CB_LVT); // 각
	assert_eq!(class(0x094d), CB_EXTEND_INCB_LINKER); // devanagari virama
	assert_eq!(class(0x0915), CB_OTHER_INCB_CONSONANT); // devanagari ka
	assert_eq!(class(b'a' as u32), CB_OTHER);

	// Extended_Pictographic drives GB11.
	assert_ne!(props(0x26a0) & EPIC_BIT, 0); // ⚠
	assert_ne!(props(0x1f600) & EPIC_BIT, 0); // 😀
	assert_eq!(props(b'0' as u32) & EPIC_BIT, 0);
	assert_eq!(props(0x1f1e6) & EPIC_BIT, 0); // RI is not EPic

	// InCB=Extend drives GB9c chains.
	assert_ne!(props(0x0300) & INCB_EXTEND_BIT, 0);
	assert_ne!(props(0x200d) & INCB_EXTEND_BIT, 0); // ZWJ is InCB Extend

	// Width classes the cluster scanner relies on.
	let wclass = |cp: u32| (props(cp) >> WIDTH_SHIFT) & 3;
	assert_eq!(wclass(b'#' as u32), WIDTH_EMOJI_TEXT);
	assert_eq!(wclass(b'0' as u32), WIDTH_EMOJI_TEXT);
	assert_eq!(wclass(0x26a0), WIDTH_EMOJI_TEXT); // ⚠ text presentation
	assert_eq!(wclass(0x2705), 2); // ✅ EAW Wide
	assert_eq!(wclass(0x1f1e6), 1); // RI counts 1 (pair sums to 2)
	assert_eq!(wclass(0x3164), 0); // Hangul filler: default ignorable
	assert_eq!(wclass(0x3131), 2); // Hangul compat jamo: UAX #11 wide
	assert_eq!(wclass(0x00ad), 0); // soft hyphen: default ignorable
	assert_eq!(wclass(0xfffd), 1); // replacement char
	assert_eq!(table_width(0xd800), 1); // lone surrogate (permissive)
	assert_eq!(table_width(0x11_0000), 1); // out of range (permissive)
}
