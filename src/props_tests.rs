//! Spot checks of the generated property tables: the width rules kitty
//! applies, in their order of precedence, and the break classes and flags the
//! cluster rules read.
//!
//! String-level width/segmentation semantics are covered by the integration
//! tests; this module pins the raw table data.

use crate::props::*;

/// Standalone cell width of one codepoint per the generated table.
fn table_width(cp: u32) -> usize {
	usize::from((props(cp) >> WIDTH_SHIFT) & 3)
}

fn class(cp: u32) -> u8 {
	props(cp) & CB_MASK
}

/// kitty's width rules, first match wins: wide (regional indicators, East
/// Asian Wide/Fullwidth, wide emoji), then zero (marks, format characters,
/// other default-ignorables, invalid code points), then one.
#[test]
fn widths_follow_kittys_rule_order() {
	for (cp, width, what) in [
		(0x1f1e6, 2, "regional indicator, East Asian Width N"),
		(0x4e2d, 2, "CJK ideograph"),
		(0x2fffd, 2, "unassigned CJK, wide by the EAW @missing default"),
		(0xff21, 2, "fullwidth A"),
		(0x1f600, 2, "emoji presentation"),
		(0x261d, 2, "text-default emoji modifier base"),
		(0x302a, 2, "wide nonspacing mark: wide wins over zero"),
		(0x3164, 2, "wide default-ignorable Hangul filler"),
		(0x115f, 2, "wide Hangul choseong filler"),
		(0x0301, 0, "Mn"),
		(0x093e, 0, "Mc"),
		(0x20dd, 0, "Me"),
		(0x00ad, 0, "Cf soft hyphen"),
		(0x0600, 0, "Cf prepended concatenation mark"),
		(0x1160, 0, "other default-ignorable Lo"),
		(0x0000, 0, "NUL"),
		(0x007f, 0, "Cc"),
		(0xd800, 0, "surrogate"),
		(0xfdd0, 0, "noncharacter"),
		(0x1fffe, 0, "plane-end noncharacter"),
		(0x10ffff, 0, "last code point, a noncharacter"),
		(0x1161, 1, "conjoining jungseong"),
		(0x0d4e, 1, "Prepend letter"),
		(0x0e33, 1, "Thai SARA AM, a spacing mark of class Lo"),
		(0x2764, 1, "text-default emoji"),
		(0x00b1, 1, "ambiguous"),
		(0xe000, 1, "private use"),
		(0x0378, 1, "unassigned"),
		(0xfffd, 1, "replacement character"),
		(0x11_0000, 1, "out of range (permissive)"),
	] {
		assert_eq!(table_width(cp), width, "U+{cp:04X} {what}");
	}
}

/// The code points a variation selector resizes: `Basic_Emoji` either way,
/// keycap bases, flag letters and modifier sequence bases.
#[test]
fn emoji_presentation_bases() {
	for cp in [0x23, 0x2a, 0x30, 0x39, 0xa9, 0x2764, 0x3030, 0x1f600, 0x1f1e6, 0x261d, 0x1f3fb] {
		assert!(is_emoji_presentation_base(cp), "U+{cp:04X}");
	}
	for cp in [0x00, 0x61, 0x4e2d, 0x2123, 0x1f1e5, 0x1f200, 0x10ffff, 0x11_0000] {
		assert!(!is_emoji_presentation_base(cp), "U+{cp:04X}");
	}
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
}
