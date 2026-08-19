//! Differential validation of the generated category/script tables against
//! the `unicode-properties` and `unicode-script` crates.
//!
//! The oracle crates may lag our UCD snapshot (see [`crate::UNICODE_VERSION`]),
//! so codepoints the oracle reports as `Unassigned` / `Unknown` are excluded:
//! newly-encoded characters diverge by version, not by bug. Everything the
//! oracle *does* assign must match exactly.

use std::{fmt::Write as _, string::String, vec::Vec};

use unicode_properties::UnicodeGeneralCategory;
use unicode_script::UnicodeScript;

use crate::ucd::{
	GeneralCategory as GC, GeneralCategoryGroup as GCG, Script, general_category,
	general_category_group, script,
};

fn oracle_category(c: char) -> GC {
	use unicode_properties::GeneralCategory as O;
	match c.general_category() {
		O::UppercaseLetter => GC::UppercaseLetter,
		O::LowercaseLetter => GC::LowercaseLetter,
		O::TitlecaseLetter => GC::TitlecaseLetter,
		O::ModifierLetter => GC::ModifierLetter,
		O::OtherLetter => GC::OtherLetter,
		O::NonspacingMark => GC::NonspacingMark,
		O::SpacingMark => GC::SpacingMark,
		O::EnclosingMark => GC::EnclosingMark,
		O::DecimalNumber => GC::DecimalNumber,
		O::LetterNumber => GC::LetterNumber,
		O::OtherNumber => GC::OtherNumber,
		O::ConnectorPunctuation => GC::ConnectorPunctuation,
		O::DashPunctuation => GC::DashPunctuation,
		O::OpenPunctuation => GC::OpenPunctuation,
		O::ClosePunctuation => GC::ClosePunctuation,
		O::InitialPunctuation => GC::InitialPunctuation,
		O::FinalPunctuation => GC::FinalPunctuation,
		O::OtherPunctuation => GC::OtherPunctuation,
		O::MathSymbol => GC::MathSymbol,
		O::CurrencySymbol => GC::CurrencySymbol,
		O::ModifierSymbol => GC::ModifierSymbol,
		O::OtherSymbol => GC::OtherSymbol,
		O::SpaceSeparator => GC::SpaceSeparator,
		O::LineSeparator => GC::LineSeparator,
		O::ParagraphSeparator => GC::ParagraphSeparator,
		O::Control => GC::Control,
		O::Format => GC::Format,
		O::Surrogate => GC::Surrogate,
		O::PrivateUse => GC::PrivateUse,
		O::Unassigned => GC::Unassigned,
	}
}

/// Renders contiguous diffs as ranges for triage, then fails when any exist.
fn assert_no_diffs<A, B>(diffs: &[(u32, A, B)])
where
	A: Copy + core::fmt::Debug + PartialEq,
	B: Copy + core::fmt::Debug + PartialEq,
{
	let mut msg = String::new();
	let mut groups = 0usize;
	let mut i = 0usize;
	while i < diffs.len() {
		let (start, ours, theirs) = diffs[i];
		let mut end = start;
		while i + 1 < diffs.len() {
			let (cp, o, t) = diffs[i + 1];
			if cp != end + 1 || o != ours || t != theirs {
				break;
			}
			end = cp;
			i += 1;
		}
		groups += 1;
		if groups <= 40 {
			let _ = writeln!(msg, "U+{start:04X}..=U+{end:04X}: ours {ours:?}, oracle {theirs:?}");
		}
		i += 1;
	}
	assert!(diffs.is_empty(), "{} diffs in {groups} ranges:\n{msg}", diffs.len());
}

/// Every scalar the oracle assigns must have an identical category; oracle
/// `Unassigned` scalars are newer-UCD additions and only checked for being
/// assigned-or-unassigned consistently (ours may know them).
#[test]
fn category_matches_unicode_properties() {
	let mut diffs: Vec<(u32, GC, GC)> = Vec::new();
	for cp in (0..0x11_0000u32).filter_map(char::from_u32) {
		let ours = general_category(cp as u32);
		let theirs = oracle_category(cp);
		if ours != theirs && theirs != GC::Unassigned {
			diffs.push((cp as u32, ours, theirs));
		}
	}
	assert_no_diffs(&diffs);
}

/// Same contract for Script, keyed by UCD long name (our variants are the
/// long names with underscores removed).
#[test]
fn script_matches_unicode_script() {
	let mut diffs: Vec<(u32, Script, unicode_script::Script)> = Vec::new();
	for cp in (0..0x11_0000u32).filter_map(char::from_u32) {
		let ours = script(cp as u32);
		let theirs = cp.script();
		if theirs == unicode_script::Script::Unknown {
			continue;
		}
		if format!("{ours:?}") != theirs.full_name().replace('_', "") {
			diffs.push((cp as u32, ours, theirs));
		}
	}
	assert_no_diffs(&diffs);
}

/// The pi-repo call sites in one place: tokenizer scanner classes
/// (`\p{L}`/`\p{N}`/`\p{M}`/`\p{Han}`), digit-class splits, and the
/// permissive out-of-range/surrogate conventions.
#[test]
fn call_site_semantics() {
	// \p{Han} includes Nl 〇 and Lm 々 (regex-syntax parity).
	assert_eq!(script('中' as u32), Script::Han);
	assert_eq!(script('〇' as u32), Script::Han);
	assert_eq!(script('々' as u32), Script::Han);
	assert_eq!(script('a' as u32), Script::Latin);
	assert_eq!(script(0x0300), Script::Inherited); // combining grave
	assert_eq!(script(' ' as u32), Script::Common);

	assert_eq!(general_category('A' as u32), GC::UppercaseLetter);
	assert_eq!(general_category('ǅ' as u32), GC::TitlecaseLetter);
	assert_eq!(general_category('0' as u32), GC::DecimalNumber);
	assert_eq!(general_category(0x0660), GC::DecimalNumber); // Arabic-Indic zero
	assert_eq!(general_category('½' as u32), GC::OtherNumber);
	assert_eq!(general_category_group('中' as u32), GCG::Letter);
	assert_eq!(general_category_group(0x0300), GCG::Mark);
	assert_eq!(general_category_group(' ' as u32), GCG::Separator);
	assert_eq!(general_category_group('!' as u32), GCG::Punctuation);
	assert_eq!(general_category_group('+' as u32), GCG::Symbol);
	assert_eq!(general_category_group('\t' as u32), GCG::Other);

	// Permissive u32 inputs `char`-based crates cannot classify.
	assert_eq!(general_category(0xD800), GC::Surrogate);
	assert_eq!(script(0xD800), Script::Unknown);
	assert_eq!(general_category(0x11_0000), GC::Unassigned);
	assert_eq!(script(0x11_0000), Script::Unknown);
	assert_eq!(general_category(u32::MAX), GC::Unassigned);
}

/// The [`crate::Ucd`] methods agree with the free functions for `u32`
/// (permissive inputs included) and `char`. Scoped import and qualified
/// paths: the oracle crates' traits also add these method names to `char`
/// in this module.
#[test]
fn trait_matches_free_functions() {
	use crate::ucd::Ucd;
	for cp in [0x41u32, 0x0300, 0x4E2D, 0xD800, 0x11_0000, u32::MAX] {
		assert_eq!(Ucd::general_category(cp), general_category(cp));
		assert_eq!(Ucd::general_category_group(cp), general_category_group(cp));
		assert_eq!(Ucd::script(cp), script(cp));
	}
	assert_eq!(Ucd::script('中'), Script::Han);
	assert_eq!(Ucd::general_category('A'), GC::UppercaseLetter);
	assert_eq!(Ucd::general_category_group('0'), GCG::Number);
}

/// Group derivation covers all 30 categories with the UAX #44 first-letter
/// rule.
#[test]
fn groups_are_exhaustive() {
	use GCG as G;
	let by_group: [(&[GC], G); 7] = [
		(
			&[
				GC::UppercaseLetter,
				GC::LowercaseLetter,
				GC::TitlecaseLetter,
				GC::ModifierLetter,
				GC::OtherLetter,
			],
			G::Letter,
		),
		(&[GC::NonspacingMark, GC::SpacingMark, GC::EnclosingMark], G::Mark),
		(&[GC::DecimalNumber, GC::LetterNumber, GC::OtherNumber], G::Number),
		(
			&[
				GC::ConnectorPunctuation,
				GC::DashPunctuation,
				GC::OpenPunctuation,
				GC::ClosePunctuation,
				GC::InitialPunctuation,
				GC::FinalPunctuation,
				GC::OtherPunctuation,
			],
			G::Punctuation,
		),
		(&[GC::MathSymbol, GC::CurrencySymbol, GC::ModifierSymbol, GC::OtherSymbol], G::Symbol),
		(&[GC::SpaceSeparator, GC::LineSeparator, GC::ParagraphSeparator], G::Separator),
		(&[GC::Control, GC::Format, GC::Surrogate, GC::PrivateUse, GC::Unassigned], G::Other),
	];
	let mut seen = 0usize;
	for (cats, group) in by_group {
		for &cat in cats {
			assert_eq!(cat.group(), group, "{cat:?}");
			seen += 1;
		}
	}
	assert_eq!(seen, 30);
}
