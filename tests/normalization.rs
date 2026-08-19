#![allow(
	clippy::unicode_not_nfc,
	reason = "normalization fixtures intentionally exercise decomposed text"
)]

use unicode_normalization::UnicodeNormalization as _;
use xutf::{IntoUnicodeNormalized, MakeUnicodeNormalized, ToUnicodeNormalized};

fn reference_nfc(input: &str) -> String {
	input.nfc().collect()
}

fn reference_nfd(input: &str) -> String {
	input.nfd().collect()
}

#[test]
fn matches_canonical_examples_and_ordering() {
	let cases = [
		"ASCII stays byte-for-byte stable",
		"A\u{030a} Å Ω Ω",
		"q\u{0307}\u{0323}",
		"\u{1e0b}\u{0323}",
		"\u{0344}\u{0f73}\u{0f75}\u{0f81}",
		"각 각 한글",
		"가\u{302e}\u{0301} café nai\u{308}ve",
		"\u{1100}\u{1161}\u{11a8}\u{1102}\u{1161}",
	];

	for input in cases {
		assert_eq!(input.to_nfc(), reference_nfc(input), "NFC for {input:?}");
		assert_eq!(input.to_nfd(), reference_nfd(input), "NFD for {input:?}");
	}
}

#[test]
fn make_normalized_uses_only_existing_capacity() {
	let source = "Å A\u{030a} q\u{0307}\u{0323} 각 \u{0344}";
	let expected_nfc = reference_nfc(source);
	let expected_nfd = reference_nfd(source);

	let mut nfc = String::with_capacity(source.len() * 3);
	nfc.push_str(source);
	let nfc_pointer = nfc.as_ptr();
	let nfc_capacity = nfc.capacity();
	nfc.make_nfc().unwrap();
	assert_eq!(nfc, expected_nfc);
	assert_eq!(nfc.as_ptr(), nfc_pointer);
	assert_eq!(nfc.capacity(), nfc_capacity);

	let mut nfd = String::with_capacity(source.len() * 3);
	nfd.push_str(source);
	let nfd_pointer = nfd.as_ptr();
	let nfd_capacity = nfd.capacity();
	nfd.make_nfd().unwrap();
	assert_eq!(nfd, expected_nfd);
	assert_eq!(nfd.as_ptr(), nfd_pointer);
	assert_eq!(nfd.capacity(), nfd_capacity);
}

#[test]
fn make_normalized_reports_capacity_without_modifying_input() {
	let source = "\u{0344}".repeat(32);
	let mut input = source.clone().into_boxed_str().into_string();
	let original_pointer = input.as_ptr();
	let original_capacity = input.capacity();
	let required = reference_nfd(&source).len();
	assert!(required > original_capacity);

	let error = input.make_nfd().unwrap_err();
	assert_eq!(error.required_capacity(), required);
	assert_eq!(input, source);
	assert_eq!(input.as_ptr(), original_pointer);
	assert_eq!(input.capacity(), original_capacity);
}

#[test]
fn consuming_normalization_reuses_sufficient_capacity() {
	let source = "A\u{030a} q\u{0307}\u{0323} \u{0344} 각";
	let expected = reference_nfd(source);
	let mut input = String::with_capacity(expected.len());
	input.push_str(source);
	let pointer = input.as_ptr();
	let capacity = input.capacity();

	let output = input.into_nfd();
	assert_eq!(output, expected);
	assert_eq!(output.as_ptr(), pointer);
	assert_eq!(output.capacity(), capacity);

	let nfc_source = reference_nfd("Å ḍ̇ 각");
	let expected = reference_nfc(&nfc_source);
	let pointer = nfc_source.as_ptr();
	let capacity = nfc_source.capacity();
	let output = nfc_source.into_nfc();
	assert_eq!(output, expected);
	assert_eq!(output.as_ptr(), pointer);
	assert_eq!(output.capacity(), capacity);
}

#[test]
fn normalizes_every_scalar_like_unicode_normalization() {
	let mut corpus = String::with_capacity(0x500000);
	for cp in 0..=0x10ffff {
		if let Some(ch) = char::from_u32(cp) {
			corpus.push(ch);
			corpus.push(' ');
		}
	}

	let expected_nfd = reference_nfd(&corpus);
	assert_eq!(corpus.to_nfd(), expected_nfd);
	assert_eq!(corpus.to_nfc(), reference_nfc(&corpus));
	assert_eq!(expected_nfd.to_nfc(), reference_nfc(&expected_nfd));
}

#[test]
fn normalizes_dense_scalar_sequence_like_unicode_normalization() {
	let corpus: String = (0..=0x10ffff).filter_map(char::from_u32).collect();
	assert_eq!(corpus.to_nfd(), reference_nfd(&corpus));
	assert_eq!(corpus.to_nfc(), reference_nfc(&corpus));
}

#[test]
fn is_nfc_quick_check_verdicts() {
	// Composed stays borrowable; decomposed and mis-ordered do not.
	assert!(xutf::is_nfc("café résumé"));
	assert!(xutf::is_nfc("한국어 テスト"));
	assert!(!xutf::is_nfc("cafe\u{301}")); // NFD combining acute
	assert!(!xutf::is_nfc("a\u{0316}\u{0301}b")); // ccc 220 after... wait 220 then 230 is ordered; reversed:
	assert!(!xutf::is_nfc("a\u{0301}\u{0316}b")); // ccc 230 then 220: mis-ordered
	// Quick-check conservatism: Maybe codepoints report false even when
	// the text happens to already be NFC.
	assert!(!xutf::is_nfc("q\u{0301}")); // acute after q composes with nothing, still Maybe
}

#[test]
fn is_nfc_codepoints_matches_str_verdict() {
	let samples = [
		"plain ascii",
		"café résumé",
		"cafe\u{301}",
		"한국어",
		"\u{1100}\u{1161}", // decomposed Hangul LV
		"a\u{0301}\u{0316}b",
		"👨‍👩‍👧‍👦 🇹🇷",
		"中文English中文",
	];
	for s in samples {
		assert_eq!(
			xutf::is_nfc_codepoints(s.chars().map(|c| c as u32)),
			xutf::is_nfc(s),
			"verdict drift for {s:?}"
		);
	}
	// Lone surrogate (impossible in str) must be inert, not a panic.
	assert!(xutf::is_nfc_codepoints([0x41, 0xd800, 0x42]));
}

#[test]
fn canonical_combining_class_known_values() {
	assert_eq!(xutf::canonical_combining_class('a' as u32), 0);
	assert_eq!(xutf::canonical_combining_class(0x0301), 230); // combining acute
	assert_eq!(xutf::canonical_combining_class(0x0316), 220); // combining grave below
	assert_eq!(xutf::canonical_combining_class(0x3099), 8); // kana voicing
	assert_eq!(xutf::canonical_combining_class(0x110000), 0); // out of range
}
