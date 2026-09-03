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

fn reference_nfkc(input: &str) -> String {
	input.nfkc().collect()
}

fn reference_nfkd(input: &str) -> String {
	input.nfkd().collect()
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
fn matches_compatibility_examples() {
	let cases = [
		"ﬁ ﬂ ﬀ ﬁ ﬁ",                          // ligatures expand
		"ＡＢＣａｂｃ０１２",                 // fullwidth folds to ASCII
		"² ³ µ Å Å ① ②",                      // superscripts, micro, angstrom, circled
		"™ © ® ℠ ㋡ ﾊﾝｸﾞﾙ",                    // symbols, circled katakana, halfwidth Hangul
		"Å A\u{030a} \u{212b}",               // angstrom sign folds then composes
		"\u{1e0b}\u{0323} q\u{0307}\u{0323}", // canonical reordering still applies
		"\u{0344} \u{0f73}\u{0f81}",          // compat expansions carrying combining marks
		"각 각 \u{3131}\u{314f}",             // Hangul jamo, syllables, compatibility jamo
		"ﬁx A\u{030a} \u{1100}\u{1161}\u{11a8} ①",
	];

	for input in cases {
		assert_eq!(input.to_nfkc(), reference_nfkc(input), "NFKC for {input:?}");
		assert_eq!(input.to_nfkd(), reference_nfkd(input), "NFKD for {input:?}");
	}
	// Spot checks against hand-computed values, not just the reference crate.
	assert_eq!("ﬁ".to_nfkc(), "fi");
	assert_eq!("ﬁ".to_nfkd(), "fi");
	assert_eq!("Ａ".to_nfkc(), "A");
	assert_eq!("²".to_nfkc(), "2");
	assert_eq!("µ".to_nfkc(), "μ"); // micro sign folds to Greek mu
	assert_eq!("\u{212b}".to_nfkc(), "Å"); // angstrom sign composes after folding
	assert_eq!("①".to_nfkd(), "1");
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

	let compat_source = "ﬁ Ａ ² µ ①";
	let expected_nfkc = reference_nfkc(compat_source);
	let expected_nfkd = reference_nfkd(compat_source);
	let mut nfkc = String::with_capacity(compat_source.len() * 3);
	nfkc.push_str(compat_source);
	let nfkc_pointer = nfkc.as_ptr();
	let nfkc_capacity = nfkc.capacity();
	nfkc.make_nfkc().unwrap();
	assert_eq!(nfkc, expected_nfkc);
	assert_eq!(nfkc.as_ptr(), nfkc_pointer);
	assert_eq!(nfkc.capacity(), nfkc_capacity);

	let mut nfkd = String::with_capacity(compat_source.len() * 3);
	nfkd.push_str(compat_source);
	let nfkd_pointer = nfkd.as_ptr();
	let nfkd_capacity = nfkd.capacity();
	nfkd.make_nfkd().unwrap();
	assert_eq!(nfkd, expected_nfkd);
	assert_eq!(nfkd.as_ptr(), nfkd_pointer);
	assert_eq!(nfkd.capacity(), nfkd_capacity);
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

	let compat_source = "ﬁ Ａ ²".to_owned();
	let expected = reference_nfkc(&compat_source);
	let mut compat = String::with_capacity(expected.len());
	compat.push_str(&compat_source);
	let pointer = compat.as_ptr();
	let capacity = compat.capacity();
	let output = compat.into_nfkc();
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
	let expected_nfkd = reference_nfkd(&corpus);
	assert_eq!(corpus.to_nfkd(), expected_nfkd);
	assert_eq!(corpus.to_nfkc(), reference_nfkc(&corpus));
	assert_eq!(expected_nfkd.to_nfkc(), reference_nfkc(&expected_nfkd));
}

#[test]
fn normalizes_dense_scalar_sequence_like_unicode_normalization() {
	let corpus: String = (0..=0x10ffff).filter_map(char::from_u32).collect();
	assert_eq!(corpus.to_nfd(), reference_nfd(&corpus));
	assert_eq!(corpus.to_nfc(), reference_nfc(&corpus));
	assert_eq!(corpus.to_nfkd(), reference_nfkd(&corpus));
	assert_eq!(corpus.to_nfkc(), reference_nfkc(&corpus));
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
fn is_nfkc_quick_check_verdicts() {
	// Compatibility characters are never NFKC-quick-check-positive.
	assert!(xutf::is_nfkc("fi AB 2"));
	assert!(xutf::is_nfkc("café résumé"));
	assert!(xutf::is_nfkc("한국어 テスト"));
	assert!(!xutf::is_nfkc("ﬁ")); // ligature expands under NFKC
	assert!(!xutf::is_nfkc("Ａ")); // fullwidth folds to ASCII
	assert!(!xutf::is_nfkc("µ")); // micro sign folds to Greek mu
	assert!(!xutf::is_nfkc("cafe\u{301}"));
	assert!(!xutf::is_nfkc("a\u{0301}\u{0316}b")); // ccc 230 then 220: mis-ordered
	// Quick-check conservatism, mirroring NFC: Maybe codepoints report false
	// even when the text happens to already be NFKC.
	assert!(!xutf::is_nfkc("q\u{0301}"));
}

#[test]
fn is_nfkc_codepoints_matches_str_verdict() {
	let samples = [
		"plain ascii",
		"fi AB 2",
		"ﬁ Ａ ² µ ①",
		"café résumé",
		"cafe\u{301}",
		"한국어",
		"\u{1100}\u{1161}",
		"a\u{0301}\u{0316}b",
		"ﬁx A\u{030a} \u{1100}\u{1161}\u{11a8} ①",
	];
	for s in samples {
		assert_eq!(
			xutf::is_nfkc_codepoints(s.chars().map(|c| c as u32)),
			xutf::is_nfkc(s),
			"verdict drift for {s:?}"
		);
	}
	assert!(xutf::is_nfkc_codepoints([0x41, 0xd800, 0x42]));
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
