//! Behavioral and differential tests for extended grapheme clusters.

use unicode_segmentation::UnicodeSegmentation;
use xutf::{Utf8, Utf16, Utf32, graphemes, graphemes_str};

const CORPUS: &[&str] = &[
	"",
	"plain ASCII prose",
	"a\r\nb\rc\nd\r\n",
	"e\u{301} o\u{308}\u{301} cafe\u{301}",
	"Ελληνικά α\u{313}\u{301}",
	"가나다 \u{1100}\u{1161}\u{11a8} \u{1102}\u{1161}",
	"\u{915}\u{94d}\u{937} क्षत्रिय",
	"தமிழ் ภาษาไทย",
	"\u{600}123 \u{605}٤٥",
	"😀 ⚠️ 0️⃣ 👍🏽",
	"👨‍👩‍👧‍👦 👩‍❤️‍💋‍👨",
	"🇺🇸🇦🇩🇨",
	"🏴\u{e0067}\u{e0062}\u{e0065}\u{e006e}\u{e0067}\u{e007f}",
	"a\u{200c}b\u{200b}c\u{2060}d",
];

fn assert_same_segmentation(s: &str) {
	let actual: Vec<&str> = graphemes_str(s).collect();
	let expected: Vec<&str> = UnicodeSegmentation::graphemes(s, true).collect();
	assert_eq!(actual, expected, "segmentation differed for {s:?}");
}

#[test]
fn curated_segmentation_matches_unicode_segmentation() {
	for &s in CORPUS {
		assert_same_segmentation(s);
	}
}

const fn next_random(x: &mut u64) -> u64 {
	*x = x
		.wrapping_mul(6_364_136_223_846_793_005)
		.wrapping_add(1_442_695_040_888_963_407);
	*x
}

const fn random_scalar(x: &mut u64) -> char {
	loop {
		let cp = (next_random(x) % 0x11_0000) as u32;
		if let Some(ch) = char::from_u32(cp) {
			return ch;
		}
	}
}

#[test]
fn deterministic_fuzz_matches_unicode_segmentation() {
	const WEIGHTED: &[char] = &[
		'\u{200d}', '\u{200d}', '\u{fe0f}', '\u{fe0f}', '\u{20e3}', '🇦', '🇧', '🇺', '🇸', '\u{94d}',
		'\u{94d}', '\u{ccd}', '\u{ccd}', '\u{1100}', '\u{1161}', '\u{11a8}', '🏻', '🏽', '🏿', '😀',
		'👩', '❤', '\u{301}', '\u{308}', '\u{600}', '\r', '\n',
	];
	let mut state = 0x47a9_8d31_c2e6_50fbu64;
	for _ in 0..2_000 {
		let len = (next_random(&mut state) % 65) as usize;
		let mut s = String::new();
		for _ in 0..len {
			let selector = next_random(&mut state);
			let ch = if selector & 3 != 0 {
				WEIGHTED[(selector as usize >> 2) % WEIGHTED.len()]
			} else {
				random_scalar(&mut state)
			};
			s.push(ch);
		}
		assert_same_segmentation(&s);
	}
}

fn utf16_clusters(units: &[u16]) -> Vec<String> {
	graphemes::<Utf16<false>>(units)
		.map(|g| String::from_utf16(g.units).unwrap())
		.collect()
}

fn utf32_clusters(units: &[u32]) -> Vec<String> {
	graphemes::<Utf32<false>>(units)
		.map(|g| {
			g.units
				.iter()
				.map(|&cp| char::from_u32(cp).unwrap())
				.collect()
		})
		.collect()
}

#[test]
fn segmentation_is_identical_across_encodings_and_byte_orders() {
	for &s in CORPUS {
		let expected: Vec<String> = graphemes::<Utf8>(s.as_bytes())
			.map(|g| String::from_utf8(g.units.to_vec()).unwrap())
			.collect();
		let native16: Vec<u16> = s.encode_utf16().collect();
		let native32: Vec<u32> = s.chars().map(u32::from).collect();
		assert_eq!(utf16_clusters(&native16), expected, "UTF-16 for {s:?}");
		assert_eq!(utf32_clusters(&native32), expected, "UTF-32 for {s:?}");

		let foreign16: Vec<u16> = native16.iter().map(|u| u.swap_bytes()).collect();
		let foreign: Vec<String> = graphemes::<Utf16<true>>(&foreign16)
			.map(|g| {
				let native: Vec<u16> = g.units.iter().map(|u| u.swap_bytes()).collect();
				String::from_utf16(&native).unwrap()
			})
			.collect();
		assert_eq!(foreign, expected, "foreign UTF-16 for {s:?}");
	}
}

fn assert_width(s: &str, expected: usize) {
	let clusters: Vec<_> = graphemes::<Utf8>(s.as_bytes()).collect();
	assert_eq!(clusters.len(), 1, "expected one cluster for {s:?}");
	assert_eq!(clusters[0].width, expected, "wrong width for {s:?}");
}

#[test]
fn cluster_width_follows_terminal_semantics() {
	for &(s, width) in &[
		("⚠️", 2),
		("ℹ️", 2),
		("❤️", 2),
		("0️⃣", 2),
		("⚠", 1),
		("✅", 2),
		("❌", 2),
		("界", 2),
		("👨‍👩‍👧", 2),
		("🇺🇸", 2),
		("🇺", 1),
		("e\u{301}", 1),
		("\u{1100}\u{1161}\u{11a8}", 2),
		("가", 2),
		("\u{915}\u{94d}\u{937}", 2),
		("\u{3164}", 0),
		("\u{3131}", 2),
		("\t", 0),
		("\r\n", 0),
		("👍🏽", 2),
		("🏴\u{e0067}\u{e0062}\u{e0065}\u{e006e}\u{e0067}\u{e007f}", 2),
		("#\u{fe0f}\u{20e3}", 2),
		("\u{fe0f}", 0),
	] {
		assert_width(s, width);
	}
}

#[test]
fn malformed_and_empty_inputs_are_permissive() {
	assert!(graphemes::<Utf8>(&[]).next().is_none());

	let surrogate = [0xd800u16];
	let cluster = graphemes::<Utf16<false>>(&surrogate).next().unwrap();
	assert_eq!(cluster.units, &surrogate);
	assert_eq!(cluster.width, 1);

	let truncated = [0xe4u8, 0xb8];
	let clusters: Vec<_> = graphemes::<Utf8>(&truncated).collect();
	assert_eq!(clusters.iter().map(|g| g.units.len()).sum::<usize>(), truncated.len());
}

#[test]
fn cloned_iterator_resumes_independently() {
	let mut original = graphemes_str("a🇺🇸e\u{301}");
	assert_eq!(original.next(), Some("a"));
	let mut cloned = original.clone();
	assert_eq!(original.next(), Some("🇺🇸"));
	assert_eq!(cloned.next(), Some("🇺🇸"));
	assert_eq!(original.collect::<Vec<_>>(), vec!["e\u{301}"]);
	assert_eq!(cloned.collect::<Vec<_>>(), vec!["e\u{301}"]);
}
