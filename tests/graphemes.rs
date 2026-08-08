//! Behavioral and differential tests for extended grapheme clusters.

use unicode_segmentation::UnicodeSegmentation;
use xutf::{Encoding, Utf8, Utf16, Utf32, grapheme_indices_str, graphemes, graphemes_str};

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

	let mut reversed: Vec<&str> = graphemes_str(s).rev().collect();
	reversed.reverse();
	assert_eq!(reversed, expected, "reverse segmentation differed for {s:?}");

	assert_eq!(graphemes_str(s).len(), expected.len(), "exact len differed for {s:?}");
}

fn assert_reverse_matches_forward<E: Encoding>(units: &[E::Unit])
where
	E::Unit: core::fmt::Debug,
{
	let forward: Vec<(Vec<E::Unit>, usize)> = graphemes::<E>(units)
		.map(|g| (g.units.to_vec(), g.width))
		.collect();
	let mut backward: Vec<(Vec<E::Unit>, usize)> = graphemes::<E>(units)
		.rev()
		.map(|g| (g.units.to_vec(), g.width))
		.collect();
	backward.reverse();
	assert_eq!(backward, forward);
	assert_eq!(graphemes::<E>(units).len(), forward.len());
}

#[test]
fn reverse_iteration_matches_forward_across_encodings() {
	for &s in CORPUS {
		assert_reverse_matches_forward::<Utf8>(s.as_bytes());

		let native16: Vec<u16> = s.encode_utf16().collect();
		assert_reverse_matches_forward::<Utf16<false>>(&native16);
		let foreign16: Vec<u16> = native16.iter().map(|u| u.swap_bytes()).collect();
		assert_reverse_matches_forward::<Utf16<true>>(&foreign16);

		let native32: Vec<u32> = s.chars().map(u32::from).collect();
		assert_reverse_matches_forward::<Utf32<false>>(&native32);
		let foreign32: Vec<u32> = native32.iter().map(|u| u.swap_bytes()).collect();
		assert_reverse_matches_forward::<Utf32<true>>(&foreign32);
	}
}

#[test]
fn double_ended_iteration_meets_in_the_middle() {
	for &s in CORPUS {
		let expected: Vec<&str> = graphemes_str(s).collect();
		let mut iter = graphemes_str(s);
		let mut front = Vec::new();
		let mut back = Vec::new();
		while let Some(g) = iter.next() {
			front.push(g);
			match iter.next_back() {
				Some(g) => back.push(g),
				None => break,
			}
		}
		back.reverse();
		front.extend(back);
		assert_eq!(front, expected, "meet-in-middle differed for {s:?}");
	}
}

#[test]
fn reverse_iteration_is_permissive_on_malformed_input() {
	assert!(graphemes::<Utf8>(&[]).next_back().is_none());

	let cases: &[&[u8]] =
		&[&[0x80], &[0xff, 0xfe, 0x80], &[0xe4, 0xb8], &[0xf0, 0x90, 0x80], &[b'a', 0x80, b'b']];
	for &input in cases {
		let consumed: usize = graphemes::<Utf8>(input).rev().map(|g| g.units.len()).sum();
		assert_eq!(consumed, input.len(), "reverse dropped units for {input:?}");
	}

	let surrogate = [0xd800u16, 0x41];
	let consumed: usize = graphemes::<Utf16<false>>(&surrogate)
		.rev()
		.map(|g| g.units.len())
		.sum();
	assert_eq!(consumed, surrogate.len());
}

#[test]
fn grapheme_indices_reverse_reports_forward_offsets() {
	for &s in CORPUS {
		let forward: Vec<(usize, &str)> = grapheme_indices_str(s).collect();
		let mut backward: Vec<(usize, &str)> = grapheme_indices_str(s).rev().collect();
		backward.reverse();
		assert_eq!(backward, forward, "indices differed for {s:?}");
	}
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

#[test]
fn grapheme_indices_str_report_byte_offsets() {
	let input = "ASCII 界 👨‍👩‍👧‍👦 🇺🇸\r\ne\u{301}";
	let indexed: Vec<_> = xutf::grapheme_indices_str(input).collect();
	let mut cursor = 0;
	for &(at, cluster) in &indexed {
		assert_eq!(at, cursor);
		assert_eq!(&input[at..at + cluster.len()], cluster);
		cursor += cluster.len();
	}
	assert_eq!(cursor, input.len());
	assert_eq!(
		indexed
			.iter()
			.map(|(_, cluster)| *cluster)
			.collect::<String>(),
		input
	);

	let clusters: Vec<_> = graphemes_str(input).collect();
	assert_eq!(
		indexed
			.iter()
			.map(|(_, cluster)| *cluster)
			.collect::<Vec<_>>(),
		clusters
	);
}

#[test]
fn grapheme_indices_count_utf16_code_units() {
	let input = "A😀界👨‍👩‍👧‍👦e\u{301}";
	let units: Vec<u16> = input.encode_utf16().collect();
	let actual: Vec<_> = xutf::grapheme_indices::<Utf16<false>>(&units)
		.map(|(at, cluster)| (at, String::from_utf16(cluster.units).unwrap()))
		.collect();
	let expected: Vec<_> = UnicodeSegmentation::grapheme_indices(input, true)
		.map(|(byte_at, cluster)| (input[..byte_at].encode_utf16().count(), cluster.to_owned()))
		.collect();
	assert_eq!(actual, expected);
}

#[test]
fn grapheme_control_status_uses_the_base_codepoint() {
	for input in ["\r", "\n", "\r\n", "\x07", "\u{9b}"] {
		let cluster = graphemes::<Utf8>(input.as_bytes()).next().unwrap();
		assert!(cluster.is_control(), "{input:?} should be a control");
	}
	for input in ["a", "\u{301}", "\u{200d}", "\u{200b}"] {
		let cluster = graphemes::<Utf8>(input.as_bytes()).next().unwrap();
		assert!(!cluster.is_control(), "{input:?} should not be a control");
	}

	let controls: Vec<_> = graphemes::<Utf8>(b"a\r\nb")
		.map(|cluster| cluster.is_control())
		.collect();
	assert_eq!(controls, [false, true, false]);
}
