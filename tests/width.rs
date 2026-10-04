use xutf::{
	Cluster, Utf8, Utf16, Utf32, simple_width, width, width_char, width_str, width_within,
	width_within_str,
};

const CORPUS: &[&str] = &[
	"plain printable ASCII 0123456789 !@#$%^&*()",
	"Cafe\u{301} nai\u{308}ve A\u{30a}",
	"Ελληνικά Кириллица",
	"漢字界面 日本語 カタカナ ひらがな",
	"ａｂｃ アイウ ｶﾀｶﾅ",
	"한글 각 값 가 각 ㄱㄴㅏㅣ \u{3164}",
	"नमस्ते हिन्दी தமிழ் ภาษาไทย",
	"┌─┬─┐│╳│└─┴─┘",
	"😀 🐈 🚀 ✅ 🌍",
	"⚠️ ℹ️ ❤️",
	"0️⃣ #️⃣",
	"👨‍👩‍👧 👩‍❤️‍👩 👨‍👨‍👦‍👦",
	"👋🏽 👍🏻 🧑🏿",
	"🇺🇸 🇯🇵 🇩🇪",
	"🏴\u{e0067}\u{e0062}\u{e0065}\u{e006e}\u{e0067}\u{e007f}",
];

/// kitty's cluster widths: the first codepoint's cells, with only the
/// variation selectors and Thai/Lao AM changing them.
#[test]
fn cluster_takes_its_first_codepoints_width() {
	for (s, expected, what) in [
		("e\u{301}", 1, "combining mark"),
		("\u{915}\u{93e}", 1, "spacing combining mark adds nothing"),
		("\u{915}\u{94d}\u{937}\u{93f}", 1, "Indic conjunct"),
		("\u{1100}\u{1161}\u{11a8}", 2, "conjoining Hangul syllable"),
		("\u{1161}", 1, "lone jungseong"),
		("a\u{302a}", 1, "wide mark after a narrow base"),
		("\u{1f1fa}\u{1f1f8}", 2, "flag"),
		("\u{1f1fa}", 2, "lone regional indicator"),
		("\u{1f1fa}\u{1f1f8}\u{1f1e9}", 4, "flag and a lone indicator"),
		("\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}", 2, "ZWJ sequence"),
		("\u{1f44d}\u{1f3fd}", 2, "modifier after its base"),
		("0\u{1f3fd}", 1, "modifier after a non-base"),
		("\u{261d}", 2, "text-default modifier base is wide"),
		("\u{600}1", 0, "prepended concatenation mark takes the digit's cell"),
		("\u{301}", 0, "lone mark"),
	] {
		assert_eq!(width_str(s), expected, "{what}: {s:?}");
	}
}

/// U+FE0F widens a one-cell emoji presentation base right before it, U+FE0E
/// narrows a two-cell one; neither touches anything else.
#[test]
fn variation_selectors_resize_presentation_bases() {
	for (s, expected, what) in [
		("\u{2764}\u{fe0f}", 2, "VS16 widens a text-default emoji"),
		("\u{2764}", 1, "text-default emoji"),
		("#\u{fe0f}\u{20e3}", 2, "keycap"),
		("#\u{20e3}", 1, "bare keycap"),
		("x\u{fe0f}", 1, "VS16 after a non-base"),
		("\u{2764}\u{301}\u{fe0f}", 1, "VS16 not right after the base"),
		("\u{1f600}\u{fe0f}", 2, "VS16 on a wide emoji"),
		("\u{1f600}\u{fe0e}", 1, "VS15 narrows a wide emoji"),
		("\u{261d}\u{fe0e}", 1, "VS15 narrows a wide modifier base"),
		("\u{1f600}\u{fe0e}\u{fe0f}", 1, "VS16 after VS15"),
		("\u{4e2d}\u{fe0e}", 2, "VS15 on a wide non-emoji"),
		("\u{2764}\u{fe0e}", 1, "VS15 on a narrow base"),
	] {
		assert_eq!(width_str(s), expected, "{what}: {s:?}");
	}
}

/// Simple characters take their standalone width and never join each other,
/// whatever their break class; everything that can join, pair or take no
/// cell is not simple.
#[test]
fn simple_characters_break_pairwise() {
	let simple = [
		'a',
		'9',
		'#',
		'\u{e9}',
		'\u{4e2d}',
		'\u{ac00}',
		'\u{ac01}',
		'\u{915}',
		'\u{1f600}',
		'\u{2764}',
		'\u{261d}',
		'\u{e000}',
		'\u{378}',
	];
	for a in simple {
		assert_eq!(simple_width(a), width_char(a), "{a:?}");
		assert_ne!(simple_width(a), 0, "{a:?}");
		for b in simple {
			assert!(!Cluster::new(a).push(b), "{a:?} {b:?}");
		}
	}
	for c in [
		'\u{301}',
		'\u{93e}',
		'\u{94d}',
		'\u{200d}',
		'\u{fe0f}',
		'\u{1f1e6}',
		'\u{1100}',
		'\u{1161}',
		'\u{11a8}',
		'\u{600}',
		'\u{d4e}',
		'\u{ad}',
		'\t',
		'\u{85}',
		'\u{fdd0}',
		'\u{1f3fb}',
	] {
		assert_eq!(simple_width(c), 0, "{c:?}");
	}
}

/// Thai and Lao AM, the spacing marks with a width of their own, widen a
/// one-cell base, through marks between them, and nothing wider.
#[test]
fn sara_am_widens_a_narrow_base() {
	for (s, expected) in [
		("\u{e01}\u{e33}", 2),
		("\u{e81}\u{eb3}", 2),
		("\u{e01}\u{e48}\u{e33}", 2),
		("\u{e33}", 1),
		("\u{4e2d}\u{e33}", 2),
		("\u{e01}\u{e33}\u{e33}", 2),
	] {
		assert_eq!(width_str(s), expected, "{s:?}");
	}
}

#[test]
fn all_encodings_measure_the_same_width() {
	for &s in CORPUS {
		let utf16: Vec<u16> = s.encode_utf16().collect();
		let utf32: Vec<u32> = s.chars().map(u32::from).collect();
		let utf16_foreign: Vec<u16> = utf16.iter().map(|u| u.swap_bytes()).collect();
		let utf32_foreign: Vec<u32> = utf32.iter().map(|u| u.swap_bytes()).collect();
		let expected = width::<Utf8>(s.as_bytes());
		assert_eq!(width::<Utf16<false>>(&utf16), expected, "{s:?}");
		assert_eq!(width::<Utf32<false>>(&utf32), expected, "{s:?}");
		assert_eq!(width::<Utf16<true>>(&utf16_foreign), expected, "{s:?}");
		assert_eq!(width::<Utf32<true>>(&utf32_foreign), expected, "{s:?}");
	}
}

#[test]
fn simd_boundaries_and_cluster_backoff() {
	const LENGTHS: &[usize] = &[1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129];
	for &k in LENGTHS {
		assert_eq!(width_str(&"x".repeat(k)), k);

		let keycap = "0".repeat(k) + "0\u{fe0f}\u{20e3}";
		assert_eq!(width_str(&keycap), k + 2, "keycap after {k} units");

		let promoted = "0".repeat(k) + "\u{fe0f}";
		assert_eq!(width_str(&promoted), k + 1, "VS16 after {k} units");

		let text_base = "x".repeat(k) + "\u{fe0f}\u{20e3}";
		assert_eq!(width_str(&text_base), k, "non-keycap base after {k} units");

		let bare_keycap = "0".repeat(k) + "\u{20e3}";
		assert_eq!(width_str(&bare_keycap), k, "bare keycap after {k} units");

		let sara_am = "x".repeat(k) + "\u{301}\u{e33}";
		assert_eq!(width_str(&sara_am), k + 1, "SARA AM after a mark after {k} units");

		let tab = "x".repeat(k) + "\t";
		let bell = "x".repeat(k) + "\x07";
		assert_eq!(width_str(&tab), k);
		assert_eq!(width_str(&bell), k);
	}
}

#[test]
fn tui_reference_widths() {
	for (s, expected) in [
		("⚠️", 2),
		("⚠", 1),
		("✅", 2),
		("0️⃣", 2),
		("a\tb", 2),
		("界", 2),
		("\u{3164}", 2),
		("e\u{301}", 1),
		("🇺🇸", 2),
		("👨‍👩‍👧", 2),
		("", 0),
	] {
		assert_eq!(width_str(s), expected, "{s:?}");
	}
}

#[test]
fn bounded_width_matches_unbounded_measurement() {
	for (s, budgets) in [
		("", &[0, 1][..]),
		("ascii", &[0, 4, 5, 6]),
		("a somewhat longer printable ASCII line", &[0, 8, 20, 38, 39]),
		("漢字a", &[0, 1, 2, 4, 5]),
		("👨‍👩‍👧 and text", &[0, 1, 2, 5, 10]),
		("ab\u{301}", &[0, 1, 2]),
	] {
		let expected = width_str(s);
		for &max_width in budgets {
			let bounded = width_within_str(s, max_width);
			assert_eq!(bounded, (expected <= max_width).then_some(expected), "{s:?} at {max_width}");
			assert_eq!(width_within::<Utf8>(s.as_bytes(), max_width), bounded, "{s:?} at {max_width}");
		}
	}

	let long = "x".repeat(1024 * 1024);
	assert_eq!(width_within_str(&long, 80), None);
}

#[test]
fn bounded_width_includes_zero_width_tail_at_edge() {
	assert_eq!(width_within_str("", 0), Some(0));
	assert_eq!(width_within_str("a", 0), None);
	assert_eq!(width_within_str("ab\u{301}", 2), Some(2));
	assert_eq!(width_within_str("ab\u{301}", 1), None);

	let exact = "界e\u{301}😀";
	let exact_width = width_str(exact);
	assert_eq!(width_within_str(exact, exact_width), Some(exact_width));
}

#[test]
fn bounded_width_matches_utf16() {
	let sample = "ASCII 漢字 e\u{301} 👨‍👩‍👧";
	let utf16: Vec<u16> = sample.encode_utf16().collect();
	for max_width in 0..=width_str(sample) + 1 {
		assert_eq!(
			width_within::<Utf16<false>>(&utf16, max_width),
			width_within_str(sample, max_width),
			"budget {max_width}"
		);
	}
}

#[test]
fn standalone_character_widths() {
	for (c, expected) in [
		('a', 1),
		('\t', 0),
		('\n', 0),
		('\r', 0),
		('中', 2),
		('\u{301}', 0),
		('\u{2764}', 1),
		('\u{1f600}', 2),
		('\u{200d}', 0),
	] {
		assert_eq!(width_char(c), expected, "{c:?}");
	}
}
