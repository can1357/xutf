use unicode_width::UnicodeWidthStr;
use xutf::{Utf8, Utf16, Utf32, width, width_str};

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

fn reference(s: &str) -> usize {
	UnicodeWidthStr::width(s)
}

#[test]
fn safe_corpus_matches_unicode_width() {
	for &s in CORPUS {
		assert_eq!(width_str(s), reference(s), "{s:?}");
	}

	let mut state = 0x8e5d_a2c7_13b9_4f61u64;
	for _ in 0..500 {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
		let count = 2 + (state as usize % 7);
		let mut s = String::new();
		for _ in 0..count {
			state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
			s.push_str(CORPUS[state as usize % CORPUS.len()]);
			s.push(' ');
		}
		assert_eq!(width_str(&s), reference(&s), "{s:?}");
	}
}

#[test]
fn deterministic_safe_scalar_fuzz_matches_unicode_width() {
	const SAFE: &[char] = &[
		' ', '!', '0', 'A', 'z', 'é', 'Ω', 'Ж', '\u{301}', '\u{308}', 'क', 'म', '\u{93f}', '\u{94d}',
		'த', '\u{bbf}', 'ก', '\u{e34}', '界', '語', 'あ', 'カ', 'ｶ', '한', 'ᄀ', 'ᅡ', 'ᆨ', 'ㄱ',
		'\u{3164}', '─', '│', '╳', '😀', '✅', '🚀', '👋', '🏽', '🇺', '🇸',
	];
	let mut state = 0xd1b5_4a32_d192_ed03u64;
	for _ in 0..2_000 {
		state = state
			.wrapping_mul(2862933555777941757)
			.wrapping_add(3037000493);
		let len = state as usize % 40;
		let mut s = String::new();
		for _ in 0..len {
			state = state
				.wrapping_mul(2862933555777941757)
				.wrapping_add(3037000493);
			s.push(SAFE[state as usize % SAFE.len()]);
		}
		assert_eq!(width_str(&s), reference(&s), "{s:?}");
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
		("\u{3164}", 0),
		("e\u{301}", 1),
		("🇺🇸", 2),
		("👨‍👩‍👧", 2),
		("", 0),
	] {
		assert_eq!(width_str(s), expected, "{s:?}");
	}
}
