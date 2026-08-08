use xutf::{
	Utf8, Utf16, Utf32, graphemes_str, width_str, wrap, wrap_measured, wrap_measured_str, wrap_str,
};

fn lines(input: &str, width: usize) -> Vec<&str> {
	wrap_str(input, width).collect()
}

#[test]
fn reference_behavior() {
	assert_eq!(lines("hello world", 5), ["hello", "world"]);
	assert_eq!(lines("foo barbazqux", 6), ["foo", "barbaz", "qux"]);
	assert_eq!(lines("   longword", 4), ["", "long", "word"]);
	assert_eq!(lines("a b c", 1), ["a", "b", "c"]);
	assert_eq!(lines("aa   bb", 2), ["aa", "bb"]);
	assert_eq!(lines("ab   ", 10), ["ab"]);
	assert_eq!(lines("", 10), [""]);
	assert_eq!(lines("a\n\nb", 10), ["a", "", "b"]);
	assert_eq!(lines("a\n", 10), ["a", ""]);
	assert_eq!(lines("\n", 10), ["", ""]);
	assert_eq!(lines("a\r\nb", 10), ["a", "b"]);
	assert_eq!(lines("a\rb", 10), ["a", "b"]);
	assert_eq!(lines("  hi", 10), ["  hi"]);
	assert_eq!(lines("a\n  b", 10), ["a", "  b"]);
	assert_eq!(lines("a\tb", 1), ["a\t", "b"]);
}

#[test]
fn clusters_are_never_split() {
	assert_eq!(lines("👨‍👩‍👧", 1), ["👨‍👩‍👧"]);
	assert_eq!(lines("界界界", 4), ["界界", "界"]);
	assert_eq!(lines("a\u{301}b", 1), ["a\u{301}", "b"]);
	assert_eq!(lines("🇺🇸🇨🇦", 2), ["🇺🇸", "🇨🇦"]);
}

#[test]
fn fuzzed_lines_obey_width_and_content_invariants() {
	const PIECES: &[&str] =
		&["a", "b", " ", " ", "\n", "\r\n", "界", "語", "😀", "👨‍👩‍👧", "e\u{301}", "🇺🇸"];
	let mut state = 0x8e6d_5a43_91c2_7b0du64;

	for case in 0..1_000 {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
		let piece_count = 1 + (state as usize % 80);
		let mut input = String::new();
		for _ in 0..piece_count {
			state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
			input.push_str(PIECES[(state as usize) % PIECES.len()]);
		}
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
		let max_width = state as usize % 25;
		let output: Vec<&str> = wrap_str(&input, max_width).collect();

		assert!(!output.is_empty(), "case {case}");
		for (index, line) in output.iter().enumerate() {
			assert!(
				width_str(line) <= max_width || graphemes_str(line).count() == 1,
				"case {case}, width {max_width}, line {line:?}"
			);
			assert!(!line.ends_with(' '), "case {case}, line {line:?}");
			if index > 0 && line.starts_with(' ') {
				assert!(
					input.contains("\n ") || input.contains("\r "),
					"only hard breaks may preserve leading spaces: case {case}, line {line:?}"
				);
			}
		}

		let source: String = input
			.chars()
			.filter(|c| !matches!(c, ' ' | '\n' | '\r'))
			.collect();
		let wrapped: String = output
			.iter()
			.flat_map(|line| line.chars())
			.filter(|c| *c != ' ')
			.collect();
		assert_eq!(wrapped, source, "case {case}");
	}
}

#[test]
fn zero_width_and_long_words_always_progress() {
	assert_eq!(lines("aaa", 0), ["a", "a", "a"]);

	let input = "a".repeat(10_000);
	let mut count = 0;
	for line in wrap_str(&input, 8) {
		assert_eq!(line.len(), 8);
		count += 1;
		assert!(count <= 1_250);
	}
	assert_eq!(count, 1_250);
}

fn assert_encoding_parity(input: &str, max_width: usize) {
	let expected: Vec<String> = wrap_str(input, max_width).map(str::to_owned).collect();

	let utf16: Vec<u16> = input.encode_utf16().collect();
	let native16: Vec<String> = wrap::<Utf16<false>>(&utf16, max_width)
		.map(|line| String::from_utf16(line).unwrap())
		.collect();
	assert_eq!(native16, expected);

	let utf32: Vec<u32> = input.chars().map(u32::from).collect();
	let native32: Vec<String> = wrap::<Utf32<false>>(&utf32, max_width)
		.map(|line| line.iter().map(|&cp| char::from_u32(cp).unwrap()).collect())
		.collect();
	assert_eq!(native32, expected);

	let foreign16: Vec<u16> = utf16.iter().map(|unit| unit.swap_bytes()).collect();
	let swapped16: Vec<String> = wrap::<Utf16<true>>(&foreign16, max_width)
		.map(|line| {
			let native: Vec<u16> = line.iter().map(|unit| unit.swap_bytes()).collect();
			String::from_utf16(&native).unwrap()
		})
		.collect();
	assert_eq!(swapped16, expected);
}

#[test]
fn encodings_have_identical_wrap_boundaries() {
	let cases = [
		("hello world", 5),
		("foo barbazqux", 6),
		("   longword", 4),
		("a b c", 1),
		("aa   bb", 2),
		("ab   ", 10),
		("", 10),
		("a\n\nb", 10),
		("a\n", 10),
		("\n", 10),
		("a\r\nb", 10),
		("a\rb", 10),
		("  hi", 10),
		("a\n  b", 10),
		("👨‍👩‍👧", 1),
		("界界界", 4),
		("a\u{301}b", 1),
		("🇺🇸🇨🇦", 2),
	];
	for (input, max_width) in cases {
		assert_encoding_parity(input, max_width);
	}
}

#[test]
fn malformed_sequences_are_never_split_by_space_trimming() {
	// 0x96 is a two-byte lead, so permissive decoding swallows the following
	// 0x20: it is a sequence tail, not a trailing space, and the cluster must
	// survive intact.
	let input = [0x86u8, 0x25, 0x96, 0x20, 0xec];
	let lines: Vec<&[u8]> = wrap::<Utf8>(&input, 0).collect();
	assert_eq!(lines, [&input[..4], &input[4..]]);
	assert_eq!(lines.concat(), input);
}

#[test]
fn simple_ascii_prose_matches_textwrap_shape() {
	// This comparison is deliberately narrow: textwrap has different word
	// separators and penalties, but coincides for single-space ASCII prose
	// with no overlong words.
	let text = "the quick brown fox jumps over the lazy dog";
	for width in [10, 20] {
		let actual: Vec<&str> = wrap_str(text, width).collect();
		let baseline = textwrap::wrap(text, width);
		let expected: Vec<&str> = baseline.iter().map(|line| line.as_ref()).collect();
		assert_eq!(actual, expected);
	}
}

#[test]
fn measured_lines_match_plain_wrap_and_report_source_positions() {
	let corpus = [
		"",
		"plain ascii paragraph with several words",
		"  leading and trailing spaces   ",
		"aa   bb      cc",
		"a\r\nb\rc\n\nd",
		"界 界界 長い単語",
		"👨‍👩‍👧 family 🇺🇸 flag",
		"supercalifragilisticexpialidocious",
		"\n",
		"a\n",
	];

	for input in corpus {
		for max_width in [0, 1, 4, 10, 80] {
			let measured: Vec<_> = wrap_measured_str(input, max_width).collect();
			let plain: Vec<_> = wrap_str(input, max_width).collect();
			assert_eq!(measured.len(), plain.len(), "{input:?}, width {max_width}");

			let mut previous_at = None;
			for (line, expected) in measured.iter().zip(&plain) {
				assert_eq!(line.as_str(), *expected, "{input:?}, width {max_width}");
				assert_eq!(
					&input[line.at..line.at + line.as_str().len()],
					line.as_str(),
					"{input:?}, width {max_width}",
				);
				assert_eq!(line.width, width_str(line.as_str()));
				if let Some(at) = previous_at {
					assert!(line.at > at, "{input:?}, width {max_width}");
				}
				previous_at = Some(line.at);
			}
		}
	}
}

#[test]
fn measured_utf16_offsets_are_code_unit_offsets() {
	let input: Vec<u16> = "a 😀 b\n界界 z".encode_utf16().collect();
	let lines: Vec<_> = wrap_measured::<Utf16<false>>(&input, 4).collect();
	let actual: Vec<_> = lines
		.iter()
		.map(|line| {
			assert_eq!(&input[line.at..line.at + line.units.len()], line.units);
			let text = String::from_utf16(line.units).unwrap();
			assert_eq!(line.width, width_str(&text));
			(line.at, line.width, text)
		})
		.collect();

	assert_eq!(actual, [
		(0, 4, "a 😀".to_owned()),
		(5, 1, "b".to_owned()),
		(7, 4, "界界".to_owned()),
		(10, 1, "z".to_owned()),
	],);
}
