use xutf::{Utf16, Utf32, graphemes_str, truncate, truncate_str, width_str};

const TOKENS: &[&str] = &[
	"a",
	"Z",
	"0",
	" ",
	"\t",
	"界",
	"語",
	"\u{301}",
	"\u{300}",
	"👨\u{200d}👩\u{200d}👧",
	"🇺🇸",
	"0\u{fe0f}\u{20e3}",
	"각",
	"क्",
	"ष",
];

const fn next_random(state: &mut u64) -> u64 {
	*state = state
		.wrapping_mul(6364136223846793005)
		.wrapping_add(1442695040888963407);
	*state
}

#[test]
fn invariant_fuzz() {
	let mut state = 0xd1b54a32d192ed03;
	for _ in 0..2_000 {
		let token_count = (next_random(&mut state) % 40) as usize;
		let mut input = String::new();
		for _ in 0..token_count {
			let index = (next_random(&mut state) % TOKENS.len() as u64) as usize;
			input.push_str(TOKENS[index]);
		}
		let max_width = (next_random(&mut state) % 41) as usize;
		let output = truncate_str(&input, max_width);

		assert!(input.starts_with(output));
		assert!(width_str(output) <= max_width);

		let mut boundary = 0;
		for cluster in graphemes_str(&input) {
			if boundary == output.len() {
				break;
			}
			boundary += cluster.len();
		}
		assert_eq!(boundary, output.len(), "cut was not a cluster boundary in {input:?}");

		if output.len() != input.len() {
			let excluded = graphemes_str(&input[output.len()..]).next().unwrap();
			assert!(width_str(output) + width_str(excluded) > max_width);
		}
		assert_eq!(truncate_str(&input, usize::MAX), input);
	}
}

#[test]
fn unit_cases() {
	assert_eq!(truncate_str("hello", 3), "hel");
	assert_eq!(truncate_str("ab👨\u{200d}👩\u{200d}👧", 3), "ab");
	assert_eq!(truncate_str("ab👨\u{200d}👩\u{200d}👧", 4), "ab👨\u{200d}👩\u{200d}👧");
	assert_eq!(truncate_str("a界", 2), "a");
	assert_eq!(truncate_str("a界", 3), "a界");
	assert_eq!(truncate_str("e\u{301}x", 1), "e\u{301}");
	assert_eq!(truncate_str("x0\u{fe0f}\u{20e3}", 2), "x");
	assert_eq!(truncate_str("a\r\nb", 1), "a\r\n");
	assert_eq!(truncate_str("\u{300}x", 0), "\u{300}");
	assert_eq!(truncate_str("xy", 0), "");

	let input = "exact 👨\u{200d}👩 fit";
	let output = truncate_str(input, width_str(input));
	assert!(core::ptr::eq(output.as_ptr(), input.as_ptr()));
	assert_eq!(output.len(), input.len());
}

fn utf32_to_string(units: &[u32]) -> String {
	units
		.iter()
		.map(|&cp| char::from_u32(cp).unwrap())
		.collect()
}

#[test]
fn cross_encoding() {
	let corpus = [
		"",
		"plain ASCII text",
		"a界語z",
		"e\u{301}x 0\u{fe0f}\u{20e3}",
		"👨\u{200d}👩\u{200d}👧 🇺🇸 각 क्‍ष",
		"a\r\nb\t界",
	];

	for input in corpus {
		let utf16: Vec<u16> = input.encode_utf16().collect();
		let utf16_foreign: Vec<u16> = utf16.iter().map(|unit| unit.swap_bytes()).collect();
		let utf32: Vec<u32> = input.chars().map(|ch| ch as u32).collect();
		let utf32_foreign: Vec<u32> = utf32.iter().map(|unit| unit.swap_bytes()).collect();

		for max_width in 0..=20 {
			let expected = truncate_str(input, max_width);

			let out16 = truncate::<Utf16<false>>(&utf16, max_width);
			assert_eq!(String::from_utf16(out16).unwrap(), expected);

			let out16_foreign = truncate::<Utf16<true>>(&utf16_foreign, max_width);
			let native16: Vec<u16> = out16_foreign.iter().map(|unit| unit.swap_bytes()).collect();
			assert_eq!(String::from_utf16(&native16).unwrap(), expected);

			let out32 = truncate::<Utf32<false>>(&utf32, max_width);
			assert_eq!(utf32_to_string(out32), expected);

			let out32_foreign = truncate::<Utf32<true>>(&utf32_foreign, max_width);
			let native32: Vec<u32> = out32_foreign.iter().map(|unit| unit.swap_bytes()).collect();
			assert_eq!(utf32_to_string(&native32), expected);
		}
	}
}

#[test]
fn simd_boundaries() {
	const EDGES: &[usize] = &[15, 16, 17, 63, 64, 65];
	for &length in EDGES {
		let input = "a".repeat(length);
		for &max_width in EDGES {
			let expected = max_width.min(length);
			assert_eq!(truncate_str(&input, max_width), &input[..expected]);
		}
	}

	let plain = "0".repeat(64);
	let input = format!("{plain}0\u{fe0f}\u{20e3}");
	assert_eq!(truncate_str(&input, 64), plain);
	assert_eq!(truncate_str(&input, 66), input);
}

#[test]
fn measured_width_matches_returned_prefix() {
	let family = "👨\u{200d}👩\u{200d}👧";
	let cases = [
		("ascii", 3, "asc"),
		("a界b", 2, "a"),
		(family, 2, family),
		("e\u{301}x", 1, "e\u{301}"),
		("", 4, ""),
		("abc", 0, ""),
		("a界", 3, "a界"),
		("a界", 20, "a界"),
	];

	for (input, max_width, expected) in cases {
		let (prefix, measured) = xutf::truncate_measured_str(input, max_width);
		assert_eq!(prefix, expected);
		assert_eq!(measured, width_str(prefix), "{input:?} at width {max_width}");
		assert_eq!(prefix, truncate_str(input, max_width));
	}
}

#[test]
fn skip_columns_unit_cases() {
	let input = "ab界z";
	let (prefix, prefix_width) = xutf::truncate_measured_str(input, 2);
	let (rest, dropped_width) = xutf::skip_columns_str(input, 2);
	assert_eq!(prefix_width, 2);
	assert_eq!(dropped_width, 2);
	assert_eq!(format!("{prefix}{rest}"), input);

	let wide = "界x";
	assert_eq!(truncate_str(wide, 1), "");
	assert_eq!(xutf::skip_columns_str(wide, 1), ("x", 2));

	let identity = "a\u{200b}b";
	let (rest, dropped_width) = xutf::skip_columns_str(identity, 0);
	assert!(core::ptr::eq(rest.as_ptr(), identity.as_ptr()));
	assert_eq!((rest, dropped_width), (identity, 0));

	assert_eq!(xutf::skip_columns_str(identity, 1), ("b", 1));
	assert_eq!(xutf::skip_columns_str("a界", 3), ("", 3));
	assert_eq!(xutf::skip_columns_str("a界", 99), ("", 3));
}

#[test]
fn skip_columns_utf16_parity() {
	let input = "ab界👨\u{200d}👩\u{200d}👧\u{200b}z";
	let utf16: Vec<u16> = input.encode_utf16().collect();

	for columns in 0..=8 {
		let (expected_tail, expected_width) = xutf::skip_columns_str(input, columns);
		let (tail, dropped_width) = xutf::skip_columns::<Utf16<false>>(&utf16, columns);
		assert_eq!(String::from_utf16(tail).unwrap(), expected_tail);
		assert_eq!(dropped_width, expected_width);
	}
}
