//! Codepoint-aligned offset benchmark: [`Text::unit_offset`] against the
//! scalar `std` walks it replaces, seeking to the middle of each input.
//!
//! Run: `cargo bench --bench offset`

use std::hint::black_box;

use rayon::prelude::*;
use xutf::Text as _;

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

/// A rope leaf and a large buffer.
const SIZES: [(&str, usize); 2] = [("2K", 2 * 1024), ("1M", 1024 * 1024)];

const SEEDS: [(&str, &str); 6] = [
	("ascii", "The quick brown fox jumps over the lazy dog. "),
	(
		"code",
		"let caf\u{e9} = format!(\"{x:>8} \u{2192} {y}\"); // na\u{ef}ve\n\tif ok { return; }\n",
	),
	("cyrillic", "Съешь же ещё этих мягких французских булок, да выпей чаю. "),
	("cjk", "空間認識能力の育成には、幼少期の遊びが重要である。"),
	("emoji", "status 👍 ready 🚀 done 🦀 "),
	("emoji only", "😀😃😄😁😆😅🤣😂"),
];

/// `str::char_indices().nth(n)` as a byte offset.
fn std_char(s: &str, n: usize) -> usize {
	s.char_indices().nth(n).map_or(s.len(), |(at, _)| at)
}

/// The byte offset of the char holding UTF-16 unit `n`.
fn std_utf16(s: &str, n: usize) -> usize {
	let mut units = 0;
	for (at, c) in s.char_indices() {
		units += c.len_utf16();
		if units > n {
			return at;
		}
	}
	s.len()
}

fn run_ratio_table() {
	let cases: Vec<(String, String)> = SIZES
		.iter()
		.flat_map(|&(size, bytes)| {
			SEEDS.iter().map(move |&(name, seed)| {
				let mut s = build_input(seed, bytes);
				let mut end = bytes.min(s.len());
				while !s.is_char_boundary(end) {
					end -= 1;
				}
				s.truncate(end);
				(format!("{name}/{size}"), s)
			})
		})
		.collect();

	let chars: Vec<_> = cases
		.par_iter()
		.map(|(label, s)| {
			let n = s.chars().count() / 2;
			let bytes = std_char(s, n);
			assert_eq!(s.unit_offset::<u32>(n), bytes);
			let row = vec![
				measure(bytes, || black_box(s.as_str()).unit_offset::<u32>(black_box(n))),
				measure(bytes, || std_char(black_box(s), black_box(n))),
			];
			(label.as_str(), row)
		})
		.collect();
	print_ratio_table("Char index to byte offset", &["xutf", "std nth"], &chars);

	let utf16: Vec<_> = cases
		.par_iter()
		.map(|(label, s)| {
			let n = s.encode_utf16().count() / 2;
			let bytes = std_utf16(s, n);
			assert_eq!(s.unit_offset::<u16>(n), bytes);
			let row = vec![
				measure(bytes, || black_box(s.as_str()).unit_offset::<u16>(black_box(n))),
				measure(bytes, || std_utf16(black_box(s), black_box(n))),
			];
			(label.as_str(), row)
		})
		.collect();
	print_ratio_table("UTF-16 index to byte offset", &["xutf", "std walk"], &utf16);
}
