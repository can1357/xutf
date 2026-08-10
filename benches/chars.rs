//! `BufRead` char-streaming benchmark against the `utf8-chars` crate.
//!
//! Run: `cargo bench --bench chars`

use std::io::BufReader;

use rayon::prelude::*;
use xutf::{BufReadCharsExt as _, Text as _, Utf16Le, Utf32Le};

mod common;
use common::{build_input, measure_with_setup, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1024 * 1024;

fn inputs() -> Vec<(&'static str, Vec<u8>)> {
	let ascii = build_input("The quick brown fox jumps over the lazy dog. ", TARGET);
	let cyrillic = build_input("Съешь же ещё этих мягких французских булок, да выпей чаю. ", TARGET);
	let mixed = build_input("café Ωμέγα → 世界 🦀 emoji, ascii tail; Straße ", TARGET);
	let cjk = build_input("空間認識能力の育成には、幼少期の遊びが重要である。", TARGET);
	// Worst case: random bytes are mostly invalid sequences, forcing the
	// exact error path on nearly every iteration for both crates.
	let mut seed = 0x243f_6a88_85a3_08d3u64;
	let random: Vec<u8> = std::iter::repeat_with(|| {
		seed = seed
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		(seed >> 33) as u8
	})
	.take(TARGET)
	.collect();
	vec![
		("ascii", ascii.into_bytes()),
		("cyrillic 2-byte", cyrillic.into_bytes()),
		("mixed 1-4 byte", mixed.into_bytes()),
		("cjk 3-byte", cjk.into_bytes()),
		("random bytes", random),
	]
}

fn count_xutf(input: &[u8]) -> usize {
	let mut reader = BufReader::new(input);
	reader.chars_raw().flatten().count()
}

fn count_utf8_chars(input: &[u8]) -> usize {
	let mut reader = BufReader::new(input);
	utf8_chars::BufReadCharsExt::chars_raw(&mut reader)
		.flatten()
		.count()
}

fn run_ratio_table() {
	let inputs = inputs();
	let rows: Vec<_> = inputs
		.par_iter()
		.map(|(name, input)| {
			let ours = measure_with_setup(input.len(), || input.as_slice(), |b| count_xutf(b));
			let theirs = measure_with_setup(input.len(), || input.as_slice(), |b| count_utf8_chars(b));
			(*name, vec![ours, theirs])
		})
		.collect();
	print_ratio_table("BufRead chars (UTF-8, BufReader 8 KiB)", &["xutf", "utf8-chars"], &rows);

	// Cross-encoding stream decode, same text per row, ratio vs our UTF-8.
	let text_rows: Vec<_> = inputs[..4]
		.par_iter()
		.map(|(name, input)| {
			let text = core::str::from_utf8(input).unwrap();
			let utf16: Vec<u16> = text.transcode();
			let utf32: Vec<u32> = text.transcode();
			let utf16_bytes: Vec<u8> = utf16.iter().flat_map(|u| u.to_le_bytes()).collect();
			let utf32_bytes: Vec<u8> = utf32.iter().flat_map(|u| u.to_le_bytes()).collect();
			let utf8 = measure_with_setup(input.len(), || input.as_slice(), |b| count_xutf(b));
			let utf16 = measure_with_setup(
				utf16_bytes.len(),
				|| utf16_bytes.as_slice(),
				|b| {
					let mut reader = BufReader::new(*b);
					reader.decode_chars_raw::<Utf16Le>().flatten().count()
				},
			);
			let utf32 = measure_with_setup(
				utf32_bytes.len(),
				|| utf32_bytes.as_slice(),
				|b| {
					let mut reader = BufReader::new(*b);
					reader.decode_chars_raw::<Utf32Le>().flatten().count()
				},
			);
			(*name, vec![utf8, utf16, utf32])
		})
		.collect();
	print_ratio_table(
		"BufRead chars by encoding (xutf only)",
		&["utf-8", "utf-16le", "utf-32le"],
		&text_rows,
	);
}
