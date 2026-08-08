//! Width truncation benchmark.
//!
//! Run default ratio bench: `cargo bench --bench truncate`
//! Run Divan bench: `DIVAN=1 cargo bench --bench truncate`

use divan::Bencher;
use rayon::prelude::*;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;
use xutf::Text;

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1024 * 1024;

fn reference(input: &str, max_width: usize) -> &str {
	let mut width = 0;
	let mut end = 0;
	for cluster in UnicodeSegmentation::graphemes(input, true) {
		let cluster_width = UnicodeWidthStr::width(cluster);
		if width + cluster_width > max_width {
			break;
		}
		width += cluster_width;
		end += cluster.len();
	}
	&input[..end]
}

fn reference_alloc(input: &str, max_width: usize) -> String {
	let mut width = 0;
	let mut output = String::new();
	for cluster in UnicodeSegmentation::graphemes(input, true) {
		let cluster_width = UnicodeWidthStr::width(cluster);
		if width + cluster_width > max_width {
			break;
		}
		width += cluster_width;
		output.push_str(cluster);
	}
	output
}

/// Reference implementation for skipping columns using unicode-segmentation
/// (borrowed remainder).
fn skip_reference(input: &str, limit: usize) -> &str {
	let mut width = 0;
	for (idx, cluster) in UnicodeSegmentation::grapheme_indices(input, true) {
		if width >= limit {
			return &input[idx..];
		}
		width += UnicodeWidthStr::width(cluster);
	}
	&input[input.len()..]
}

/// Reference implementation for skipping columns using unicode-segmentation
/// (owned String remainder).
fn skip_reference_alloc(input: &str, limit: usize) -> String {
	let mut width = 0;
	let mut output = String::new();
	let mut skipping = true;
	for cluster in UnicodeSegmentation::graphemes(input, true) {
		if skipping {
			if width >= limit {
				skipping = false;
				output.push_str(cluster);
			} else {
				width += UnicodeWidthStr::width(cluster);
				if width >= limit {
					skipping = false;
				}
			}
		} else {
			output.push_str(cluster);
		}
	}
	output
}

/// Measures width truncation throughput for all contenders.
fn measurements(input: &str, max_width: usize) -> Vec<f64> {
	vec![
		measure(input.len(), || input.truncate_width(max_width).len()),
		measure(input.len(), || {
			let (rest, width) = input.truncate_measured(max_width);
			rest.len() + width
		}),
		measure(input.len(), || reference(input, max_width).len()),
		measure(input.len(), || reference_alloc(input, max_width).len()),
	]
}

/// Measures column skipping throughput for all contenders.
fn skip_measurements(input: &str, limit: usize) -> Vec<f64> {
	vec![
		measure(input.len(), || {
			let (rest, width) = input.skip_columns(limit);
			rest.len() + width
		}),
		measure(input.len(), || skip_reference(input, limit).len()),
		measure(input.len(), || skip_reference_alloc(input, limit).len()),
	]
}

fn run_ratio_table() {
	let ascii = build_input("The quick brown fox jumps over the lazy dog; 0123456789. ", TARGET);
	let cjk = build_input("日本語の端末表示幅を測る。中文文本測試。한국어 텍스트. ", TARGET);
	let emoji = build_input("status 👨‍👩‍👧 ready 🚀 e\u{301} flags 🇺🇸 key 0\u{fe0f}\u{20e3} ", TARGET);

	let ascii_limits = [80, 256, ascii.visible_width() / 2];
	let cjk_limits = [80, 256, cjk.visible_width() / 2];

	let cases = [
		("ascii/80", &ascii, ascii_limits[0]),
		("ascii/256", &ascii, ascii_limits[1]),
		("ascii/half", &ascii, ascii_limits[2]),
		("cjk/80", &cjk, cjk_limits[0]),
		("cjk/256", &cjk, cjk_limits[1]),
		("cjk/half", &cjk, cjk_limits[2]),
		("emoji/80", &emoji, 80),
		("emoji/256", &emoji, 256),
		("emoji/half", &emoji, emoji.visible_width() / 2),
	];

	let rows: Vec<_> = cases
		.par_iter()
		.map(|(label, input, limit)| (*label, measurements(input, *limit)))
		.collect();
	print_ratio_table(
		"Width truncation",
		&["xutf", "xutf measured", "useg+uwidth", "useg+uwidth alloc"],
		&rows,
	);

	let skip_rows: Vec<_> = cases
		.par_iter()
		.map(|(label, input, limit)| (*label, skip_measurements(input, *limit)))
		.collect();
	print_ratio_table(
		"Skip columns",
		&["xutf skip", "useg+uwidth", "useg+uwidth alloc"],
		&skip_rows,
	);
}

fn get_input_and_limit(case: &str) -> (String, usize) {
	let ascii = build_input("The quick brown fox jumps over the lazy dog; 0123456789. ", TARGET);
	let cjk = build_input("日本語の端末表示幅を測る。中文文本測試。한국어 텍스트. ", TARGET);
	let emoji = build_input("status 👨‍👩‍👧 ready 🚀 e\u{301} flags 🇺🇸 key 0\u{fe0f}\u{20e3} ", TARGET);

	match case {
		"ascii/80" => (ascii, 80),
		"ascii/256" => (ascii, 256),
		"ascii/half" => {
			let w = ascii.visible_width() / 2;
			(ascii, w)
		},
		"cjk/80" => (cjk, 80),
		"cjk/256" => (cjk, 256),
		"cjk/half" => {
			let w = cjk.visible_width() / 2;
			(cjk, w)
		},
		"emoji/80" => (emoji, 80),
		"emoji/256" => (emoji, 256),
		"emoji/half" => {
			let w = emoji.visible_width() / 2;
			(emoji, w)
		},
		_ => unreachable!(),
	}
}

#[divan::bench_group]
mod ascii_80 {
	use super::*;

	fn input() -> (String, usize) {
		get_input_and_limit("ascii/80")
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.truncate_width(limit).len());
	}

	#[divan::bench]
	fn useg_uwidth(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| reference(&s, limit).len());
	}

	#[divan::bench]
	fn useg_uwidth_alloc(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| reference_alloc(&s, limit).len());
	}
}

#[divan::bench_group]
mod cjk_80 {
	use super::*;

	fn input() -> (String, usize) {
		get_input_and_limit("cjk/80")
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.truncate_width(limit).len());
	}

	#[divan::bench]
	fn useg_uwidth(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| reference(&s, limit).len());
	}

	#[divan::bench]
	fn useg_uwidth_alloc(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| reference_alloc(&s, limit).len());
	}
}

#[divan::bench_group]
mod emoji_80 {
	use super::*;

	fn input() -> (String, usize) {
		get_input_and_limit("emoji/80")
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.truncate_width(limit).len());
	}

	#[divan::bench]
	fn useg_uwidth(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| reference(&s, limit).len());
	}

	#[divan::bench]
	fn useg_uwidth_alloc(bencher: Bencher) {
		let (s, limit) = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| reference_alloc(&s, limit).len());
	}
}
