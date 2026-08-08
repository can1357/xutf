//! Word wrap benchmark.
//!
//! Run default ratio bench: `cargo bench --bench wrap`
//! Run Divan bench: `DIVAN=1 cargo bench --bench wrap`

use divan::Bencher;
use rayon::prelude::*;
use xutf::Text;

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1024 * 1024;

fn utf8_checksum(input: &str, width: usize) -> usize {
	let (lines, bytes) = input
		.wrap(width)
		.fold((0usize, 0usize), |(count, bytes), line| (count + 1, bytes + line.len()));
	lines.wrapping_add(bytes)
}

/// Sums byte offset, display width, and byte length for each line produced by
/// `wrap_measured`.
fn utf8_measured_checksum(input: &str, width: usize) -> usize {
	input
		.wrap_measured(width)
		.fold(0usize, |acc, line| acc.wrapping_add(line.at + line.width + line.as_str().len()))
}

/// Sums display width computed via `visible_width` for each line produced by
/// `wrap`.
fn utf8_remeasure_checksum(input: &str, width: usize) -> usize {
	input
		.wrap(width)
		.fold(0usize, |acc, line| acc.wrapping_add(line.visible_width()))
}

fn utf16_checksum(input: &[u16], width: usize) -> usize {
	let (lines, bytes) = input
		.wrap(width)
		.fold((0usize, 0usize), |(count, bytes), line| (count + 1, bytes + line.len() * 2));
	lines.wrapping_add(bytes)
}
fn run_ratio_table() {
	let inputs = [
		("ascii", build_input("The quick brown fox jumps over the lazy dog. ", TARGET)),
		("cjk", build_input("日本語中文文本測試韓國語文章段落。", TARGET)),
		(
			"mixed-emoji",
			build_input(
				"Deploy 🚀 the family 👨‍👩‍👧 dashboard with alerts 🔔 and status ✅ now. ",
				TARGET,
			),
		),
	];

	for width in [40, 80] {
		let rows: Vec<_> = inputs
			.par_iter()
			.map(|(name, input)| {
				let utf16: Vec<u16> = input.encode_utf16().collect();
				(*name, vec![
					measure(input.len(), || utf8_checksum(input, width)),
					measure(input.len(), || utf8_measured_checksum(input, width)),
					measure(input.len(), || utf8_remeasure_checksum(input, width)),
					measure(input.len(), || {
						let wrapped = textwrap::wrap(input.as_str(), width);
						let (lines, bytes) = wrapped
							.iter()
							.fold((0usize, 0usize), |(count, bytes), line| {
								(count + 1, bytes + line.len())
							});
						lines.wrapping_add(bytes)
					}),
					measure(utf16.len() * 2, || utf16_checksum(&utf16, width)),
				])
			})
			.collect();
		print_ratio_table(
			&format!("word wrap at {width} columns"),
			&["xutf utf8", "xutf measured", "xutf + remeasure", "textwrap", "xutf utf16"],
			&rows,
		);
	}
}

#[divan::bench_group]
mod ascii_80 {
	use super::*;

	fn input() -> String {
		build_input("The quick brown fox jumps over the lazy dog. ", TARGET)
	}

	#[divan::bench]
	fn xutf_utf8(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf8_checksum(&s, 80));
	}

	#[divan::bench]
	fn xutf_measured(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf8_measured_checksum(&s, 80));
	}

	#[divan::bench]
	fn xutf_remeasure(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf8_remeasure_checksum(&s, 80));
	}

	#[divan::bench]
	fn textwrap_lib(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let wrapped = textwrap::wrap(s.as_str(), 80);
				let (lines, bytes) = wrapped
					.iter()
					.fold((0usize, 0usize), |(count, bytes), line| (count + 1, bytes + line.len()));
				lines.wrapping_add(bytes)
			});
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf16_checksum(&utf16, 80));
	}
}

#[divan::bench_group]
mod cjk_80 {
	use super::*;

	fn input() -> String {
		build_input("日本語中文文本測試韓國語文章段落。", TARGET)
	}

	#[divan::bench]
	fn xutf_utf8(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf8_checksum(&s, 80));
	}

	#[divan::bench]
	fn xutf_measured(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf8_measured_checksum(&s, 80));
	}

	#[divan::bench]
	fn xutf_remeasure(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf8_remeasure_checksum(&s, 80));
	}

	#[divan::bench]
	fn textwrap_lib(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let wrapped = textwrap::wrap(s.as_str(), 80);
				let (lines, bytes) = wrapped
					.iter()
					.fold((0usize, 0usize), |(count, bytes), line| (count + 1, bytes + line.len()));
				lines.wrapping_add(bytes)
			});
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf16_checksum(&utf16, 80));
	}
}
