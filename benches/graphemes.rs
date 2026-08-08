//! Grapheme cluster iteration benchmark.
//!
//! Run default ratio bench: `cargo bench --bench graphemes`
//! Run Divan bench: `DIVAN=1 cargo bench --bench graphemes`

use divan::Bencher;
use rayon::prelude::*;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;
use xutf::{Utf8, Utf16, graphemes};

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1024 * 1024;

fn xutf_checksum_utf8(s: &str) -> usize {
	graphemes::<Utf8>(s.as_bytes()).fold(0usize, |sum, g| sum.wrapping_add(1 + g.width))
}

fn xutf_checksum_utf16(units: &[u16]) -> usize {
	graphemes::<Utf16<false>>(units).fold(0usize, |sum, g| sum.wrapping_add(1 + g.width))
}

fn useg_width_checksum(s: &str) -> usize {
	s.graphemes(true)
		.fold(0usize, |sum, g| sum.wrapping_add(1 + UnicodeWidthStr::width(g)))
}

fn run_ratio_table() {
	let inputs = [
		(
			"ascii",
			build_input(
				"The quick brown fox jumps over the lazy dog; terminal text stays responsive. ",
				TARGET,
			),
		),
		(
			"cjk",
			build_input("终端里的汉字排版很重要。日本語の文章と한국어 문장을混ぜた prose。", TARGET),
		),
		("emoji-soup", build_input("👨‍👩‍👧‍👦 🇺🇸 ⚠️ 👍🏽 👩‍❤️‍💋‍👨 0️⃣ 🚀 ", TARGET)),
		("mixed", build_input("Fast ASCII, naïve café, 界面文本, क्षत्रिय, 👨‍👩‍👧 and 🇦🇩. ", TARGET)),
	];

	let rows: Vec<_> = inputs
		.par_iter()
		.map(|(name, s)| {
			let utf16: Vec<u16> = s.encode_utf16().collect();
			(*name, vec![
				measure(s.len(), || xutf_checksum_utf8(s)),
				measure(s.len(), || s.graphemes(true).count()),
				measure(s.len(), || useg_width_checksum(s)),
				measure(s.len(), || xutf_checksum_utf16(&utf16)),
				measure(s.len(), || {
					let roundtrip: String = char::decode_utf16(utf16.iter().copied())
						.map(|ch| ch.unwrap_or(char::REPLACEMENT_CHARACTER))
						.collect();
					useg_width_checksum(&roundtrip)
				}),
			])
		})
		.collect();

	print_ratio_table(
		"Grapheme clusters",
		&["xutf", "unicode-seg", "useg+uwidth", "xutf utf16", "utf16 via String"],
		&rows,
	);
}

#[divan::bench_group]
mod ascii {
	use super::*;

	fn input() -> String {
		build_input(
			"The quick brown fox jumps over the lazy dog; terminal text stays responsive. ",
			TARGET,
		)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf8(&s));
	}

	#[divan::bench]
	fn unicode_seg(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.graphemes(true).count());
	}

	#[divan::bench]
	fn useg_uwidth(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| useg_width_checksum(&s));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf16(&utf16));
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let roundtrip: String = char::decode_utf16(utf16.iter().copied())
					.map(|ch| ch.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				useg_width_checksum(&roundtrip)
			});
	}
}

#[divan::bench_group]
mod cjk {
	use super::*;

	fn input() -> String {
		build_input("终端里的汉字排版很重要。日本語の文章と한국어 문장을混ぜた prose。", TARGET)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf8(&s));
	}

	#[divan::bench]
	fn unicode_seg(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.graphemes(true).count());
	}

	#[divan::bench]
	fn useg_uwidth(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| useg_width_checksum(&s));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf16(&utf16));
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let roundtrip: String = char::decode_utf16(utf16.iter().copied())
					.map(|ch| ch.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				useg_width_checksum(&roundtrip)
			});
	}
}

#[divan::bench_group]
mod emoji_soup {
	use super::*;

	fn input() -> String {
		build_input("👨‍👩‍👧‍👦 🇺🇸 ⚠️ 👍🏽 👩‍❤️‍💋‍👨 0️⃣ 🚀 ", TARGET)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf8(&s));
	}

	#[divan::bench]
	fn unicode_seg(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.graphemes(true).count());
	}

	#[divan::bench]
	fn useg_uwidth(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| useg_width_checksum(&s));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf16(&utf16));
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let roundtrip: String = char::decode_utf16(utf16.iter().copied())
					.map(|ch| ch.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				useg_width_checksum(&roundtrip)
			});
	}
}

#[divan::bench_group]
mod mixed {
	use super::*;

	fn input() -> String {
		build_input("Fast ASCII, naïve café, 界面文本, क्षत्रिय, 👨‍👩‍👧 and 🇦🇩. ", TARGET)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf8(&s));
	}

	#[divan::bench]
	fn unicode_seg(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.graphemes(true).count());
	}

	#[divan::bench]
	fn useg_uwidth(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| useg_width_checksum(&s));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| xutf_checksum_utf16(&utf16));
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let roundtrip: String = char::decode_utf16(utf16.iter().copied())
					.map(|ch| ch.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				useg_width_checksum(&roundtrip)
			});
	}
}
