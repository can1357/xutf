//! Visible width benchmark.
//!
//! Run default ratio bench: `cargo bench --bench width`
//! Run Divan bench: `DIVAN=1 cargo bench --bench width`

use divan::Bencher;
use rayon::prelude::*;
use unicode_width::UnicodeWidthStr;
use xutf::Text;

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1 << 20;

fn run_ratio_table() {
	let inputs = [
		("ascii", build_input("The quick brown fox jumps over 13 lazy dogs. ", TARGET)),
		("cjk", build_input("漢字界面 日本語 カタカナ 한글 ", TARGET)),
		("emoji", build_input("😀 ✅ 🚀 ⚠️ ℹ️ ❤️ 0️⃣ 👨‍👩‍👧 👋🏽 🇺🇸 ", TARGET)),
		("mixed", build_input("status: ✅ café 漢字 e\u{301} 🇯🇵 box ├─┤ ", TARGET)),
	];
	let utf16: Vec<Vec<u16>> = inputs
		.iter()
		.map(|(_, input)| input.encode_utf16().collect())
		.collect();

	let rows: Vec<_> = inputs
		.par_iter()
		.enumerate()
		.map(|(index, (name, input))| {
			let units = &utf16[index];
			(*name, vec![
				measure(input.len(), || input.visible_width()),
				measure(input.len(), || UnicodeWidthStr::width(input.as_str())),
				measure(input.len(), || units.visible_width()),
				measure(input.len(), || {
					let decoded: String = char::decode_utf16(units.iter().copied())
						.map(|result| result.unwrap_or(char::REPLACEMENT_CHARACTER))
						.collect();
					UnicodeWidthStr::width(decoded.as_str())
				}),
			])
		})
		.collect();

	print_ratio_table(
		"Visible width",
		&["xutf", "unicode-width", "xutf utf16", "utf16 via String"],
		&rows,
	);

	let bounded_inputs = [
		("ascii", inputs[0].1.as_str()),
		("cjk", inputs[1].1.as_str()),
		("emoji", inputs[2].1.as_str()),
		("mixed", inputs[3].1.as_str()),
		(
			"ascii 80-col (fits)",
			"The quick brown fox jumps over 13 lazy dogs. 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXY",
		),
	];

	let bounded_rows: Vec<_> = bounded_inputs
		.par_iter()
		.map(|(name, input)| {
			(*name, vec![
				measure(input.len(), || input.width_within(80).unwrap_or(usize::MAX)),
				measure(input.len(), || usize::from(input.visible_width() <= 80)),
				measure(input.len(), || usize::from(UnicodeWidthStr::width(*input) <= 80)),
			])
		})
		.collect();

	print_ratio_table(
		"Bounded width (fits in 80 cols?)",
		&["xutf width_within", "xutf width full", "unicode-width full"],
		&bounded_rows,
	);
}

#[divan::bench_group]
mod ascii {
	use super::*;

	fn input() -> String {
		build_input("The quick brown fox jumps over 13 lazy dogs. ", TARGET)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.visible_width());
	}

	#[divan::bench]
	fn unicode_width(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| UnicodeWidthStr::width(s.as_str()));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf16.visible_width());
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let decoded: String = char::decode_utf16(utf16.iter().copied())
					.map(|result| result.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				UnicodeWidthStr::width(decoded.as_str())
			});
	}
}

#[divan::bench_group]
mod cjk {
	use super::*;

	fn input() -> String {
		build_input("漢字界面 日本語 カタカナ 한글 ", TARGET)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.visible_width());
	}

	#[divan::bench]
	fn unicode_width(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| UnicodeWidthStr::width(s.as_str()));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf16.visible_width());
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let decoded: String = char::decode_utf16(utf16.iter().copied())
					.map(|result| result.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				UnicodeWidthStr::width(decoded.as_str())
			});
	}
}

#[divan::bench_group]
mod emoji {
	use super::*;

	fn input() -> String {
		build_input("😀 ✅ 🚀 ⚠️ ℹ️ ❤️ 0️⃣ 👨‍👩‍👧 👋🏽 🇺🇸 ", TARGET)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.visible_width());
	}

	#[divan::bench]
	fn unicode_width(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| UnicodeWidthStr::width(s.as_str()));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf16.visible_width());
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let decoded: String = char::decode_utf16(utf16.iter().copied())
					.map(|result| result.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				UnicodeWidthStr::width(decoded.as_str())
			});
	}
}

#[divan::bench_group]
mod mixed {
	use super::*;

	fn input() -> String {
		build_input("status: ✅ café 漢字 e\u{301} 🇯🇵 box ├─┤ ", TARGET)
	}

	#[divan::bench]
	fn xutf(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| s.visible_width());
	}

	#[divan::bench]
	fn unicode_width(bencher: Bencher) {
		let s = input();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| UnicodeWidthStr::width(s.as_str()));
	}

	#[divan::bench]
	fn xutf_utf16(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| utf16.visible_width());
	}

	#[divan::bench]
	fn utf16_via_string(bencher: Bencher) {
		let s = input();
		let utf16: Vec<u16> = s.encode_utf16().collect();
		bencher
			.counter(divan::counter::BytesCount::of_slice(s.as_bytes()))
			.bench_local(|| {
				let decoded: String = char::decode_utf16(utf16.iter().copied())
					.map(|result| result.unwrap_or(char::REPLACEMENT_CHARACTER))
					.collect();
				UnicodeWidthStr::width(decoded.as_str())
			});
	}
}
