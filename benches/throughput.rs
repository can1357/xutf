//! Throughput comparison: xutf vs simdutf (Lemire, C++ via FFI),
//! `encoding_rs::mem` (Firefox), and std char-iterator baselines.
//!
//! Run default ratio bench: `cargo bench --bench throughput`
//! Run Divan bench: `DIVAN=1 cargo bench --bench throughput`

use divan::Bencher;
use rayon::prelude::*;
use xutf::{AsciiCase, Utf8, Utf16, Utf32, transcode_into};

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 256 * 1024;

fn get_inputs() -> Vec<(&'static str, String)> {
	let mut inputs = INPUT_NAMES
		.iter()
		.map(|&name| (name, build_input(input_seed(name), TARGET)))
		.collect::<Vec<_>>();
	if let Ok(filter) = std::env::var("XUTF_BENCH_INPUT") {
		inputs.retain(|(name, _)| name.contains(&filter));
	}
	inputs
}

fn utf8_to_utf16_rows(inputs: &[(&'static str, String)]) -> Vec<(&'static str, Vec<f64>)> {
	inputs
		.par_iter()
		.map(|(name, s)| {
			let src = s.as_bytes();
			let len = src.len();
			let impls: Vec<Box<dyn FnOnce() -> f64 + Send>> = vec![
				Box::new(|| {
					let mut dst = vec![0u16; len + 16];
					measure(len, || transcode_into::<Utf8, Utf16>(src, &mut dst, AsciiCase::Preserve).1)
				}),
				Box::new(|| {
					let mut dst = vec![0u16; len + 16];
					measure(len, || {
						// SAFETY: `src` is valid UTF-8 and `dst` has at least one u16 per input byte.
						unsafe {
							simdutf::convert_utf8_to_utf16(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u16; len + 16];
					measure(len, || {
						// SAFETY: `src` is valid UTF-8 and `dst` has at least one u16 per input byte.
						unsafe {
							simdutf::convert_valid_utf8_to_utf16(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u16; len + 16];
					measure(len, || encoding_rs::mem::convert_utf8_to_utf16(src, &mut dst))
				}),
				Box::new(|| {
					let mut dst = vec![0u16; len + 16];
					measure(len, || {
						let mut n = 0;
						for u in s.encode_utf16() {
							dst[n] = u;
							n += 1;
						}
						n
					})
				}),
			];
			let results = impls.into_iter().map(|f| f()).collect();
			(*name, results)
		})
		.collect()
}

fn utf16_to_utf8_rows(inputs: &[(&'static str, String)]) -> Vec<(&'static str, Vec<f64>)> {
	inputs
		.par_iter()
		.map(|(name, s)| {
			let src: Vec<u16> = s.encode_utf16().collect();
			let src_bytes = src.len() * 2;
			let len = src.len();
			let impls: Vec<Box<dyn FnOnce() -> f64 + Send>> = vec![
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						transcode_into::<Utf16, Utf8>(&src, &mut dst, AsciiCase::Preserve).1
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						// SAFETY: `src` is valid UTF-16 and `dst` reserves four bytes per code unit.
						unsafe {
							simdutf::convert_utf16_to_utf8(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						// SAFETY: `src` is valid UTF-16 and `dst` reserves four bytes per code unit.
						unsafe {
							simdutf::convert_valid_utf16_to_utf8(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || encoding_rs::mem::convert_utf16_to_utf8(&src, &mut dst))
				}),
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						let mut n = 0;
						for r in char::decode_utf16(src.iter().copied()) {
							n += r.unwrap().encode_utf8(&mut dst[n..]).len();
						}
						n
					})
				}),
			];
			let results = impls.into_iter().map(|f| f()).collect();
			(*name, results)
		})
		.collect()
}

fn utf8_to_utf32_rows(inputs: &[(&'static str, String)]) -> Vec<(&'static str, Vec<f64>)> {
	inputs
		.par_iter()
		.map(|(name, s)| {
			let src = s.as_bytes();
			let len = src.len();
			let impls: Vec<Box<dyn FnOnce() -> f64 + Send>> = vec![
				Box::new(|| {
					let mut dst = vec![0u32; len + 16];
					measure(len, || transcode_into::<Utf8, Utf32>(src, &mut dst, AsciiCase::Preserve).1)
				}),
				Box::new(|| {
					let mut dst = vec![0u32; len + 16];
					measure(len, || {
						// SAFETY: `src` is valid UTF-8 and `dst` has at least one u32 per input byte.
						unsafe {
							simdutf::convert_utf8_to_utf32(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u32; len + 16];
					measure(len, || {
						// SAFETY: `src` is valid UTF-8 and `dst` has at least one u32 per input byte.
						unsafe {
							simdutf::convert_valid_utf8_to_utf32(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u32; len + 16];
					measure(len, || {
						let mut n = 0;
						for c in s.chars() {
							dst[n] = c as u32;
							n += 1;
						}
						n
					})
				}),
			];
			let results = impls.into_iter().map(|f| f()).collect();
			(*name, results)
		})
		.collect()
}

fn utf32_to_utf8_rows(inputs: &[(&'static str, String)]) -> Vec<(&'static str, Vec<f64>)> {
	inputs
		.par_iter()
		.map(|(name, s)| {
			let src: Vec<u32> = s.chars().map(|c| c as u32).collect();
			let src_bytes = src.len() * 4;
			let len = src.len();
			let impls: Vec<Box<dyn FnOnce() -> f64 + Send>> = vec![
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						transcode_into::<Utf32, Utf8>(&src, &mut dst, AsciiCase::Preserve).1
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						// SAFETY: `src` contains Unicode scalar values and `dst` reserves four bytes per value.
						unsafe {
							simdutf::convert_utf32_to_utf8(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						// SAFETY: `src` contains Unicode scalar values and `dst` reserves four bytes per value.
						unsafe {
							simdutf::convert_valid_utf32_to_utf8(src.as_ptr(), len, dst.as_mut_ptr())
						}
					})
				}),
				Box::new(|| {
					let mut dst = vec![0u8; len * 4 + 16];
					measure(src_bytes, || {
						let mut n = 0;
						for &cp in &src {
							n += char::from_u32(cp).unwrap().encode_utf8(&mut dst[n..]).len();
						}
						n
					})
				}),
			];
			let results = impls.into_iter().map(|f| f()).collect();
			(*name, results)
		})
		.collect()
}

fn equality_rows(inputs: &[(&'static str, String)]) -> Vec<(&'static str, Vec<f64>)> {
	inputs
		.par_iter()
		.map(|(name, s)| {
			let a = s.as_bytes();
			let b = a.to_vec();
			let b16: Vec<u16> = s.encode_utf16().collect();
			let len = a.len();
			let impls: Vec<Box<dyn FnOnce() -> f64 + Send>> = vec![
				Box::new(|| measure(len, || xutf::equals::<Utf8, Utf8>(a, &b) as usize)),
				Box::new(|| measure(len, || (a == &b[..]) as usize)),
				Box::new(|| {
					measure(len, || xutf::equals_ignore_ascii_case::<Utf8, Utf8>(a, &b) as usize)
				}),
				Box::new(|| measure(len, || a.eq_ignore_ascii_case(&b) as usize)),
				Box::new(|| measure(len, || xutf::equals::<Utf8, Utf16>(a, &b16) as usize)),
				Box::new(|| {
					let mut scratch = vec![0u16; len + 16];
					measure(len, || {
						let n = encoding_rs::mem::convert_utf8_to_utf16(a, &mut scratch);
						(scratch[..n] == b16[..]) as usize
					})
				}),
			];
			let results = impls.into_iter().map(|f| f()).collect();
			(*name, results)
		})
		.collect()
}

fn run_ratio_table() {
	let inputs = get_inputs();

	let rows1 = utf8_to_utf16_rows(&inputs);
	print_ratio_table(
		"UTF-8 -> UTF-16",
		&["xutf", "simdutf", "simdutf-valid", "encoding_rs", "std-chars"],
		&rows1,
	);

	let rows2 = utf16_to_utf8_rows(&inputs);
	print_ratio_table(
		"UTF-16 -> UTF-8",
		&["xutf", "simdutf", "simdutf-valid", "encoding_rs", "std-chars"],
		&rows2,
	);

	let rows3 = utf8_to_utf32_rows(&inputs);
	print_ratio_table("UTF-8 -> UTF-32", &["xutf", "simdutf", "simdutf-valid", "std-chars"], &rows3);

	let rows4 = utf32_to_utf8_rows(&inputs);
	print_ratio_table("UTF-32 -> UTF-8", &["xutf", "simdutf", "simdutf-valid", "std-chars"], &rows4);

	let rows5 = equality_rows(&inputs);
	print_ratio_table(
		"equality",
		&["xutf 8/8", "memcmp", "xutf i8/8", "std-icase", "xutf 8/16", "cvt+memcmp"],
		&rows5,
	);
}

const INPUT_NAMES: &[&str] = &[
	"ascii",
	"mixed",
	"cyrillic-2b",
	"pure-2b",
	"cjk-3b",
	"pure-3b",
	"emoji-4b",
	"syn-m12",
	"syn-alt",
	"syn-m3",
];

fn input_seed(name: &str) -> &'static str {
	match name {
		"ascii" => "The quick brown fox jumps over the lazy dog; 0123456789. ",
		"mixed" => "Cafe patrons enjoy a naïve résumé of 42 items — mostly plain text. ",
		"cyrillic-2b" => "Съешь же ещё этих мягких французских булок да выпей чаю. ",
		"pure-2b" => "Ж",
		"cjk-3b" => "日本語のテキストです。中文文本測試。한국어 텍스트입니다。",
		"pure-3b" => "日",
		"emoji-4b" => "😀😁😂🤣😃😄😅😆🥲🙂🎉🚀",
		"syn-m12" => "aaaaaaaaaaaaaaaé",
		"syn-alt" => "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaé",
		"syn-m3" => "aaaaaaaaaaaaaaa—",
		_ => unreachable!(),
	}
}

#[divan::bench_group(sample_count = 500)]
mod utf8_to_utf16 {
	use super::*;

	#[divan::bench(args = INPUT_NAMES)]
	fn xutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src = s.as_bytes();
		let mut dst = vec![0u16; src.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(src))
			.bench_local(|| transcode_into::<Utf8, Utf16>(src, &mut dst, AsciiCase::Preserve).1);
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn simdutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src = s.as_bytes();
		let mut dst = vec![0u16; src.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(src))
			.bench_local(|| {
				// SAFETY: `src` is valid UTF-8 and `dst` has at least one u16 per input byte.
				unsafe {
					simdutf::convert_valid_utf8_to_utf16(src.as_ptr(), src.len(), dst.as_mut_ptr())
				}
			});
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn encoding_rs(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src = s.as_bytes();
		let mut dst = vec![0u16; src.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(src))
			.bench_local(|| encoding_rs::mem::convert_utf8_to_utf16(src, &mut dst));
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn std_chars(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src = s.as_bytes();
		let mut dst = vec![0u16; src.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(src))
			.bench_local(|| {
				let mut n = 0;
				for u in s.encode_utf16() {
					dst[n] = u;
					n += 1;
				}
				n
			});
	}
}

#[divan::bench_group(sample_count = 500)]
mod utf16_to_utf8 {
	use super::*;

	#[divan::bench(args = INPUT_NAMES)]
	fn xutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src: Vec<u16> = s.encode_utf16().collect();
		let src_bytes = src.len() * 2;
		let mut dst = vec![0u8; src.len() * 4 + 16];

		bencher
			.counter(divan::counter::BytesCount::new(src_bytes))
			.bench_local(|| transcode_into::<Utf16, Utf8>(&src, &mut dst, AsciiCase::Preserve).1);
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn simdutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src: Vec<u16> = s.encode_utf16().collect();
		let src_bytes = src.len() * 2;
		let mut dst = vec![0u8; src.len() * 4 + 16];

		bencher
			.counter(divan::counter::BytesCount::new(src_bytes))
			.bench_local(|| {
				// SAFETY: `src` is valid UTF-16 and `dst` reserves four bytes per code unit.
				unsafe {
					simdutf::convert_valid_utf16_to_utf8(src.as_ptr(), src.len(), dst.as_mut_ptr())
				}
			});
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn encoding_rs(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src: Vec<u16> = s.encode_utf16().collect();
		let src_bytes = src.len() * 2;
		let mut dst = vec![0u8; src.len() * 4 + 16];

		bencher
			.counter(divan::counter::BytesCount::new(src_bytes))
			.bench_local(|| encoding_rs::mem::convert_utf16_to_utf8(&src, &mut dst));
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn std_chars(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src: Vec<u16> = s.encode_utf16().collect();
		let src_bytes = src.len() * 2;
		let mut dst = vec![0u8; src.len() * 4 + 16];

		bencher
			.counter(divan::counter::BytesCount::new(src_bytes))
			.bench_local(|| {
				let mut n = 0;
				for r in char::decode_utf16(src.iter().copied()) {
					n += r.unwrap().encode_utf8(&mut dst[n..]).len();
				}
				n
			});
	}
}

#[divan::bench_group(sample_count = 500)]
mod utf8_to_utf32 {
	use super::*;

	#[divan::bench(args = INPUT_NAMES)]
	fn xutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src = s.as_bytes();
		let mut dst = vec![0u32; src.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(src))
			.bench_local(|| transcode_into::<Utf8, Utf32>(src, &mut dst, AsciiCase::Preserve).1);
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn simdutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src = s.as_bytes();
		let mut dst = vec![0u32; src.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(src))
			.bench_local(|| {
				// SAFETY: `src` is valid UTF-8 and `dst` has at least one u32 per input byte.
				unsafe {
					simdutf::convert_valid_utf8_to_utf32(src.as_ptr(), src.len(), dst.as_mut_ptr())
				}
			});
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn std_chars(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src = s.as_bytes();
		let mut dst = vec![0u32; src.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(src))
			.bench_local(|| {
				let mut n = 0;
				for c in s.chars() {
					dst[n] = c as u32;
					n += 1;
				}
				n
			});
	}
}

#[divan::bench_group(sample_count = 500)]
mod utf32_to_utf8 {
	use super::*;

	#[divan::bench(args = INPUT_NAMES)]
	fn xutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src: Vec<u32> = s.chars().map(|c| c as u32).collect();
		let src_bytes = src.len() * 4;
		let mut dst = vec![0u8; src.len() * 4 + 16];

		bencher
			.counter(divan::counter::BytesCount::new(src_bytes))
			.bench_local(|| transcode_into::<Utf32, Utf8>(&src, &mut dst, AsciiCase::Preserve).1);
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn simdutf(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src: Vec<u32> = s.chars().map(|c| c as u32).collect();
		let src_bytes = src.len() * 4;
		let mut dst = vec![0u8; src.len() * 4 + 16];

		bencher
			.counter(divan::counter::BytesCount::new(src_bytes))
			.bench_local(|| {
				// SAFETY: `src` contains Unicode scalar values and `dst` reserves four bytes per value.
				unsafe {
					simdutf::convert_valid_utf32_to_utf8(src.as_ptr(), src.len(), dst.as_mut_ptr())
				}
			});
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn std_chars(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let src: Vec<u32> = s.chars().map(|c| c as u32).collect();
		let src_bytes = src.len() * 4;
		let mut dst = vec![0u8; src.len() * 4 + 16];

		bencher
			.counter(divan::counter::BytesCount::new(src_bytes))
			.bench_local(|| {
				let mut n = 0;
				for &cp in &src {
					n += char::from_u32(cp).unwrap().encode_utf8(&mut dst[n..]).len();
				}
				n
			});
	}
}

#[divan::bench_group(sample_count = 500)]
mod equality {
	use super::*;

	#[divan::bench(args = INPUT_NAMES)]
	fn xutf_8_8(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let a = s.as_bytes();
		let b = a.to_vec();

		bencher
			.counter(divan::counter::BytesCount::of_slice(a))
			.bench_local(|| xutf::equals::<Utf8, Utf8>(a, &b));
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn memcmp(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let a = s.as_bytes();
		let b = a.to_vec();

		bencher
			.counter(divan::counter::BytesCount::of_slice(a))
			.bench_local(|| a == &b[..]);
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn xutf_i8_8(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let a = s.as_bytes();
		let b = a.to_vec();

		bencher
			.counter(divan::counter::BytesCount::of_slice(a))
			.bench_local(|| xutf::equals_ignore_ascii_case::<Utf8, Utf8>(a, &b));
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn std_icase(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let a = s.as_bytes();
		let b = a.to_vec();

		bencher
			.counter(divan::counter::BytesCount::of_slice(a))
			.bench_local(|| a.eq_ignore_ascii_case(&b));
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn xutf_8_16(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let a = s.as_bytes();
		let b16: Vec<u16> = s.encode_utf16().collect();

		bencher
			.counter(divan::counter::BytesCount::of_slice(a))
			.bench_local(|| xutf::equals::<Utf8, Utf16>(a, &b16));
	}

	#[divan::bench(args = INPUT_NAMES)]
	fn cvt_memcmp(bencher: Bencher, name: &str) {
		let s = build_input(input_seed(name), TARGET);
		let a = s.as_bytes();
		let b16: Vec<u16> = s.encode_utf16().collect();
		let mut scratch = vec![0u16; a.len() + 16];

		bencher
			.counter(divan::counter::BytesCount::of_slice(a))
			.bench_local(|| {
				let n = encoding_rs::mem::convert_utf8_to_utf16(a, &mut scratch);
				scratch[..n] == b16[..]
			});
	}
}
