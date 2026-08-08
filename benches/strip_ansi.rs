//! ANSI stripping benchmark against the `strip-ansi` crate.
//!
//! Run default ratio bench: `cargo bench --bench strip_ansi`
//! Run Divan bench: `DIVAN=1 cargo bench --bench strip_ansi`

use std::hint::black_box;

use divan::{Bencher, counter::BytesCount};
use rayon::prelude::*;
use strip_ansi::strip_ansi as reference_strip_ansi;
use xutf::{IntoAnsiStripped, MakeAnsiStripped, ToAnsiStripped};

mod common;
use common::{build_input, measure, measure_with_setup, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1 << 20;

fn inputs() -> Vec<(&'static str, String)> {
	vec![
		("plain", build_input("The quick brown fox jumps over 13 lazy dogs. ", TARGET)),
		(
			"sgr",
			build_input(
				"\x1b[38;5;244mINFO\x1b[0m request completed in \x1b[32m12ms\x1b[0m\n",
				TARGET,
			),
		),
		("dense", build_input("\x1b[1mone\x1b[0m \x1b[31mtwo\x1b[0m \x1b[4mthree\x1b[0m ", TARGET)),
		(
			"unicode",
			build_input("\x1b[36m状态\x1b[0m café \x1b[33m漢字 🦀\x1b[0m e\u{301}\n", TARGET),
		),
	]
}

fn run_ratio_table() {
	let rows: Vec<_> = inputs()
		.par_iter()
		.map(|(name, input)| {
			let expected = reference_strip_ansi(input);
			assert_eq!(input.to_ansi_stripped(), expected);
			assert_eq!(input.clone().into_ansi_stripped(), expected);
			let native_utf16: Vec<u16> = input.encode_utf16().collect();

			(*name, vec![
				measure(input.len(), || {
					let output = input.to_ansi_stripped();
					black_box(output.as_bytes());
					output.len()
				}),
				measure(input.len(), || {
					let output = reference_strip_ansi(input);
					black_box(output.as_bytes());
					output.len()
				}),
				measure_with_setup(
					input.len(),
					|| input.as_bytes().to_vec(),
					|working| {
						let mut output = working.as_mut_slice();
						output.make_ansi_stripped();
						black_box(&*output);
						output.len()
					},
				),
				measure_with_setup(
					input.len(),
					|| native_utf16.clone(),
					|working| {
						let mut output = working.as_mut_slice();
						output.make_ansi_stripped();
						black_box(&*output);
						output.len()
					},
				),
			])
		})
		.collect();

	print_ratio_table(
		"ANSI stripping",
		&["xutf str", "strip-ansi", "xutf in-place", "xutf utf16"],
		&rows,
	);
}

fn divan_input() -> String {
	build_input("\x1b[38;5;244mINFO\x1b[0m request completed in \x1b[32m12ms\x1b[0m\n", TARGET)
}

#[divan::bench]
fn xutf_str(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| input.to_ansi_stripped().len());
}

#[divan::bench]
fn xutf_owned(bencher: Bencher) {
	let input = divan_input();
	let bytes = input.len();
	bencher
		.counter(BytesCount::new(bytes))
		.with_inputs(|| input.clone())
		.bench_local_values(|value| value.into_ansi_stripped().len());
}

#[divan::bench]
fn xutf_in_place(bencher: Bencher) {
	let input = divan_input();
	let bytes = input.len();
	bencher
		.counter(BytesCount::new(bytes))
		.with_inputs(|| input.as_bytes().to_vec())
		.bench_local_values(|mut value| {
			let mut output = value.as_mut_slice();
			output.make_ansi_stripped();
			output.len()
		});
}

#[divan::bench]
fn strip_ansi_crate(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| reference_strip_ansi(&input).len());
}
