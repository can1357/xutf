#![allow(clippy::unicode_not_nfc, reason = "benchmarks intentionally compare decomposed text")]

//! NFC/NFD benchmark against `unicode-normalization`.
//!
//! Run: `cargo bench --bench normalize`

use std::hint::black_box;

use rayon::prelude::*;
use unicode_normalization::UnicodeNormalization as _;
use xutf::{MakeUnicodeNormalized, ToUnicodeNormalized};

mod common;
use common::{build_input, measure, measure_with_setup, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1024 * 1024;

fn inputs() -> Vec<(&'static str, String)> {
	let nfc_latin = "Crème brûlée, piñata, Ångström, déjà vu; voilà! ";
	let nfd_latin: String = nfc_latin.nfd().collect();
	let ordered_marks = "q\u{0323}\u{0307} a\u{0327}\u{0301} x\u{0334}\u{0323}\u{0308} ";
	let reversed_marks = "q\u{0307}\u{0323} a\u{0301}\u{0327} x\u{0308}\u{0323}\u{0334} ";

	vec![
		(
			"ascii",
			build_input(
				"The quick brown fox jumps over the lazy dog; normalization should disappear here. ",
				TARGET,
			),
		),
		("nfc-latin", build_input(nfc_latin, TARGET)),
		("nfd-latin", build_input(&nfd_latin, TARGET)),
		("ordered-marks", build_input(ordered_marks, TARGET)),
		("reversed-marks", build_input(reversed_marks, TARGET)),
		("hangul-jamo", build_input("한글 각 가나다 한글 ", TARGET)),
		(
			"mixed",
			build_input("Fast café, A\u{030a}, 한글과 한글, q\u{0307}\u{0323}, 界面, 🦀. ", TARGET),
		),
	]
}

fn run_ratio_table() {
	let inputs = inputs();
	let nfc_rows: Vec<_> = inputs
		.par_iter()
		.map(|(name, input)| {
			let in_place = measure_with_setup(
				input.len(),
				|| {
					let mut working = String::with_capacity(input.len() * 3);
					working.push_str(input);
					working
				},
				|working| {
					working.make_nfc().unwrap();
					black_box(working.as_bytes());
					working.len()
				},
			);
			let owned = measure(input.len(), || {
				let output = input.to_nfc();
				black_box(output.as_bytes());
				output.len()
			});
			let reference = measure(input.len(), || {
				let output: String = input.nfc().collect();
				black_box(output.as_bytes());
				output.len()
			});
			(*name, vec![in_place, reference, owned])
		})
		.collect();
	print_ratio_table(
		"NFC normalization",
		&["xutf in-place", "unicode-normalization", "xutf owned"],
		&nfc_rows,
	);

	let nfd_rows: Vec<_> = inputs
		.par_iter()
		.map(|(name, input)| {
			let in_place = measure_with_setup(
				input.len(),
				|| {
					let mut working = String::with_capacity(input.len() * 3);
					working.push_str(input);
					working
				},
				|working| {
					working.make_nfd().unwrap();
					black_box(working.as_bytes());
					working.len()
				},
			);
			let owned = measure(input.len(), || {
				let output = input.to_nfd();
				black_box(output.as_bytes());
				output.len()
			});
			let reference = measure(input.len(), || {
				let output: String = input.nfd().collect();
				black_box(output.as_bytes());
				output.len()
			});
			(*name, vec![in_place, reference, owned])
		})
		.collect();
	print_ratio_table(
		"NFD normalization",
		&["xutf in-place", "unicode-normalization", "xutf owned"],
		&nfd_rows,
	);
}
