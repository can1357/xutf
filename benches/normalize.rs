#![allow(clippy::unicode_not_nfc, reason = "benchmarks intentionally compare decomposed text")]

//! NFC/NFD/NFKC/NFKD benchmark against `unicode-normalization`.
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
		("compat", build_input("ﬁ ﬂ ＡＢＣ ² ³ µ Å Å ① ② ™ ｶﾀｶﾅ café. ", TARGET)),
	]
}

#[derive(Clone, Copy)]
enum Form {
	Nfc,
	Nfd,
	Nfkc,
	Nfkd,
}

impl Form {
	const fn title(self) -> &'static str {
		match self {
			Self::Nfc => "NFC normalization",
			Self::Nfd => "NFD normalization",
			Self::Nfkc => "NFKC normalization",
			Self::Nfkd => "NFKD normalization",
		}
	}

	fn make_normalized(self, working: &mut String) {
		match self {
			Self::Nfc => working.make_nfc().unwrap(),
			Self::Nfd => working.make_nfd().unwrap(),
			Self::Nfkc => working.make_nfkc().unwrap(),
			Self::Nfkd => working.make_nfkd().unwrap(),
		}
	}

	fn to_normalized(self, input: &str) -> String {
		match self {
			Self::Nfc => input.to_nfc(),
			Self::Nfd => input.to_nfd(),
			Self::Nfkc => input.to_nfkc(),
			Self::Nfkd => input.to_nfkd(),
		}
	}

	fn reference(self, input: &str) -> String {
		match self {
			Self::Nfc => input.nfc().collect(),
			Self::Nfd => input.nfd().collect(),
			Self::Nfkc => input.nfkc().collect(),
			Self::Nfkd => input.nfkd().collect(),
		}
	}
}

fn run_ratio_table() {
	let inputs = inputs();
	for form in [Form::Nfc, Form::Nfd, Form::Nfkc, Form::Nfkd] {
		let rows: Vec<_> = inputs
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
						form.make_normalized(working);
						black_box(working.as_bytes());
						working.len()
					},
				);
				let owned = measure(input.len(), || {
					let output = form.to_normalized(input);
					black_box(output.as_bytes());
					output.len()
				});
				let reference = measure(input.len(), || {
					let output = form.reference(input);
					black_box(output.as_bytes());
					output.len()
				});
				(*name, vec![in_place, reference, owned])
			})
			.collect();
		print_ratio_table(
			form.title(),
			&["xutf in-place", "unicode-normalization", "xutf owned"],
			&rows,
		);
	}
}
