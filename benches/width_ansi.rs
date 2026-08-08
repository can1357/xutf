//! ANSI-aware visible width benchmark against allocating strip-and-measure
//! baselines.
//!
//! Run default ratio bench: `cargo bench --bench width_ansi`
//! Run Divan bench: `DIVAN=1 cargo bench --bench width_ansi`

use divan::{Bencher, counter::BytesCount};
use rayon::prelude::*;
use strip_ansi::strip_ansi as reference_strip_ansi;
use unicode_width::UnicodeWidthStr;
use xutf::{Text, ToAnsiStripped, width_ansi_str};

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1 << 20;

fn inputs() -> Vec<(&'static str, String)> {
	vec![
		(
			"sgr-heavy",
			build_input(
				"\x1b[38;5;244mINFO\x1b[0m \x1b[1mrequest\x1b[0m completed in \x1b[32m12ms\x1b[0m\n",
				TARGET,
			),
		),
		("ascii", build_input("The quick brown fox jumps over 13 lazy dogs. ", TARGET)),
		("mixed", build_input("状态 café 漢字 🦀 👩‍🚀 e\u{301}\n", TARGET)),
		(
			"hyperlinks",
			build_input(
				"\x1b]8;;https://example.com/docs\x07documentation\x1b]8;;\x1b\\ \x1b]8;;https://example.com/status\x07status\x1b]8;;\x1b\\\n",
				TARGET,
			),
		),
	]
}

fn reference_width(input: &str) -> usize {
	let stripped = reference_strip_ansi(input);
	UnicodeWidthStr::width(stripped.as_str())
}

fn run_ratio_table() {
	let rows: Vec<_> = inputs()
		.par_iter()
		.map(|(name, input)| {
			let expected = width_ansi_str(input);
			let stripped = input.to_ansi_stripped();
			assert_eq!(stripped.visible_width(), expected);
			// No cross-crate assertion: strip-ansi 0.1 leaves OSC-8 payloads
			// behind and unicode-width diverges from xutf's emoji-presentation
			// widths, so only xutf's one-pass and two-pass must agree.

			(*name, vec![
				measure(input.len(), || width_ansi_str(input)),
				measure(input.len(), || {
					let stripped = input.to_ansi_stripped();
					stripped.visible_width()
				}),
				measure(input.len(), || reference_width(input)),
			])
		})
		.collect();

	print_ratio_table(
		"ANSI-aware visible width",
		&["xutf one-pass", "xutf two-pass", "strip-ansi + unicode-width"],
		&rows,
	);
}

fn divan_input() -> String {
	build_input(
		"\x1b[38;5;244mINFO\x1b[0m \x1b[1mrequest\x1b[0m completed in \x1b[32m12ms\x1b[0m\n",
		TARGET,
	)
}

#[divan::bench]
fn xutf_one_pass(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| width_ansi_str(&input));
}

#[divan::bench]
fn xutf_two_pass(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| {
			let stripped = input.to_ansi_stripped();
			stripped.visible_width()
		});
}

#[divan::bench]
fn strip_ansi_unicode_width(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| reference_width(&input));
}
