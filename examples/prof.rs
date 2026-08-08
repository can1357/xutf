//! Profiling loop: `cargo run --release --example prof -- <pair> <input>
//! [secs]`. Pins one transcode pair on one corpus so `sample`/Instruments can
//! attribute time, and prints achieved GB/s.
use std::time::{Duration, Instant};

use xutf::{AsciiCase, Utf8, Utf16, Utf32, transcode_into};

fn build_input(seed: &str, target_bytes: usize) -> String {
	let mut s = String::with_capacity(target_bytes + seed.len());
	while s.len() < target_bytes {
		s.push_str(seed);
	}
	s
}

fn main() {
	let mut args = std::env::args().skip(1);
	let pair = args.next().expect("pair: 8-16 | 8-32 | 16-8 | 32-8");
	let input = args.next().expect("input name");
	let secs: f64 = args.next().map_or(2.0, |s| s.parse().unwrap());
	let seed = match input.as_str() {
		"ascii" => "The quick brown fox jumps over the lazy dog; 0123456789. ",
		"mixed" => "Cafe patrons enjoy a naïve résumé of 42 items — mostly plain text. ",
		"cyrillic-2b" => "Съешь же ещё этих мягких французских булок да выпей чаю. ",
		"pure-2b" => "ж",
		"cjk-3b" => "日本語のテキストです，中文文本, 한국어 ",
		"pure-3b" => "日",
		"emoji-4b" => "😀😁😂🤣😃😄😅😆🥲🙂🎉🚀",
		"syn-m12" => "aaaaaaaaaaaaaaaé",
		"syn-alt" => "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaé",
		"syn-m3" => "aaaaaaaaaaaaaaa—",
		other => panic!("unknown input {other}"),
	};
	let src8 = build_input(seed, 256 * 1024);
	let src16: Vec<u16> = src8.encode_utf16().collect();
	let src32: Vec<u32> = src8.chars().map(|c| c as u32).collect();
	let mut out16 = vec![0u16; src8.len() + 16];
	let mut out32 = vec![0u32; src8.len() + 16];
	let mut out8 = vec![0u8; src8.len() * 4 + 16];

	let duration = Duration::from_secs_f64(secs);
	let start = Instant::now();
	let mut bytes = 0usize;
	while start.elapsed() < duration {
		match pair.as_str() {
			"8-16" => {
				let (..) =
					transcode_into::<Utf8, Utf16>(src8.as_bytes(), &mut out16, AsciiCase::Preserve);
				bytes += src8.len();
			},
			"8-32" => {
				let (..) =
					transcode_into::<Utf8, Utf32>(src8.as_bytes(), &mut out32, AsciiCase::Preserve);
				bytes += src8.len();
			},
			"16-8" => {
				let (..) = transcode_into::<Utf16, Utf8>(&src16, &mut out8, AsciiCase::Preserve);
				bytes += src16.len() * 2;
			},
			"32-8" => {
				let (..) = transcode_into::<Utf32, Utf8>(&src32, &mut out8, AsciiCase::Preserve);
				bytes += src32.len() * 4;
			},
			other => panic!("unknown pair {other}"),
		}
	}
	let gbs = bytes as f64 / start.elapsed().as_secs_f64() / 1e9;
	println!(
		"{pair} {input}: {gbs:.2} GB/s (sink {})",
		out16[0] as usize + out32[0] as usize + out8[0] as usize
	);
}
