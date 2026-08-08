use divan::Bencher;
use xutf::{AsciiCase, Utf8, Utf16, transcode_into};

const TARGET: usize = 256 * 1024;
const INPUTS: &[&str] = &["cyrillic", "cjk"];

fn input(name: &str) -> String {
	let seed = match name {
		"cyrillic" => "Съешь же ещё этих мягких французских булок да выпей чаю. ",
		"cjk" => "日本語のテキストです。中文文本測試。한국어 텍盛입니다。",
		_ => unreachable!(),
	};
	let mut out = String::with_capacity(TARGET + seed.len());
	while out.len() < TARGET {
		out.push_str(seed);
	}
	out
}

#[divan::bench(args = INPUTS)]
fn bench_utf8_to_utf16(bencher: Bencher, name: &str) {
	let src = input(name);
	let mut dst = vec![0u16; src.len() + 16];
	bencher
		.counter(divan::counter::BytesCount::of_slice(src.as_bytes()))
		.bench_local(|| transcode_into::<Utf8, Utf16>(src.as_bytes(), &mut dst, AsciiCase::Preserve));
}

fn main() {
	divan::main();
}
