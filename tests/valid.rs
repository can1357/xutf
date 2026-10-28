//! `valid_prefix`, `valid_text_prefix`, `Utf8Decoder`, `wide_run` and
//! `plain_prefix` against `std` and scalar references.

use xutf::{
	Utf8Decoder, Utf8Step, WIDE_RUN, plain_prefix, simple_width, valid_prefix, valid_text_prefix,
	wide_run,
};

/// The standard library's valid prefix of `bytes`.
fn std_valid(bytes: &[u8]) -> usize {
	std::str::from_utf8(bytes).map_or_else(|error| error.valid_up_to(), str::len)
}

/// Checks both prefixes of `bytes` against the standard library.
fn check(bytes: &[u8]) {
	assert_eq!(valid_prefix(bytes).len(), std_valid(bytes), "{bytes:02x?}");
	let text = bytes
		.iter()
		.position(|&b| b < 0x20 || b == 0x7f)
		.unwrap_or(bytes.len());
	assert_eq!(valid_text_prefix(bytes).len(), std_valid(&bytes[..text]), "{bytes:02x?}");
}

/// A xorshift generator: `next(bound)` is below `bound`.
fn rng(mut seed: u64) -> impl FnMut(usize) -> usize {
	move |bound| {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		(seed % bound as u64) as usize
	}
}

#[test]
fn every_byte_pair_agrees_with_std() {
	for a in 0..=255u8 {
		for b in 0..=255u8 {
			check(&[a, b]);
		}
	}
}

#[test]
fn boundary_bytes_agree_with_std_at_every_block_offset() {
	// Every boundary byte, in sequences up to four long, at offsets that
	// put them across the 16- and 64-byte steps and in the scalar tail,
	// followed by more text or cut short.
	let edges = [
		0x00, 0x41, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xe1,
		0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf3, 0xf4, 0xf5, 0xff,
	];
	let mut buffer = Vec::new();
	for &a in &edges {
		for &b in &edges {
			for &c in &edges {
				for &d in &edges {
					for (offset, tail) in [(0, ""), (13, "中🙂"), (30, "é中"), (61, "xé"), (77, "é")]
					{
						buffer.clear();
						buffer.resize(offset, b'a');
						buffer.extend_from_slice(&[a, b, c, d]);
						buffer.extend_from_slice(tail.as_bytes());
						buffer.resize(100, b'z');
						check(&buffer);
						check(&buffer[..offset + 3]);
					}
				}
			}
		}
	}
}

#[test]
fn random_text_agrees_with_std_at_every_cut() {
	const PIECES: [&str; 12] = [
		"a",
		"hello ",
		"é",
		"中文字",
		"🙂",
		"\u{301}",
		"\r\n",
		"\x1b[31m",
		"\u{85}",
		"\u{10ffff}",
		"\u{d7ff}",
		"\u{e000}",
	];
	let mut next = rng(0x9e37_79b9_7f4a_7c15);
	for _ in 0..300 {
		let mut bytes = Vec::new();
		for _ in 0..40 {
			match next(30) {
				// A stray byte now and then: continuations, leads, invalid.
				0 => bytes.push(0x80 | next(0x80) as u8),
				n => bytes.extend_from_slice(PIECES[n % PIECES.len()].as_bytes()),
			}
		}
		for end in 0..=bytes.len() {
			check(&bytes[..end]);
		}
	}
}

#[test]
fn the_decoder_replaces_maximal_subparts_like_std() {
	let mut next = rng(0x2545_f491_4f6c_dd1d);
	for _ in 0..2000 {
		let bytes: Vec<u8> = (0..next(24))
			.map(|_| match next(4) {
				0 => next(0x80) as u8,
				_ => {
					[0x80, 0x8f, 0x9f, 0xbf, 0xc2, 0xdf, 0xe0, 0xe4, 0xed, 0xef, 0xf0, 0xf4, 0xf5, 0xff]
						[next(14)]
				},
			})
			.collect();
		let mut decoder = Utf8Decoder::new();
		let mut out = String::new();
		for &byte in &bytes {
			loop {
				let pending = decoder.is_pending();
				match decoder.push(byte) {
					Utf8Step::Char(c) => out.push(c),
					Utf8Step::Pending => {},
					Utf8Step::Reject => {
						out.push(char::REPLACEMENT_CHARACTER);
						if pending {
							continue;
						}
					},
				}
				break;
			}
		}
		if decoder.abandon() {
			out.push(char::REPLACEMENT_CHARACTER);
		}
		assert_eq!(out, String::from_utf8_lossy(&bytes), "{bytes:02x?}");
	}
}

#[test]
fn wide_runs_end_where_a_character_is_not_simple_and_wide() {
	const PIECES: [&str; 14] =
		["中", "文", "：", "。", "가", "ｱ", "Ａ", "ひ", "a", "é", "…", "🙂", "\u{302a}", "\u{3000}"];
	let mut next = rng(0x1234_5678_9abc_def1);
	let mut points = [0; WIDE_RUN];
	for _ in 0..5000 {
		let mut text = String::new();
		for _ in 0..next(24) {
			text.push_str(PIECES[next(PIECES.len())]);
		}
		let mut bytes = text.into_bytes();
		if next(4) == 0 {
			// Malformed: a lead cut short, an overlong form, a surrogate.
			let bad: [&[u8]; 3] = [b"\xe4\xb8", b"\xe0\x80\x80", b"\xed\xa0\x80"];
			let at = next(bytes.len() + 1);
			bytes.splice(at..at, bad[next(3)].iter().copied());
		}
		let expected: Vec<u16> = bytes
			.utf8_chunks()
			.next()
			.map(|chunk| chunk.valid())
			.unwrap_or_default()
			.chars()
			.take_while(|&c| c.len_utf8() == 3 && simple_width(c) == 2)
			.take(WIDE_RUN)
			.map(|c| c as u16)
			.collect();
		let run = wide_run(&bytes, &mut points);
		assert_eq!(&points[..run], expected, "{:?}", String::from_utf8_lossy(&bytes));
	}
}

#[test]
fn plain_prefix_stops_at_the_first_other_byte() {
	for len in 0..80 {
		for at in 0..=len {
			for byte in 0..=255u8 {
				let mut buffer = vec![b'a'; len];
				if at < len {
					buffer[at] = byte;
				}
				let plain = buffer
					.iter()
					.take_while(|&&byte| matches!(byte, 0x20..=0x7e))
					.count();
				assert_eq!(plain_prefix(&buffer), plain, "{len} {at} {byte:#x}");
			}
		}
	}
}
