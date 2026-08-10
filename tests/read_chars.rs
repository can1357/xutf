//! Behavior tests for `BufReadCharsExt` stream decoding.
//!
//! The UTF-8 half mirrors the `utf8-chars` crate's test suite so the two
//! stay drop-in interchangeable; the rest covers the other encodings and
//! buffer-boundary (straddle) handling.

use std::io::{BufRead, BufReader, ErrorKind};

use xutf::{BufReadCharsExt, ReadCharError, StreamDecode, Utf16Be, Utf16Le, Utf32Be, Utf32Le};

fn raw(bytes: &[u8]) -> Vec<Result<char, ReadCharError>> {
	BufReader::new(bytes).chars_raw().collect()
}

#[test]
fn read_valid_unicode() {
	assert_eq!(
		vec!['A', 'B', 'c', 'd', ' ', 'А', 'Б', 'в', 'г', 'д', ' ', 'U', '\0'],
		BufReader::new("ABcd АБвгд U\0".as_bytes())
			.chars_raw()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);
}

#[test]
fn edgecase_one_two_bytes() {
	assert_eq!(
		vec!['\x7F'],
		raw(&[0x7f])
			.into_iter()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);
	assert_eq!(
		vec!['\u{0080}'],
		raw(&[0xc2, 0x80])
			.into_iter()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);

	let res = raw(&[0xc2]);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xc2][..], err.as_bytes());
	assert_eq!(ErrorKind::UnexpectedEof, err.as_io_error().kind());

	let res = raw(&[0xc1, 0xbf]);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xc1, 0xbf][..], err.as_bytes());
	assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
}

#[test]
fn edgecase_two_three_bytes() {
	assert_eq!(
		vec!['\u{07FF}'],
		raw(&[0xdf, 0xbf])
			.into_iter()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);
	assert_eq!(
		vec!['\u{0800}'],
		raw(&[0xe0, 0xa0, 0x80])
			.into_iter()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);

	let res = raw(&[0xe0, 0xa0]);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xe0, 0xa0][..], err.as_bytes());
	assert_eq!(ErrorKind::UnexpectedEof, err.as_io_error().kind());

	let res = raw(&[0xe0, 0x9f, 0xbf]);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xe0, 0x9f, 0xbf][..], err.as_bytes());
	assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
}

#[test]
fn edgecase_three_four_bytes() {
	assert_eq!(
		vec!['\u{00FFFF}'],
		raw(&[0xef, 0xbf, 0xbf])
			.into_iter()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);
	assert_eq!(
		vec!['\u{010000}'],
		raw(&[0xf0, 0x90, 0x80, 0x80])
			.into_iter()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);

	let res = raw(&[0xf0, 0x90, 0x80]);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xf0, 0x90, 0x80][..], err.as_bytes());
	assert_eq!(ErrorKind::UnexpectedEof, err.as_io_error().kind());

	let res = raw(&[0xf0, 0x8f, 0xbf, 0xbf]);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xf0, 0x8f, 0xbf, 0xbf][..], err.as_bytes());
	assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
}

#[test]
fn edgecase_four_bytes_max() {
	assert_eq!(
		vec!['\u{10FFFF}'],
		raw(&[0xf4, 0x8f, 0xbf, 0xbf])
			.into_iter()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);

	let res = raw(&[0xf8, 0x41]);
	assert_eq!(2, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xf8][..], err.as_bytes());
	assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
	assert_eq!(&'A', res[1].as_ref().unwrap());

	// Scalar values above U+10FFFF are rejected after the full read.
	let res = raw(&[0xf4, 0x90, 0x80, 0x80]);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xf4, 0x90, 0x80, 0x80][..], err.as_bytes());
	assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
}

#[test]
fn read_io_valid_unicode() {
	assert_eq!(
		vec!['A', 'B', 'c', 'd', ' ', 'А', 'Б', 'в', 'г', 'д', ' ', 'U', '\0'],
		BufReader::new("ABcd АБвгд U\0".as_bytes())
			.chars()
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);
}

#[test]
fn read_valid_unicode_from_dyn_read() {
	let bytes: &mut dyn BufRead = &mut BufReader::new("ABcd АБвгд UV".as_bytes());
	assert_eq!(
		vec!['A', 'B', 'c', 'd', ' ', 'А', 'Б', 'в', 'г', 'д', ' ', 'U', 'V'],
		bytes.chars_raw().map(|x| x.unwrap()).collect::<Vec<_>>()
	);
}

#[test]
fn do_not_take_extra_bytes() {
	let mut bytes = BufReader::new("ABcd АБвгд UV".as_bytes());
	assert_eq!(
		vec!['A', 'B', 'c', 'd'],
		bytes
			.chars_raw()
			.take(4)
			.map(|x| x.unwrap())
			.collect::<Vec<_>>()
	);
	assert_eq!(
		vec![' ', 'А', 'Б', 'в', 'г', 'д', ' ', 'U', 'V'],
		bytes.chars_raw().map(|x| x.unwrap()).collect::<Vec<_>>()
	);
}

#[test]
fn read_value_out_of_range() {
	let res = raw(&[0xf5, 0x8f, 0xbf, 0xbf]);
	assert_eq!(1, res.len());
	assert_eq!(&[0xf5, 0x8f, 0xbf, 0xbf][..], res[0].as_ref().err().unwrap().as_bytes());
}

#[test]
fn read_io_value_out_of_range() {
	let mut bytes = BufReader::new(&[0xf5, 0x8f, 0xbf, 0xbf][..]);
	let res = bytes.chars().collect::<Vec<_>>();
	assert_eq!(1, res.len());
	assert_eq!(ErrorKind::InvalidData, res[0].as_ref().err().unwrap().kind());
}

#[test]
fn read_io_incomplete_twobyte() {
	let mut bytes = BufReader::new(&[0xc3][..]); // 0xC3 0xA4 = 'ä'
	let res = bytes.chars().collect::<Vec<_>>();
	assert_eq!(1, res.len());
	assert_eq!(ErrorKind::UnexpectedEof, res[0].as_ref().err().unwrap().kind());
}

#[test]
fn read_io_incomplete_threebyte() {
	let mut bytes = BufReader::new(&[0xe1, 0xba][..]); // 0xE1 0xBA 0xB9 = 'ẹ'
	let res = bytes.chars().collect::<Vec<_>>();
	assert_eq!(1, res.len());
	assert_eq!(ErrorKind::UnexpectedEof, res[0].as_ref().err().unwrap().kind());
}

#[test]
fn read_surrogate() {
	let res = raw(&[0xed, 0xa0, 0x80]);
	assert_eq!(1, res.len());
	assert_eq!(&[0xed, 0xa0, 0x80][..], res[0].as_ref().err().unwrap().as_bytes());
}

#[test]
fn read_invalid_sequences() {
	let res = raw(&[0x81, 0x82, 0xc1, 0x07, 0xc1, 0x87, 0xc2, 0xc2, 0x82, 0xf7, 0x88, 0x89, 0x07]);
	assert_eq!(9, res.len());
	assert_eq!(&[0x81][..], res[0].as_ref().err().unwrap().as_bytes());
	assert_eq!(&[0x82][..], res[1].as_ref().err().unwrap().as_bytes());
	assert_eq!(&[0xc1][..], res[2].as_ref().err().unwrap().as_bytes());
	assert_eq!('\x07', *res[3].as_ref().unwrap());
	assert_eq!(&[0xc1, 0x87][..], res[4].as_ref().err().unwrap().as_bytes());
	assert_eq!(&[0xc2][..], res[5].as_ref().err().unwrap().as_bytes());
	assert_eq!('\u{82}', *res[6].as_ref().unwrap());
	assert_eq!(&[0xf7, 0x88, 0x89][..], res[7].as_ref().err().unwrap().as_bytes());
	assert_eq!('\x07', *res[8].as_ref().unwrap());
}

#[test]
fn read_char_single_shot() {
	let mut bytes = BufReader::new("aи𐍈".as_bytes());
	assert_eq!(Some('a'), bytes.read_char().unwrap());
	assert_eq!(Some('и'), bytes.read_char_raw().unwrap());
	assert_eq!(Some('𐍈'), bytes.read_char().unwrap());
	assert_eq!(None, bytes.read_char().unwrap());
}

#[test]
fn error_display_and_source() {
	let res = raw(&[0xe0, 0xa0]);
	let err = res[0].as_ref().err().unwrap();
	assert_eq!("invalid byte sequence E0 A0 read (unexpected EOF)", err.to_string());
	assert!(std::error::Error::source(err).is_some());
	// Converting to io::Error keeps kind and byte context.
	let res = raw(&[0xed, 0xa0, 0x80]);
	let io_err: std::io::Error = res.into_iter().next().unwrap().unwrap_err().into();
	assert_eq!(ErrorKind::InvalidData, io_err.kind());
	assert!(io_err.to_string().contains("ED A0 80"));
}

// ─── deterministic pseudo-random helpers ────────────────────────────────────

const fn lcg(state: &mut u64) -> u64 {
	*state = state
		.wrapping_mul(6364136223846793005)
		.wrapping_add(1442695040888963407);
	*state >> 33
}

fn random_string(seed: u64, len: usize) -> String {
	let mut state = seed;
	let mut s = String::new();
	while s.chars().count() < len {
		if let Some(c) = char::from_u32((lcg(&mut state) as u32) % 0x110000) {
			s.push(c);
		}
	}
	s
}
/// Regression: draining a full 64-char lookahead batch via `next()` leaves
/// the queue indices at capacity; a following fold-driven consumer (`map` +
/// `collect::<String>` routes through `Iterator::fold`) must rewind them, or
/// `decide` cannot queue and UTF-16/32 misclassify the next valid unit as
/// invalid.
#[test]
fn next_through_full_batch_then_fold() {
	let text: String = ('a'..='z').cycle().take(100).collect();
	let utf8 = text.as_bytes().to_vec();
	let utf16 = utf16_bytes::<false>(&text);
	let utf32 = utf32_bytes::<false>(&text);

	let mut reader = BufReader::new(&utf8[..]);
	let mut iter = reader.chars_raw();
	let head: String = (0..64).map(|_| iter.next().unwrap().unwrap()).collect();
	let tail: String = iter.map(|c| c.unwrap()).collect();
	assert_eq!(text, head + &tail);

	let mut reader = BufReader::new(&utf16[..]);
	let mut iter = reader.decode_chars_raw::<Utf16Le>();
	let head: String = (0..64).map(|_| iter.next().unwrap().unwrap()).collect();
	let tail: String = iter.map(|c| c.unwrap()).collect();
	assert_eq!(text, head + &tail);

	let mut reader = BufReader::new(&utf32[..]);
	let mut iter = reader.decode_chars_raw::<Utf32Le>();
	let head: String = (0..64).map(|_| iter.next().unwrap().unwrap()).collect();
	let tail: String = iter.map(|c| c.unwrap()).collect();
	assert_eq!(text, head + &tail);
}

/// A panic out of a fold-driven callback must leave the reader positioned
/// exactly after the char that was handed to the panicking call: the batch
/// behind it is neither skipped nor re-delivered.
#[test]
fn fold_panic_leaves_reader_positioned() {
	let text: String = ('a'..='z').cycle().take(200).collect();
	let mut reader = BufReader::new(text.as_bytes());

	let hook = std::panic::take_hook();
	std::panic::set_hook(Box::new(|_| {}));
	let seen = std::cell::RefCell::new(String::new());
	let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		reader.chars_raw().for_each(|c| {
			let c = c.unwrap();
			seen.borrow_mut().push(c);
			assert!(c != 'e', "stop");
		});
	}));
	std::panic::set_hook(hook);

	assert!(unwound.is_err());
	let seen = seen.into_inner();
	assert_eq!("abcde", seen);
	// The panicking char counts as yielded; everything after it survives.
	let rest: String = reader.chars_raw().map(|c| c.unwrap()).collect();
	assert_eq!(text[seen.len()..], rest);
}

/// Every valid string round-trips regardless of the reader's buffer size
/// (capacities 1-3 force every multi-byte char across a buffer boundary).
#[test]
fn roundtrip_strings_all_capacities() {
	for seed in 0..64u64 {
		let s = random_string(seed + 1, 199);
		for cap in [1, 2, 3, 7, 64, 8192] {
			let mut t = String::new();
			BufReader::with_capacity(cap, s.as_bytes())
				.chars_raw()
				.for_each(|c| t.push(c.unwrap()));
			assert_eq!(s, t, "cap {cap} seed {seed}");
		}
	}
}

/// Every input byte is accounted for, in order, by either a decoded char or
/// an error's byte sequence (the `utf8-chars` `read_array` property).
#[test]
fn roundtrip_arbitrary_bytes() {
	for seed in 0..256u64 {
		let mut state = seed.wrapping_add(99);
		let len = (lcg(&mut state) % 300) as usize;
		let bytes: Vec<u8> = (0..len).map(|_| lcg(&mut state) as u8).collect();
		for cap in [1, 3, 8192] {
			let mut t = Vec::new();
			BufReader::with_capacity(cap, &bytes[..])
				.chars_raw()
				.for_each(|c| match c {
					Ok(c) => t.extend_from_slice(c.to_string().as_bytes()),
					Err(e) => t.extend_from_slice(e.as_bytes()),
				});
			assert_eq!(bytes, t, "cap {cap} seed {seed}");
		}
	}
}

// ─── UTF-16 / UTF-32 ────────────────────────────────────────────────────────

fn utf16_bytes<const BE: bool>(s: &str) -> Vec<u8> {
	s.encode_utf16()
		.flat_map(|u| if BE { u.to_be_bytes() } else { u.to_le_bytes() })
		.collect()
}

fn utf32_bytes<const BE: bool>(s: &str) -> Vec<u8> {
	s.chars()
		.flat_map(|c| {
			if BE {
				(c as u32).to_be_bytes()
			} else {
				(c as u32).to_le_bytes()
			}
		})
		.collect()
}

fn decode_all<E: StreamDecode>(bytes: &[u8], cap: usize) -> Vec<Result<char, ReadCharError>> {
	BufReader::with_capacity(cap, bytes)
		.decode_chars_raw::<E>()
		.collect()
}

#[test]
fn utf16_roundtrip_both_orders_all_capacities() {
	for seed in 0..32u64 {
		let s = random_string(seed + 7, 151);
		let le = utf16_bytes::<false>(&s);
		let be = utf16_bytes::<true>(&s);
		for cap in [1, 2, 3, 7, 8192] {
			let t: String = decode_all::<Utf16Le>(&le, cap)
				.into_iter()
				.map(|c| c.unwrap())
				.collect();
			assert_eq!(s, t, "le cap {cap} seed {seed}");
			let t: String = decode_all::<Utf16Be>(&be, cap)
				.into_iter()
				.map(|c| c.unwrap())
				.collect();
			assert_eq!(s, t, "be cap {cap} seed {seed}");
		}
	}
}

#[test]
fn utf16_lone_surrogates() {
	// Lone low surrogate.
	let res = decode_all::<Utf16Be>(&[0xdc, 0x00, 0x00, 0x41], 8192);
	assert_eq!(2, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0xdc, 0x00][..], err.as_bytes());
	assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
	assert_eq!('A', *res[1].as_ref().unwrap());

	// High surrogate followed by a non-low unit: the follower survives as
	// the next char (lookahead is not lost), at every buffer size.
	for cap in [1, 2, 3, 8192] {
		let res = decode_all::<Utf16Be>(&[0xd8, 0x00, 0x00, 0x41], cap);
		assert_eq!(2, res.len(), "cap {cap}");
		let err = res[0].as_ref().err().unwrap();
		assert_eq!(&[0xd8, 0x00][..], err.as_bytes());
		assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
		assert_eq!('A', *res[1].as_ref().unwrap());
	}

	// High surrogate at EOF.
	let res = decode_all::<Utf16Le>(&[0x00, 0xd8], 8192);
	assert_eq!(1, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0x00, 0xd8][..], err.as_bytes());
	assert_eq!(ErrorKind::UnexpectedEof, err.as_io_error().kind());

	// Odd trailing byte.
	let res = decode_all::<Utf16Le>(&[0x41, 0x00, 0x42], 8192);
	assert_eq!(2, res.len());
	assert_eq!('A', *res[0].as_ref().unwrap());
	let err = res[1].as_ref().err().unwrap();
	assert_eq!(&[0x42][..], err.as_bytes());
	assert_eq!(ErrorKind::UnexpectedEof, err.as_io_error().kind());
}

#[test]
fn utf16_surrogate_pairs() {
	// U+1F980 = D83E DD80.
	let res = decode_all::<Utf16Le>(&[0x3e, 0xd8, 0x80, 0xdd], 8192);
	assert_eq!(vec!['🦀'], res.into_iter().map(|c| c.unwrap()).collect::<Vec<_>>());
	let res = decode_all::<Utf16Be>(&[0xd8, 0x3e, 0xdd, 0x80], 2);
	assert_eq!(vec!['🦀'], res.into_iter().map(|c| c.unwrap()).collect::<Vec<_>>());
}

#[test]
fn utf32_roundtrip_and_errors() {
	for seed in 0..32u64 {
		let s = random_string(seed + 13, 151);
		let le = utf32_bytes::<false>(&s);
		let be = utf32_bytes::<true>(&s);
		for cap in [1, 3, 5, 8192] {
			let t: String = decode_all::<Utf32Le>(&le, cap)
				.into_iter()
				.map(|c| c.unwrap())
				.collect();
			assert_eq!(s, t, "le cap {cap} seed {seed}");
			let t: String = decode_all::<Utf32Be>(&be, cap)
				.into_iter()
				.map(|c| c.unwrap())
				.collect();
			assert_eq!(s, t, "be cap {cap} seed {seed}");
		}
	}

	// Surrogate scalar is invalid.
	let res = decode_all::<Utf32Be>(&[0x00, 0x00, 0xd8, 0x00, 0x00, 0x00, 0x00, 0x41], 8192);
	assert_eq!(2, res.len());
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0x00, 0x00, 0xd8, 0x00][..], err.as_bytes());
	assert_eq!(ErrorKind::InvalidData, err.as_io_error().kind());
	assert_eq!('A', *res[1].as_ref().unwrap());

	// Out of range.
	let res = decode_all::<Utf32Le>(&[0x00, 0x00, 0x11, 0x00], 8192);
	assert_eq!(ErrorKind::InvalidData, res[0].as_ref().err().unwrap().as_io_error().kind());

	// Truncated unit at EOF.
	let res = decode_all::<Utf32Le>(&[0x41, 0x00, 0x00], 8192);
	let err = res[0].as_ref().err().unwrap();
	assert_eq!(&[0x41, 0x00, 0x00][..], err.as_bytes());
	assert_eq!(ErrorKind::UnexpectedEof, err.as_io_error().kind());
}

/// Byte-accounting property for UTF-16: chars re-encode plus error bytes
/// reconstruct the input exactly, in order.
#[test]
fn utf16_roundtrip_arbitrary_bytes() {
	for seed in 0..128u64 {
		let mut state = seed.wrapping_add(31);
		let len = (lcg(&mut state) % 200) as usize;
		let bytes: Vec<u8> = (0..len).map(|_| lcg(&mut state) as u8).collect();
		for cap in [1, 2, 3, 8192] {
			let mut t = Vec::new();
			for c in BufReader::with_capacity(cap, &bytes[..]).decode_chars_raw::<Utf16Le>() {
				match c {
					Ok(c) => {
						let mut units = [0u16; 2];
						for u in c.encode_utf16(&mut units) {
							t.extend_from_slice(&u.to_le_bytes());
						}
					},
					Err(e) => t.extend_from_slice(e.as_bytes()),
				}
			}
			assert_eq!(bytes, t, "cap {cap} seed {seed}");
		}
	}
}
