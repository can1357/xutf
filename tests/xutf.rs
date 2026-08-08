//! Behavior tests against `std`'s UTF codecs as reference.

use std::cmp::Ordering;

use xutf::{
	AsciiCase, Bom, Encoding, Text, Utf8, Utf16, Utf16Be, Utf32, chars, codepoints, compare,
	compare_ignore_ascii_case, detect_bom, equals, equals_ignore_ascii_case, from_bytes, to_string,
	transcode, transcode_into, transcode_with_case, transcoded_len,
};

fn utf16(s: &str) -> Vec<u16> {
	s.encode_utf16().collect()
}

fn utf32(s: &str) -> Vec<u32> {
	s.chars().map(|c| c as u32).collect()
}

fn swap16(units: &[u16]) -> Vec<u16> {
	units.iter().map(|u| u.swap_bytes()).collect()
}

fn swap32(units: &[u32]) -> Vec<u32> {
	units.iter().map(|u| u.swap_bytes()).collect()
}

/// Mixed corpus: lengths straddle both SIMD tiers (16/8) and the scalar
/// tail; contents cover ASCII, 2/3/4-byte UTF-8 and surrogate pairs.
const CORPUS: &[&str] = &[
	"",
	"a",
	"Z9",
	"hello",
	"exactly8char",
	"The quick brown fox jumps over the lazy dog 0123456789",
	"pure ascii that is quite a bit longer than one simd register width",
	"naïve café résumé",
	"Ünïcödé mixed with ASCII runs that are long enough to vectorize nicely",
	"ελληνικά και ASCII",
	"日本語のテキスト",
	"mixed 日本語 with ascii and 𝕸𝖆𝖙𝖍 and more ascii tail to vectorize",
	"𐤀𐤁𐤂 phoenician",
	"\u{12000}\u{12001} cuneiform",
	"edge\u{10FFFF}\u{0}\u{7F}\u{80}\u{7FF}\u{800}\u{FFFF}\u{10000}",
	"ASCII HEAD then tail: \u{1F600}\u{1F601}\u{1F602}",
	"ЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖЖ",
	"日日日日日日日日日日日日日日日日日日",
	"😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀",
];

// ---------------------------------------------------------------- codecs --

#[test]
fn utf8_matches_std_for_all_scalars() {
	for cp in (0u32..=0x10ffff).filter(|c| !(0xd800..=0xdfff).contains(c)) {
		let c = char::from_u32(cp).unwrap();
		let mut reference = [0u8; 4];
		let reference = c.encode_utf8(&mut reference).as_bytes();
		let out = transcode::<Utf32, Utf8>(&[cp]);
		assert_eq!(out, reference, "encode U+{cp:04X}");
		assert_eq!(transcode::<Utf8, Utf32>(reference), [cp], "decode U+{cp:04X}");
	}
}

#[test]
fn utf16_matches_std_for_all_scalars() {
	for cp in (0u32..=0x10ffff).filter(|c| !(0xd800..=0xdfff).contains(c)) {
		let c = char::from_u32(cp).unwrap();
		let mut reference = [0u16; 2];
		let reference = c.encode_utf16(&mut reference);
		let out = transcode::<Utf32, Utf16>(&[cp]);
		assert_eq!(out, reference, "encode U+{cp:04X}");
		assert_eq!(transcode::<Utf16, Utf32>(reference), [cp], "decode U+{cp:04X}");
	}
}

#[test]
fn utf16_c_plus_plus_bit13_regression() {
	// U+12000: adj = 0x2000 has bit 13 set; the C++ original emitted
	// D808 FC00 instead of D808 DC00.
	assert_eq!(transcode::<Utf32, Utf16>(&[0x12000]), [0xd808, 0xdc00]);
}

#[test]
fn utf8_truncated_tail_decodes_to_zero() {
	// 4-byte lead with only two continuation bytes: consume rest, yield 0.
	let cps: Vec<u32> = codepoints::<Utf8>(&[b'a', 0xf0, 0x90, 0x80]).collect();
	assert_eq!(cps, [b'a' as u32, 0]);
}

#[test]
fn codepoints_iterate_backward() {
	let s = "a\u{e9}\u{4e2d}\u{1f600}";
	let forward: Vec<u32> = codepoints::<Utf8>(s.as_bytes()).collect();

	let mut backward: Vec<u32> = codepoints::<Utf8>(s.as_bytes()).rev().collect();
	backward.reverse();
	assert_eq!(backward, forward);

	let utf16: Vec<u16> = s.encode_utf16().collect();
	let mut backward: Vec<u32> = codepoints::<Utf16>(&utf16).rev().collect();
	backward.reverse();
	assert_eq!(backward, forward);

	let foreign16: Vec<u16> = utf16.iter().map(|u| u.swap_bytes()).collect();
	let mut backward: Vec<u32> = codepoints::<Utf16<true>>(&foreign16).rev().collect();
	backward.reverse();
	assert_eq!(backward, forward);

	let utf32: Vec<u32> = s.chars().map(u32::from).collect();
	let mut backward: Vec<u32> = codepoints::<Utf32>(&utf32).rev().collect();
	backward.reverse();
	assert_eq!(backward, forward);

	// Lone surrogates pass through backward like forward.
	assert_eq!(codepoints::<Utf16>(&[0xd808]).rev().collect::<Vec<_>>(), [0xd808]);

	// Malformed UTF-8 tails consume one unit per step from the back; the
	// boundaries differ from forward decoding but nothing is dropped.
	assert_eq!(codepoints::<Utf8>(&[b'a', 0xf0, 0x90, 0x80]).rev().count(), 4);
}

#[test]
fn malformed_same_encoding_transcode_keeps_truncated_tails() {
	let utf8 = [0x9e];
	let utf16 = [0xd800];

	assert_eq!(transcode_with_case::<Utf8, Utf8>(&utf8, AsciiCase::Lower), utf8,);
	assert_eq!(transcode_with_case::<Utf16, Utf16>(&utf16, AsciiCase::Upper), utf16,);

	let mut out = [0u8; 1];
	assert_eq!(transcode_into::<Utf8, Utf8>(&utf8, &mut out, AsciiCase::Lower), (1, 1),);
	assert_eq!(out, utf8);
}

#[test]
fn comparison_decodes_truncated_same_encoding_tails() {
	let truncated = [0x9e];
	let newline = *b"\n";

	assert_eq!(compare::<Utf8, Utf8>(&truncated, &newline), Ordering::Less);
	assert!(!equals::<Utf8, Utf8>(&truncated, &newline));
}

fn scalar_utf16(cps: &[u32]) -> Vec<u16> {
	let mut out = Vec::new();
	for &cp in cps {
		if cp <= 0xffff {
			out.push(cp as u16);
		} else {
			let adjusted = cp - 0x10000;
			out.push((0xd800 | (adjusted >> 10)) as u16);
			out.push((0xdc00 | (adjusted & 0x03ff)) as u16);
		}
	}
	out
}

#[test]
fn utf8_block_fast_paths_match_permissive_decoder() {
	let mut cases = Vec::new();

	let mut mixed_two = vec![b'a'];
	for _ in 0..31 {
		mixed_two.extend_from_slice(&[0xc0, 0x80]);
	}
	mixed_two.push(b' ');
	cases.push(("mixed 2-byte", mixed_two));

	let mut mixed_three = vec![b'a'];
	for _ in 0..21 {
		mixed_three.extend_from_slice(&[0xe0, 0x80, 0x80]);
	}
	cases.push(("mixed 3-byte", mixed_three));
	let mut del_mixed = vec![0x7f];
	for _ in 0..21 {
		del_mixed.extend_from_slice(&[0xe0, 0x80, 0x80]);
	}
	cases.push(("DEL plus 3-byte", del_mixed));

	let mut surrogate_three = vec![b'a'];
	for _ in 0..21 {
		surrogate_three.extend_from_slice(&[0xed, 0xa0, 0x80]);
	}
	cases.push(("surrogate 3-byte", surrogate_three));
	let mut mostly_ascii = Vec::new();
	for _ in 0..4 {
		mostly_ascii.extend_from_slice(
			b"Caf\xc3\xa9 patrons enjoy a na\xc3\xafve r\xc3\xa9sum\xc3\xa9 of 42 items \xe2\x80\x94 mostly plain text. ",
		);
	}
	cases.push(("mostly ASCII", mostly_ascii));

	let mut trailing_three = vec![0xd8, 0xb5];
	trailing_three.extend_from_slice(&[b'a'; 61]);
	trailing_three.extend_from_slice(&[0xe0, 0x80, 0x80]);
	cases.push(("2-byte start plus trailing 3-byte", trailing_three));

	let mut invalid_leads = Vec::new();
	for _ in 0..16 {
		invalid_leads.extend_from_slice(&[b'a', 0x80, 0x80]);
	}
	invalid_leads.extend_from_slice(&[b' '; 16]);
	cases.push(("continuations after ASCII", invalid_leads));

	// A 2-byte lead whose successor is not a continuation still ends a
	// sequence in the block loop's boundary mask, and a continuation at byte
	// 15 keeps the window off the 16-lane ASCII branch: the 12-byte widened
	// store must not claim those bytes as ASCII.
	let mut unpaired_lead = vec![0xd8, 0x00, 0x01, 0x00, 0x5b];
	unpaired_lead.extend_from_slice(&[0x00; 5]);
	unpaired_lead.extend_from_slice(&[0x15, 0x01, 0x01, 0x00, 0x2f]);
	unpaired_lead.extend_from_slice(&[0xa1; 7]);
	unpaired_lead.extend_from_slice(&[b'7'; 45]);
	cases.push(("unpaired 2-byte lead in a mixed block", unpaired_lead));
	for (name, input) in cases {
		let cps: Vec<u32> = codepoints::<Utf8>(&input).collect();
		assert_eq!(transcode::<Utf8, Utf16>(&input), scalar_utf16(&cps), "{name} 8->16");
		assert_eq!(transcode::<Utf8, Utf32>(&input), cps, "{name} 8->32");
	}
}

#[test]
fn utf16_lone_surrogates_pass_through() {
	assert_eq!(codepoints::<Utf16>(&[0xd808]).collect::<Vec<_>>(), [0xd808]);
	assert_eq!(codepoints::<Utf16>(&[0xdc37]).collect::<Vec<_>>(), [0xdc37]);
	let replaced: Vec<char> = chars::<Utf16>(&[0x41, 0xd808]).collect();
	assert_eq!(replaced, ['A', char::REPLACEMENT_CHARACTER]);
}

// ------------------------------------------------------------- transcode --

#[test]
fn transcode_all_pairs_matches_std() {
	for s in CORPUS {
		let (u8s, u16s, u32s) = (s.as_bytes().to_vec(), utf16(s), utf32(s));

		assert_eq!(transcode::<Utf8, Utf16>(&u8s), u16s, "{s:?} 8->16");
		assert_eq!(transcode::<Utf8, Utf32>(&u8s), u32s, "{s:?} 8->32");
		assert_eq!(transcode::<Utf16, Utf8>(&u16s), u8s, "{s:?} 16->8");
		assert_eq!(transcode::<Utf16, Utf32>(&u16s), u32s, "{s:?} 16->32");
		assert_eq!(transcode::<Utf32, Utf8>(&u32s), u8s, "{s:?} 32->8");
		assert_eq!(transcode::<Utf32, Utf16>(&u32s), u16s, "{s:?} 32->16");
		assert_eq!(transcode::<Utf8, Utf8>(&u8s), u8s, "{s:?} 8->8");

		assert_eq!(to_string::<Utf16>(&u16s).unwrap(), *s);
	}
}

#[test]
fn transcode_utf16_mixed_block_covers_both_mask_halves() {
	let text = "aЖ日".repeat(16);
	assert_eq!(transcode::<Utf16, Utf8>(&utf16(&text)), text.as_bytes());
}

#[test]
fn transcode_foreign_endianness_roundtrips() {
	for s in CORPUS {
		let be16 = swap16(&utf16(s));
		let be32 = swap32(&utf32(s));
		assert_eq!(transcode::<Utf16<true>, Utf8>(&be16), s.as_bytes(), "{s:?} be16->8");
		assert_eq!(transcode::<Utf8, Utf16<true>>(s.as_bytes()), be16, "{s:?} 8->be16");
		assert_eq!(transcode::<Utf32<true>, Utf8>(&be32), s.as_bytes(), "{s:?} be32->8");
		assert_eq!(transcode::<Utf16<true>, Utf32<true>>(&be16), be32, "{s:?} be16->be32");
	}
}

#[test]
fn transcoded_len_matches_transcode() {
	for s in CORPUS {
		assert_eq!(transcoded_len::<Utf8, Utf16>(s.as_bytes()), utf16(s).len(), "{s:?}");
		assert_eq!(transcoded_len::<Utf8, Utf32>(s.as_bytes()), utf32(s).len(), "{s:?}");
		assert_eq!(transcoded_len::<Utf32, Utf8>(&utf32(s)), s.len(), "{s:?}");
		assert_eq!(transcoded_len::<Utf16, Utf16>(&utf16(s)), utf16(s).len(), "{s:?}");
	}
}

#[test]
fn transcoded_len_counts_unpaired_utf16_surrogates_across_byte_orders() {
	// A high surrogate followed by a non-surrogate decodes as one (garbage)
	// BMP codepoint out of two units, so the swapped output is one unit
	// shorter; the byte-identical copy keeps both units.
	let src = [0xd801u16, 0x0101];
	assert_eq!(transcode::<Utf16<false>, Utf16<true>>(&src).len(), 1);
	assert_eq!(transcoded_len::<Utf16<false>, Utf16<true>>(&src), 1);
	assert_eq!(transcoded_len::<Utf16<false>, Utf16<false>>(&src), 2);
	assert_eq!(transcoded_len::<Utf32<false>, Utf32<true>>(&[0xd801, 0x0101]), 2);
}

#[test]
fn transcode_into_never_splits_codepoints() {
	let s = "ab\u{1F600}cd\u{12000}";
	let src = s.as_bytes();
	for cap in 0..=s.chars().map(char::len_utf16).sum::<usize>() {
		let mut dst = vec![0u16; cap];
		let (read, written) = transcode_into::<Utf8, Utf16>(src, &mut dst, AsciiCase::Preserve);
		assert!(written <= cap);
		// The written prefix decodes to a codepoint-prefix of the source.
		let expect: Vec<u16> = utf16(s)[..written].to_vec();
		assert_eq!(&dst[..written], expect, "cap {cap}");
		// Everything read was fully written.
		assert_eq!(transcoded_len::<Utf8, Utf16>(&src[..read]), written, "cap {cap}");
	}
}

#[test]
fn case_folding_is_ascii_only() {
	for s in CORPUS {
		let lower: Vec<u8> = transcode_with_case::<Utf8, Utf8>(s.as_bytes(), AsciiCase::Lower);
		assert_eq!(lower, s.to_ascii_lowercase().as_bytes(), "{s:?} lower");
		let upper: Vec<u16> = transcode_with_case::<Utf8, Utf16>(s.as_bytes(), AsciiCase::Upper);
		assert_eq!(upper, utf16(&s.to_ascii_uppercase()), "{s:?} upper cross-width");
	}
}

// --------------------------------------------------------------- compare --

/// Reference: codepoint-order comparison with optional ASCII folding.
fn reference_cmp(a: &str, b: &str, caseless: bool) -> Ordering {
	let fold = |c: char| {
		if caseless && c.is_ascii() {
			c.to_ascii_lowercase()
		} else {
			c
		}
	};
	let ka: Vec<u32> = a.chars().map(fold).map(|c| c as u32).collect();
	let kb: Vec<u32> = b.chars().map(fold).map(|c| c as u32).collect();
	ka.cmp(&kb)
}

#[test]
fn compare_matches_reference_across_encodings() {
	for a in CORPUS {
		for b in CORPUS {
			let (ord, iord) = (reference_cmp(a, b, false), reference_cmp(a, b, true));
			let (a16, b16) = (utf16(a), utf16(b));
			let (a32, b32) = (utf32(a), utf32(b));

			assert_eq!(compare::<Utf8, Utf8>(a.as_bytes(), b.as_bytes()), ord, "{a:?}/{b:?} 8/8");
			assert_eq!(compare::<Utf8, Utf16>(a.as_bytes(), &b16), ord, "{a:?}/{b:?} 8/16");
			assert_eq!(compare::<Utf16, Utf32>(&a16, &b32), ord, "{a:?}/{b:?} 16/32");
			assert_eq!(compare::<Utf32, Utf8>(&a32, b.as_bytes()), ord, "{a:?}/{b:?} 32/8");
			assert_eq!(
				compare_ignore_ascii_case::<Utf8, Utf16>(a.as_bytes(), &b16),
				iord,
				"{a:?}/{b:?} icase"
			);

			assert_eq!(equals::<Utf8, Utf16>(a.as_bytes(), &b16), ord == Ordering::Equal);
			assert_eq!(
				equals_ignore_ascii_case::<Utf16, Utf32>(&a16, &b32),
				iord == Ordering::Equal,
				"{a:?}/{b:?} ieq"
			);
		}
	}
}

#[test]
fn compare_case_pairs() {
	let a = "The Quick BROWN Fox With A Long Enough ASCII Prefix 12345";
	let b = "the quick brown fox with a long enough ascii prefix 12345";
	assert_eq!(compare_ignore_ascii_case::<Utf8, Utf8>(a.as_bytes(), b.as_bytes()), Ordering::Equal);
	assert!(equals_ignore_ascii_case::<Utf8, Utf16>(a.as_bytes(), &utf16(b)));
	assert_ne!(compare::<Utf8, Utf8>(a.as_bytes(), b.as_bytes()), Ordering::Equal);
}

#[test]
fn trailing_nul_is_not_equal() {
	// The C++ original returned the next codepoint's value (0) here and
	// declared the strings equal.
	assert_eq!(compare::<Utf8, Utf8>(b"abc\0", b"abc"), Ordering::Greater);
	assert_eq!(compare::<Utf8, Utf8>(b"abc", b"abc\0"), Ordering::Less);
	assert!(!equals::<Utf8, Utf8>(b"abc\0", b"abc"));
}

#[test]
fn cesu_surrogate_pair_is_not_a_supplementary_codepoint() {
	// CESU-style UTF-8 (`ED A0 80 ED B0 80`) decodes to two lone surrogate
	// codepoints D800/DC00; the UTF-16 pair [D800, DC00] decodes to the one
	// codepoint U+10000. Raw-unit shortcuts must not conflate them.
	let cesu = [0xed, 0xa0, 0x80, 0xed, 0xb0, 0x80];
	let pair = [0xd800u16, 0xdc00];
	assert!(!equals::<Utf8, Utf16>(&cesu, &pair));
	assert!(!equals::<Utf16, Utf8>(&pair, &cesu));
	// The genuine four-byte encoding of U+10000 is equal to the pair.
	let genuine = [0xf0, 0x90, 0x80, 0x80];
	assert!(equals::<Utf8, Utf16>(&genuine, &pair));
	// Longer strings exercise the chunked transcode-and-compare shortcut.
	let mut a = b"prefix that is long enough to vectorize ".to_vec();
	a.extend_from_slice(&cesu);
	let mut b: Vec<u16> = a[..a.len() - 6].iter().map(|&c| c as u16).collect();
	b.extend_from_slice(&pair);
	assert!(!equals::<Utf8, Utf16>(&a, &b));
	// A permissive out-of-range codepoint (U+110000) re-encodes into two
	// low-surrogate units; it is not two lone U+DC00 codepoints.
	assert!(!equals::<Utf8, Utf16>(&[0xf4, 0x90, 0x80, 0x80], &[0xdc00u16, 0xdc00]));
	// Overlong 4-byte forms can smuggle surrogate codepoints past the
	// 3-byte CESU pattern.
	let overlong = [0xf0, 0x8d, 0xa0, 0x80, 0xf0, 0x8d, 0xb0, 0x80];
	assert!(!equals::<Utf8, Utf16>(&overlong, &pair));
	// Malformed continuations are masked with `& 0x3F`, so `ED 60 80` also
	// decodes to U+D800; byte-pattern filters alone would miss it.
	let masked = [0xed, 0x60, 0x80, 0xed, 0x70, 0x80];
	assert!(!equals::<Utf8, Utf16>(&masked, &pair));
}

#[test]
fn out_of_range_utf32_matches_scalar_encoding() {
	// Permissive: values above U+10FFFF still encode like the scalar
	// four-byte writer (payload bits masked, `F7 BF BF BF` for u32::MAX),
	// even on inputs long enough to engage every SIMD tier.
	let src = [u32::MAX; 64];
	let out = transcode::<Utf32, Utf8>(&src);
	assert_eq!(out.len(), 256);
	for chunk in out.chunks(4) {
		assert_eq!(chunk, [0xf7, 0xbf, 0xbf, 0xbf]);
	}
	// Mixed in-range/out-of-range blocks stay consistent too.
	let mut mixed = [0x110000u32; 32];
	mixed[7] = 0x41;
	mixed[19] = 0x1f600;
	let out = transcode::<Utf32, Utf8>(&mixed);
	let mut expect = Vec::new();
	for &cp in &mixed {
		// Reference: the scalar encoder via a one-element slice.
		expect.extend_from_slice(&transcode::<Utf32, Utf8>(&[cp]));
	}
	assert_eq!(out, expect);
}

#[test]
fn compare_foreign_endianness_matches_reference() {
	for a in CORPUS {
		for b in CORPUS {
			let (ord, iord) = (reference_cmp(a, b, false), reference_cmp(a, b, true));
			let abe = swap16(&utf16(a));
			let bbe32 = swap32(&utf32(b));

			assert_eq!(compare::<Utf16<true>, Utf8>(&abe, b.as_bytes()), ord, "{a:?}/{b:?}");
			assert_eq!(compare::<Utf16<true>, Utf32<true>>(&abe, &bbe32), ord, "{a:?}/{b:?}");
			assert_eq!(
				compare_ignore_ascii_case::<Utf16<true>, Utf16<true>>(&abe, &swap16(&utf16(b))),
				iord,
				"{a:?}/{b:?} icase be/be"
			);
			assert_eq!(
				equals_ignore_ascii_case::<Utf16<true>, Utf8>(&abe, b.as_bytes()),
				iord == Ordering::Equal,
				"{a:?}/{b:?} ieq be/8"
			);
		}
	}
}

#[test]
fn foreign_ascii_case_pairs_fold() {
	// Byte-swapped ASCII is not lane-ASCII: the raw-unit shortcut must not
	// fire, and folding must happen on the decoded codepoint.
	let upper = [('A' as u16).swap_bytes()];
	let lower = [('a' as u16).swap_bytes()];
	assert!(equals_ignore_ascii_case::<Utf16<true>, Utf16<true>>(&upper, &lower));
	assert_eq!(
		compare_ignore_ascii_case::<Utf16<true>, Utf16<true>>(&upper, &lower),
		Ordering::Equal
	);
	// And raw units that *look* ASCII are not: 0x0041 raw is U+4100.
	let cjk_be = [0x0041u16]; // native value 0x4100 once swapped
	assert!(!equals::<Utf16<true>, Utf8>(&cjk_be, b"A"));
	assert!(!equals_ignore_ascii_case::<Utf16<true>, Utf8>(&cjk_be, b"a"));
}

// ------------------------------------------------------------------- bom --

#[test]
fn bom_detection_and_decode() {
	let s = "BOM test ασδφ 𝔴𝔬𝔯𝔨s fine with long ascii tails everywhere";
	let (u16n, u32n) = (utf16(s), utf32(s));

	// UTF-8 BOM.
	let mut bytes = vec![0xef, 0xbb, 0xbf];
	bytes.extend_from_slice(s.as_bytes());
	assert_eq!(detect_bom(&bytes), (Bom::Utf8, 3));
	assert_eq!(from_bytes::<Utf16>(&bytes), u16n);

	// UTF-16, native and swapped byte order.
	let mut native: Vec<u8> = 0xfeffu16.to_ne_bytes().to_vec();
	native.extend(u16n.iter().flat_map(|u| u.to_ne_bytes()));
	assert_eq!(detect_bom(&native), (Bom::Utf16Native, 2));
	assert_eq!(from_bytes::<Utf8>(&native), s.as_bytes());

	let mut swapped: Vec<u8> = 0xfeffu16.swap_bytes().to_ne_bytes().to_vec();
	swapped.extend(u16n.iter().flat_map(|u| u.swap_bytes().to_ne_bytes()));
	assert_eq!(detect_bom(&swapped), (Bom::Utf16Swapped, 2));
	assert_eq!(from_bytes::<Utf8>(&swapped), s.as_bytes());

	// UTF-32, native and swapped byte order.
	let mut native: Vec<u8> = 0xfeffu32.to_ne_bytes().to_vec();
	native.extend(u32n.iter().flat_map(|u| u.to_ne_bytes()));
	assert_eq!(detect_bom(&native), (Bom::Utf32Native, 4));
	assert_eq!(from_bytes::<Utf8>(&native), s.as_bytes());

	let mut swapped: Vec<u8> = 0xfeffu32.swap_bytes().to_ne_bytes().to_vec();
	swapped.extend(u32n.iter().flat_map(|u| u.swap_bytes().to_ne_bytes()));
	assert_eq!(detect_bom(&swapped), (Bom::Utf32Swapped, 4));
	assert_eq!(from_bytes::<Utf8>(&swapped), s.as_bytes());

	// No BOM: treated as UTF-8.
	assert_eq!(detect_bom(s.as_bytes()), (Bom::None, 0));
	assert_eq!(from_bytes::<Utf32>(s.as_bytes()), u32n);
}

#[test]
fn utf32_ascii_then_astral_transition() {
	// 8 ASCII u32s followed by supplementary/astral codepoint and more ASCII.
	let mut input = vec![b'a' as u32; 8];
	input.extend_from_slice(&[
		0x1f600,
		b'b' as u32,
		b'c' as u32,
		b'd' as u32,
		b'e' as u32,
		b'f' as u32,
		b'g' as u32,
		b'h' as u32,
	]);
	// Pad to ensure SIMD loop boundary triggers.
	input.extend_from_slice(&[b'z' as u32; 32]);
	let expected = transcode::<Utf32, Utf8>(&input);
	let mut reference = Vec::new();
	for &cp in &input {
		let c = char::from_u32(cp).unwrap();
		let mut buf = [0u8; 4];
		reference.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
	}
	assert_eq!(expected, reference);
}

#[test]
fn encoding_from_bytes_associated_method() {
	let s = "naïve café 👋";
	let u16n = utf16(s);

	let mut bom_utf16_le: Vec<u8> = 0xfeffu16.to_le_bytes().to_vec();
	bom_utf16_le.extend(u16n.iter().flat_map(|u| u.to_le_bytes()));

	let decoded_string: String = Utf8::from_bytes(&bom_utf16_le);
	assert_eq!(decoded_string, s);

	let decoded_u16: Vec<u16> = <Utf16>::from_bytes(&bom_utf16_le);
	assert_eq!(decoded_u16, u16n);

	let decoded_u32: Vec<u32> = <Utf32>::from_bytes(&bom_utf16_le);
	assert_eq!(decoded_u32, utf32(s));

	let mut bom_utf16_be: Vec<u8> = 0xfeffu16.to_be_bytes().to_vec();
	bom_utf16_be.extend(u16n.iter().flat_map(|u| u.to_be_bytes()));

	let decoded_be: Vec<u16> = Utf16Be::from_bytes(&bom_utf16_be);
	assert_eq!(swap16(&decoded_be), u16n);
}

#[test]
fn text_trait_transcode_and_equality() {
	let s = "naïve café 👋";
	let u16n = utf16(s);
	let u32n = utf32(s);

	// Inferred return type transcoding from &str
	let vec_u16: Vec<u16> = s.transcode();
	assert_eq!(vec_u16, u16n);

	let vec_u32: Vec<u32> = s.transcode();
	assert_eq!(vec_u32, u32n);

	let vec_u8: Vec<u8> = s.transcode();
	assert_eq!(vec_u8, s.as_bytes());

	// Owned containers work as receivers (autoderef) and as operands.
	let back_to_string: String = u16n.transcode();
	assert_eq!(back_to_string, s);

	// Transcode into pre-allocated slice
	let len = u16n[..].transcoded_len::<u8>();
	assert_eq!(len, s.len());
	let mut buf = vec![0u8; len];
	let (read, written) = u16n[..].transcode_into(&mut buf);
	assert_eq!(read, u16n.len());
	assert_eq!(written, s.len());
	assert_eq!(buf, s.as_bytes());

	// Cross-encoding equality, borrowed or owned operands.
	assert!(s.eq_text(&u16n));
	assert!(u16n.eq_text(&u32n));
	assert!(s.eq_text(&u32n[..]));
	assert!(u16n.eq_text(s));

	let upper_ascii = "NAIVE CAFE 👋";
	let lower_ascii = "naive cafe 👋";
	assert!(lower_ascii.eq_text_ignore_ascii_case(upper_ascii));
	assert!(s.eq_text_ignore_ascii_case(&u16n));
}
