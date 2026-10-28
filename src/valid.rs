//! UTF-8 validation for byte streams cut at arbitrary points.
//!
//! [`valid_prefix`] is the longest well-formed prefix of a buffer, what
//! [`core::str::from_utf8`] calls its valid part; [`valid_text_prefix`] stops
//! it at the first C0 control or DEL, which a terminal executes rather than
//! prints. [`Utf8Decoder`] takes what they leave (a sequence cut off at the
//! end of a read, malformed bytes) one byte at a time.
//!
//! Sixteen bytes at a time are checked with the lookup algorithm of Keiser
//! and Lemire (*Validating UTF-8 In Less Than One Instruction Per Byte*,
//! 2021): three nibble-table lookups classify each byte with the one before
//! it, and a byte two or three after a three- or four-byte lead must be a
//! continuation. All-ASCII blocks skip the tables. [`UTF8_DFA`] finishes
//! the bytes past the last whole block, and finds the exact end of the valid
//! part in a block that failed.

use core::simd::{
	Simd,
	cmp::{SimdPartialEq, SimdPartialOrd},
	num::{SimdInt, SimdUint},
	simd_swizzle,
};

use crate::kernel::lookup_u8x16;

/// Bytes checked per step.
const BLOCK: usize = 16;

type Block = Simd<u8, BLOCK>;

/// The longest prefix of `bytes` that is well-formed UTF-8.
///
/// Well-formed is RFC 3629's: no overlong forms, surrogates or code points
/// past U+10FFFF. The prefix is the valid part [`core::str::from_utf8`]
/// reports: a sequence the bytes end in the middle of is left out, for the
/// next read to complete.
///
/// ```
/// assert_eq!(xutf::valid_prefix("héllo".as_bytes()), "héllo");
/// assert_eq!(xutf::valid_prefix(b"ab\xe4\xb8"), "ab"); // 中 cut short
/// assert_eq!(xutf::valid_prefix(b"ab\xffcd"), "ab");
/// ```
#[must_use]
pub fn valid_prefix(bytes: &[u8]) -> &str {
	prefix::<false>(bytes)
}

/// [`valid_prefix`] up to the first C0 control or DEL: the text a terminal
/// prints before it executes a control. C1 controls written as UTF-8
/// (`C2 80`..=`C2 9F`) are text here.
///
/// ```
/// assert_eq!(xutf::valid_text_prefix("中文\r\n".as_bytes()), "中文");
/// assert_eq!(xutf::valid_text_prefix(b"ab\x7fcd"), "ab");
/// ```
#[must_use]
pub fn valid_text_prefix(bytes: &[u8]) -> &str {
	prefix::<true>(bytes)
}

/// [`valid_prefix`], or with `TEXT` [`valid_text_prefix`].
#[inline(always)]
fn prefix<const TEXT: bool>(bytes: &[u8]) -> &str {
	// Every character that ends before `at` is well-formed.
	let mut at = 0;
	let mut prev = Block::splat(0);
	// The lanes of the last block that ended in the middle of a character.
	let mut cut = Block::splat(0);
	// Four blocks at a time while they hold well-formed text, then one at a
	// time from the four where something stops, to find its block.
	while let Some(chunk) = bytes.get(at..at + 4 * BLOCK) {
		let blocks: [Block; 4] = core::array::from_fn(|i| Block::from_slice(&chunk[i * BLOCK..]));
		if TEXT
			&& blocks
				.iter()
				.fold(Block::splat(0), |any, &block| any | controls(block))
				!= Block::splat(0)
		{
			break;
		}
		let Some(next) = check(&blocks, prev, cut) else {
			break;
		};
		(prev, cut) = (blocks[3], next);
		at += 4 * BLOCK;
	}
	let mut end = bytes.len();
	while let Some(chunk) = bytes.get(at..at + BLOCK) {
		let block = Block::from_slice(chunk);
		if TEXT && let Some(control) = controls(block).simd_ne(Block::splat(0)).first_set() {
			end = at + control;
			break;
		}
		let Some(next) = check(&[block], prev, cut) else {
			break;
		};
		(prev, cut) = (block, next);
		at += BLOCK;
	}
	// A character that straddles `at` starts at the last lead byte of the
	// three before it; no lead there means `at` is a character boundary.
	let start = (at.saturating_sub(3)..at)
		.rev()
		.find(|&i| bytes[i] >= 0xc0)
		.unwrap_or(at);
	let rest = &bytes[start..end];
	let rest = if TEXT {
		&rest[..rest
			.iter()
			.position(|&b| b < 0x20 || b == 0x7f)
			.unwrap_or(rest.len())]
	} else {
		rest
	};
	let valid = start + valid_len(rest);
	// SAFETY: the blocks before `start` passed the checks, which accept
	// exactly well-formed UTF-8 (the tests compare them with the standard
	// library), and `valid_len` measured the rest.
	unsafe { core::str::from_utf8_unchecked(&bytes[..valid]) }
}

/// One error flag of [`special_cases`]: lead or ASCII followed by a lead
/// or ASCII where a continuation must come.
const TOO_SHORT: u8 = 1 << 0;
/// ASCII followed by a continuation.
const TOO_LONG: u8 = 1 << 1;
/// `E0` followed by `80..=9F`.
const OVERLONG_3: u8 = 1 << 2;
/// Past U+10FFFF: `F4 90..` and above.
const TOO_LARGE: u8 = 1 << 3;
/// `ED` followed by `A0..=BF`.
const SURROGATE: u8 = 1 << 4;
/// `C0` or `C1`.
const OVERLONG_2: u8 = 1 << 5;
/// Past U+10FFFF with an `80..=8F` continuation: `F5 80..` and above.
const TOO_LARGE_1000: u8 = 1 << 6;
/// `F0` followed by `80..=8F`; shares a bit with [`TOO_LARGE_1000`].
const OVERLONG_4: u8 = 1 << 6;
/// A continuation followed by one: valid only as the third or fourth byte
/// of a character ([`continuations`]).
const TWO_CONTS: u8 = 1 << 7;
/// The flags the first byte's high nibble alone decides.
const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTS;

/// Flags by the high nibble of the byte before.
const BYTE_1_HIGH: Block = Simd::from_array([
	// 0___: ASCII.
	TOO_LONG,
	TOO_LONG,
	TOO_LONG,
	TOO_LONG,
	TOO_LONG,
	TOO_LONG,
	TOO_LONG,
	TOO_LONG,
	// 10__: a continuation.
	TWO_CONTS,
	TWO_CONTS,
	TWO_CONTS,
	TWO_CONTS,
	// 1100, 1101: a two-byte lead.
	TOO_SHORT | OVERLONG_2,
	TOO_SHORT,
	// 1110: a three-byte lead.
	TOO_SHORT | OVERLONG_3 | SURROGATE,
	// 1111: a four-byte lead or worse.
	TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4,
]);

/// Flags by the low nibble of the byte before.
const BYTE_1_LOW: Block = Simd::from_array([
	CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
	CARRY | OVERLONG_2,
	CARRY,
	CARRY,
	CARRY | TOO_LARGE,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
	CARRY | TOO_LARGE | TOO_LARGE_1000,
]);

/// Flags by the high nibble of the byte itself.
const BYTE_2_HIGH: Block = Simd::from_array([
	// 0___: ASCII.
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
	// 1000.
	TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
	// 1001.
	TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
	// 101_.
	TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
	TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
	// 11__: a lead.
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
	TOO_SHORT,
]);

/// Subtracted from a block, leaves nonzero lanes where its last three bytes
/// start a character the block ends in the middle of: a lead of two or more
/// bytes last, of three or more second to last, of four third to last.
const CUT: Block = Simd::from_array([
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0xdf, 0xbf,
]);

/// All ones in the lanes of C0 controls and DEL.
#[inline(always)]
fn controls(block: Block) -> Block {
	(block.simd_lt(Block::splat(0x20)) | block.simd_eq(Block::splat(0x7f)))
		.to_simd()
		.cast()
}

/// Checks `blocks` after `prev`, which left `cut` ([`CUT`]): `None` when
/// they are malformed, else the lanes the last one leaves cut.
#[inline(always)]
fn check<const N: usize>(blocks: &[Block; N], prev: Block, cut: Block) -> Option<Block> {
	let any = blocks
		.iter()
		.fold(Block::splat(0), |any, &block| any | block);
	if any.reduce_max() < 0x80 {
		// ASCII is valid unless it cuts the last character short.
		return (cut == Block::splat(0)).then_some(cut);
	}
	let mut error = Block::splat(0);
	let mut prev = prev;
	for &block in blocks {
		error |= special_cases(block, prev) ^ continuations(block, prev);
		prev = block;
	}
	(error == Block::splat(0)).then(|| prev.saturating_sub(CUT))
}

/// The error flags of each byte of `input` with the byte before it (the
/// last of `prev` before the first), all of whose bits but [`TWO_CONTS`]
/// mean an error.
#[inline(always)]
fn special_cases(input: Block, prev: Block) -> Block {
	let prev1: Block =
		simd_swizzle!(prev, input, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]);
	let nibble = Block::splat(0x0f);
	lookup_u8x16(BYTE_1_HIGH, prev1 >> 4)
		& lookup_u8x16(BYTE_1_LOW, prev1 & nibble)
		& lookup_u8x16(BYTE_2_HIGH, input >> 4)
}

/// [`TWO_CONTS`] where a byte of `input` must be a continuation after
/// another: two after a three- or four-byte lead, or three after a
/// four-byte one.
#[inline(always)]
fn continuations(input: Block, prev: Block) -> Block {
	let prev2: Block =
		simd_swizzle!(prev, input, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
	let prev3: Block =
		simd_swizzle!(prev, input, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]);
	// Only `111_____` stays at 0x80 or above, and `1111____` in the second.
	let third = prev2.saturating_sub(Block::splat(0xe0 - 0x80));
	let fourth = prev3.saturating_sub(Block::splat(0xf0 - 0x80));
	(third | fourth) & Block::splat(TWO_CONTS)
}

/// How many of `bytes` lead with well-formed UTF-8, scalar: [`UTF8_DFA`]
/// eight bytes at a time over ASCII.
fn valid_len(bytes: &[u8]) -> usize {
	const HIGH: u64 = u64::from_ne_bytes([0x80; 8]);
	// The state is a bit offset into the rows of `UTF8_DFA`; a shift only
	// reads its low six bits, so the bits above them are never cleared.
	let mut state = 0u64;
	let (chunks, remainder) = bytes.as_chunks::<8>();
	let mut rejected = false;
	for chunk in chunks {
		if state & 63 == UTF8_ACCEPT && u64::from_ne_bytes(*chunk) & HIGH == 0 {
			continue;
		}
		for &byte in chunk {
			state = UTF8_DFA[usize::from(byte)].wrapping_shr(state as u32);
		}
		if state & 63 == UTF8_REJECT {
			rejected = true;
			break;
		}
	}
	if !rejected {
		for &byte in remainder {
			state = UTF8_DFA[usize::from(byte)].wrapping_shr(state as u32);
		}
	}
	match state & 63 {
		UTF8_ACCEPT => bytes.len(),
		// The standard library finds where the error is.
		UTF8_REJECT => core::str::from_utf8(bytes).map_or_else(|error| error.valid_up_to(), str::len),
		// Valid but for a character the bytes end in the middle of: it starts
		// at the last byte that is not a continuation byte.
		_ => bytes
			.iter()
			.rposition(|&byte| byte & 0xc0 != 0x80)
			.unwrap_or_default(),
	}
}

/// [`UTF8_DFA`]'s accepting state: between characters.
const UTF8_ACCEPT: u64 = 0;
/// [`UTF8_DFA`]'s rejecting state, which it never leaves.
const UTF8_REJECT: u64 = 6;

/// A UTF-8 validating automaton as one row per byte value: the next state
/// from state `s` (a multiple of six) is bits `s..s + 6` of the byte's row,
/// one shift per byte and no branch, where mixed-length text costs
/// [`core::str::from_utf8`] a mispredicted branch every few characters.
/// Nine states: accept, reject, and seven mid-character ones that say which
/// continuation bytes may follow (RFC 3629's table: no overlongs, no
/// surrogates, nothing above U+10FFFF).
const UTF8_DFA: [u64; 256] = {
	/// One more continuation byte, any.
	const ONE: u64 = 2;
	/// Two more, any.
	const TWO: u64 = 3;
	/// After E0: A0..=BF, then one more.
	const E0: u64 = 4;
	/// After ED: 80..=9F, then one more.
	const ED: u64 = 5;
	/// After F0: 90..=BF, then two more.
	const F0: u64 = 6;
	/// Three more, any.
	const THREE: u64 = 7;
	/// After F4: 80..=8F, then two more.
	const F4: u64 = 8;
	const ACCEPT: u64 = UTF8_ACCEPT / 6;
	const REJECT: u64 = UTF8_REJECT / 6;
	let mut table = [0u64; 256];
	let mut byte = 0;
	while byte < 256 {
		let b = byte as u8;
		let any = matches!(b, 0x80..=0xbf);
		let mut row = 0;
		let mut state = 0;
		while state < 9 {
			let next = match state {
				ACCEPT => match b {
					0x00..=0x7f => ACCEPT,
					0xc2..=0xdf => ONE,
					0xe0 => E0,
					0xed => ED,
					0xe1..=0xef => TWO,
					0xf0 => F0,
					0xf1..=0xf3 => THREE,
					0xf4 => F4,
					_ => REJECT,
				},
				ONE if any => ACCEPT,
				TWO if any => ONE,
				E0 if matches!(b, 0xa0..=0xbf) => ONE,
				ED if matches!(b, 0x80..=0x9f) => ONE,
				F0 if matches!(b, 0x90..=0xbf) => TWO,
				THREE if any => TWO,
				F4 if matches!(b, 0x80..=0x8f) => TWO,
				_ => REJECT,
			};
			row |= (next * 6) << (state * 6);
			state += 1;
		}
		table[byte] = row;
		byte += 1;
	}
	table
};

/// What one byte did to a [`Utf8Decoder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Utf8Step {
	/// The byte completed a character.
	Char(char),
	/// The byte continued a character that is not complete yet.
	Pending,
	/// The byte cannot come here: what was pending (a maximal subpart, in
	/// Unicode's terms) stands for one U+FFFD, and a byte that cut a
	/// character short must be pushed again as the start of the next one
	/// ([`Utf8Decoder::push`]).
	Reject,
}

/// A character partly decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Partial {
	/// Continuation bytes still needed.
	remaining: u8,
	/// The range the next byte must fall in.
	lo:        u8,
	hi:        u8,
	/// The code point so far.
	code:      u32,
}

/// Decodes UTF-8 one byte at a time, holding a character cut off between
/// two reads: the slow path beside [`valid_prefix`], for malformed input and
/// the ends of reads.
///
/// Accepts exactly the well-formed sequences of RFC 3629 (no overlongs,
/// surrogates or code points past U+10FFFF; 8-bit C1 controls are
/// continuation bytes, so malformed). Malformed input is replaced as
/// Unicode recommends (maximal subparts), one U+FFFD each.
///
/// ```
/// use xutf::{Utf8Decoder, Utf8Step};
///
/// let mut decoder = Utf8Decoder::new();
/// assert_eq!(decoder.push(0xe4), Utf8Step::Pending);
/// assert_eq!(decoder.push(0xb8), Utf8Step::Pending);
/// assert_eq!(decoder.push(0xad), Utf8Step::Char('中'));
/// // `E4 41`: the `E4` is one replacement, and `41` starts over.
/// assert_eq!(decoder.push(0xe4), Utf8Step::Pending);
/// assert_eq!(decoder.push(0x41), Utf8Step::Reject);
/// assert_eq!(decoder.push(0x41), Utf8Step::Char('A'));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Utf8Decoder {
	/// The character being decoded, if any.
	partial: Option<Partial>,
}

impl Utf8Decoder {
	/// A decoder between characters.
	#[must_use]
	pub const fn new() -> Self {
		Self { partial: None }
	}

	/// Whether a character is partly decoded.
	#[must_use]
	pub const fn is_pending(&self) -> bool {
		self.partial.is_some()
	}

	/// Drops a partly decoded character, returning whether there was one:
	/// what reads as one U+FFFD when the stream moves on without it.
	pub const fn abandon(&mut self) -> bool {
		self.partial.take().is_some()
	}

	/// Takes `byte`. After [`Utf8Step::Reject`] for a byte that cut a
	/// pending character short ([`Utf8Decoder::is_pending`] before the
	/// push), push the same byte again: it may start the next character.
	pub const fn push(&mut self, byte: u8) -> Utf8Step {
		let Some(part) = self.partial else {
			let (remaining, lo, hi, code) = match byte {
				0x00..=0x7f => return Utf8Step::Char(byte as char),
				0xc2..=0xdf => (1, 0x80, 0xbf, byte & 0x1f),
				// The first continuation is narrower after E0 (overlongs),
				// ED (surrogates), F0 (overlongs) and F4 (past U+10FFFF).
				0xe0 => (2, 0xa0, 0xbf, byte & 0x0f),
				0xed => (2, 0x80, 0x9f, byte & 0x0f),
				0xe1..=0xef => (2, 0x80, 0xbf, byte & 0x0f),
				0xf0 => (3, 0x90, 0xbf, byte & 0x07),
				0xf4 => (3, 0x80, 0x8f, byte & 0x07),
				0xf1..=0xf3 => (3, 0x80, 0xbf, byte & 0x07),
				_ => return Utf8Step::Reject,
			};
			self.partial = Some(Partial { remaining, lo, hi, code: code as u32 });
			return Utf8Step::Pending;
		};
		if byte < part.lo || byte > part.hi {
			self.partial = None;
			return Utf8Step::Reject;
		}
		let code = (part.code << 6) | (byte & 0x3f) as u32;
		if part.remaining > 1 {
			self.partial = Some(Partial { remaining: part.remaining - 1, lo: 0x80, hi: 0xbf, code });
			return Utf8Step::Pending;
		}
		self.partial = None;
		// The ranges above admit no surrogate and nothing past U+10FFFF.
		Utf8Step::Char(match char::from_u32(code) {
			Some(c) => c,
			None => char::REPLACEMENT_CHARACTER,
		})
	}
}
