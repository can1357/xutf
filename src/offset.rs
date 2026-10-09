//! Codepoint-aligned offsets into encoded text ([`unit_offset`]), the inverse
//! of [`transcoded_len`](crate::transcoded_len).
//!
//! UTF-8 sources run 64-byte block kernels: one classification proves that a
//! block's continuation bytes sit exactly where its lead bytes demand them,
//! which makes every other byte a codepoint start, so a popcount counts the
//! block's output units and one bit select finds the codepoint holding unit
//! `n`. Bulk blocks advance by a fixed stride, carrying the demands of a
//! sequence that straddles into the next block, so no load waits on the
//! previous block's masks; the last bytes run boundary-anchored windows that
//! also handle a sequence the input truncates. Blocks that fail the proof
//! (malformed input) and every other encoding walk the permissive scalar codec
//! behind a SIMD ASCII run, so all paths agree with [`Encoding::decode`] on
//! any input.

use crate::{
	encoding::{Encoding, Kind, unit_preserving},
	kernel::{classify_block_u8x64, nth_set_bit, utf8_four_u8x64},
	simd::{NARROW, TRANS_WIDE, ascii_run},
	unit::Unit,
	utf8::Utf8,
};

/// Offset in `src` (in its units) of unit `n` of its transcoding to `T`,
/// rounded down to the start of the codepoint holding that unit; `src.len()`
/// when the transcoding is `n` units or shorter.
///
/// The inverse of [`transcoded_len`](crate::transcoded_len): the largest
/// codepoint boundary `k` with `transcoded_len::<F, T>(&src[..k]) <= n`.
/// `unit_offset::<Utf8, Utf32>` is the byte offset of `str::chars()` item `n`
/// (`char_indices().nth(n)`), `unit_offset::<Utf8, Utf16>` maps a UTF-16 index
/// (JavaScript, Cocoa, Win32) to a byte offset, and `unit_offset::<Utf16,
/// Utf8>` a byte offset to a UTF-16 index. Codepoint boundaries are the
/// permissive decoder's, so malformed input gets the same answer as walking
/// [`codepoints`](crate::codepoints).
///
/// # Example
/// ```
/// use xutf::{Utf8, Utf16, Utf32, unit_offset};
///
/// let s = "a😀b".as_bytes();
/// assert_eq!(unit_offset::<Utf8, Utf32>(s, 2), 5); // char 2, `b`, at byte 5
/// assert_eq!(unit_offset::<Utf8, Utf16>(s, 2), 1); // inside the pair: the emoji
/// assert_eq!(unit_offset::<Utf8, Utf16>(s, 4), s.len());
/// ```
pub fn unit_offset<F: Encoding, T: Encoding>(src: &[F::Unit], n: usize) -> usize {
	if const { unit_preserving::<F, T>() && F::KIND as u8 == Kind::Utf32 as u8 } {
		// Every UTF-32 unit is a codepoint.
		return n.min(src.len());
	}
	let (mut at, mut left) = (0, n);
	if const { F::KIND as u8 == Kind::Utf8 as u8 && !unit_preserving::<F, T>() } {
		// SAFETY: the sealed `Kind::Utf8` pins the unit type to `u8`.
		let src = unsafe { crate::native::as_units::<F::Unit, u8>(src) };
		if strides::<T>(src, &mut at, &mut left) || windows::<T>(src, &mut at, &mut left) {
			return at;
		}
	}
	walk::<F, T>(src, &mut at, &mut left, src.len());
	at
}

/// Bytes a stride block needs after its start: the block, and the rest of a
/// four-byte sequence starting in its last byte.
const STRIDE_SPAN: usize = 64 + 3;

/// Counts fixed 64-byte UTF-8 blocks from the boundary `at` while
/// [`STRIDE_SPAN`] bytes remain, spending the blocks' output units (UTF-16
/// when `T` is, else codepoints) from `left`. Returns `true` with `at` on the
/// codepoint holding the unit that does not fit; otherwise leaves `at` on the
/// last boundary reached.
#[inline(always)]
fn strides<T: Encoding>(src: &[u8], at: &mut usize, left: &mut usize) -> bool {
	let wide = const { T::KIND as u8 == Kind::Utf16 as u8 };
	let len = src.len();
	let mut base = *at;
	// From the block before: the continuations its last leads demand (bits
	// 0..=2), whether its byte 63 is a four-byte lead (and supplementary by
	// its own bits), and its starts and units for rewinding to a boundary.
	let (mut demand, mut four_last, mut plane_last) = (0u64, 0u64, 0u64);
	let (mut prev_starts, mut prev_units) = (0u64, 0u64);
	while len - base >= STRIDE_SPAN {
		// SAFETY: `base + 64 <= len`.
		let (cont, leads, three, has4) = unsafe { classify_block_u8x64(src.as_ptr().add(base)) };
		let (four, plane, next) = if has4 || (wide && four_last != 0) {
			// SAFETY: as above.
			unsafe { utf8_four_u8x64(src.as_ptr().add(base)) }
		} else {
			(0, 0, 0)
		};
		if ((leads << 1) | (three << 2) | (four << 3) | demand) != cont {
			// Malformed: resume from the last boundary with the scalar codec.
			rewind(at, left, base, demand, prev_starts, prev_units);
			if walk::<Utf8, T>(src, at, left, base + 64) {
				return true;
			}
			(base, demand, four_last, plane_last) = (*at, 0, 0, 0);
			continue;
		}
		let starts = !cont;
		// One bit per output unit: a supplementary codepoint's second UTF-16
		// unit sits on its second byte.
		let units = if wide {
			starts | (plane << 1) | plane_last | (((four << 1) | four_last) & next)
		} else {
			starts
		};
		let count = units.count_ones() as usize;
		if count > *left {
			let unit = nth_set_bit(units, *left as u32);
			// The second unit of a pair rounds down to its lead, which may end
			// the block before.
			*at = base + unit as usize - ((!starts >> unit) & 1) as usize;
			return true;
		}
		*left -= count;
		demand = (leads >> 63) | (three >> 62) | (four >> 61);
		(four_last, plane_last) = (four >> 63, plane >> 63);
		(prev_starts, prev_units) = (starts, units);
		base += 64;
	}
	rewind(at, left, base, demand, prev_starts, prev_units);
	false
}

/// Moves `at` to the boundary before the stride block at `base`: the block
/// itself, or the lead of a sequence straddling into it from the block before,
/// whose units go back to `left`.
#[inline(always)]
const fn rewind(
	at: &mut usize,
	left: &mut usize,
	base: usize,
	demand: u64,
	prev_starts: u64,
	prev_units: u64,
) {
	if demand == 0 {
		*at = base;
	} else {
		let lead = prev_starts.ilog2();
		*at = base - 64 + lead as usize;
		*left += (prev_units >> lead).count_ones() as usize;
	}
}

/// Counts UTF-8 from the boundary `at` to the end in 64-byte windows anchored
/// on boundaries, the last one overlapping its predecessor; a window stops at
/// a sequence running past it. Returns `true` with `at` on the codepoint
/// holding the unit `left` cannot pay for; otherwise leaves `at` on the last
/// boundary reached, short of the end only for a sequence the input
/// truncates.
#[inline(always)]
fn windows<T: Encoding>(src: &[u8], at: &mut usize, left: &mut usize) -> bool {
	let wide = const { T::KIND as u8 == Kind::Utf16 as u8 };
	let len = src.len();
	while len >= 64 && *at < len {
		// The final window overlaps the one before it; its bits below `at`
		// are already counted.
		let base = (*at).min(len - 64);
		let live = !0u64 << (*at - base);
		// SAFETY: `base + 64 <= len`.
		let (cont, leads, three, has4) = unsafe { classify_block_u8x64(src.as_ptr().add(base)) };
		let (four, plane, next) = if has4 {
			// SAFETY: as above.
			unsafe { utf8_four_u8x64(src.as_ptr().add(base)) }
		} else {
			(0, 0, 0)
		};
		let (leads, three, four, plane) = (leads & live, three & live, four & live, plane & live);
		if ((leads << 1) | (three << 2) | (four << 3)) != (cont & live) {
			if walk::<Utf8, T>(src, at, left, base + 64) {
				return true;
			}
			continue;
		}
		let starts = !cont & live;
		let spill = ((leads >> 63) | (three >> 62) | (four >> 61)) != 0;
		let end = if spill { starts.ilog2() } else { 64 };
		let counted = if spill { (1u64 << end) - 1 } else { !0 };
		let pairs = if wide {
			(plane << 1) | ((four << 1) & next)
		} else {
			0
		};
		let units = (starts | pairs) & counted;
		let count = units.count_ones() as usize;
		if count > *left {
			let unit = nth_set_bit(units, *left as u32);
			*at = base + unit as usize - ((!starts >> unit) & 1) as usize;
			return true;
		}
		*left -= count;
		*at = base + end as usize;
		if spill && base + 64 == len {
			// Truncated by the end of the input: the decoder's business.
			break;
		}
	}
	false
}

/// Decodes codepoints from the boundary `at` while `at < stop`, spending
/// their output units from `left`; returns `true` with `at` on the first
/// codepoint that does not fit.
#[inline(always)]
fn walk<F: Encoding, T: Encoding>(
	src: &[F::Unit],
	at: &mut usize,
	left: &mut usize,
	stop: usize,
) -> bool {
	while *at < stop {
		if const { !F::FOREIGN } && src[*at].to_u32() < 0x80 {
			// An ASCII unit is a one-unit codepoint in every encoding.
			let limit = (stop - *at).min(*left);
			let run = if limit >= TRANS_WIDE {
				ascii_run::<F::Unit, TRANS_WIDE>(&src[*at..], limit, |_, _| {})
			} else if limit >= NARROW {
				ascii_run::<F::Unit, NARROW>(&src[*at..], limit, |_, _| {})
			} else {
				0
			};
			*at += run;
			*left -= run;
			if *at == stop {
				break;
			}
		}
		let mut rest = &src[*at..];
		let cp = F::decode(&mut rest);
		let next = src.len() - rest.len();
		let units = if const { unit_preserving::<F, T>() } {
			next - *at
		} else {
			T::encoded_length(cp)
		};
		if units > *left {
			return true;
		}
		*left -= units;
		*at = next;
	}
	false
}
