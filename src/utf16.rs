//! UTF-16 codec with optional foreign (byte-swapped) endianness.

use core::hint::assert_unchecked;

use crate::encoding::{Encoding, Kind};

/// UTF-16. `FOREIGN` selects byte order opposite to native (see
/// [`Utf16Le`](crate::Utf16Le) / [`Utf16Be`](crate::Utf16Be)).
pub struct Utf16<const FOREIGN: bool = false>;

impl<const FOREIGN: bool> Encoding for Utf16<FOREIGN> {
	type Container = alloc::vec::Vec<u16>;
	type Unit = u16;

	const FOREIGN: bool = FOREIGN;
	const KIND: Kind = Kind::Utf16;
	const MAX_UNITS: usize = 2;

	#[inline(always)]
	fn run_length(front: u16) -> usize {
		let front = if FOREIGN { front.swap_bytes() } else { front };
		let n = 1 + ((front >> 10) == 0x36) as usize; // 0xD800 >> 10
		// SAFETY: 1 + bool.
		unsafe { assert_unchecked((1..=Self::MAX_UNITS).contains(&n)) };
		n
	}

	#[inline(always)]
	fn encoded_length(cp: u32) -> usize {
		1 + (cp >> 16 != 0) as usize
	}

	#[inline(always)]
	fn encode(cp: u32, out: &mut [u16]) -> usize {
		// Branchless: encode as a surrogate pair, then swap the first unit
		// for the BMP value if there is no high unit.
		let word = cp as u16;
		let has_high = cp != word as u32;
		let adj = cp.wrapping_sub(0x10000);
		let mut lo = 0xd800 | (adj >> 10) as u16;
		let hi = 0xdc00 | (adj as u16 & 0x3ff);
		if !has_high {
			lo = word;
		}
		let (lo, hi) = if FOREIGN {
			(lo.swap_bytes(), hi.swap_bytes())
		} else {
			(lo, hi)
		};
		// hi first: for BMP values both land on index 0 and lo wins.
		out[has_high as usize] = hi;
		out[0] = lo;
		1 + has_high as usize
	}

	#[inline(always)]
	fn decode(input: &mut &[u16]) -> u32 {
		let s = *input;
		let lo = if FOREIGN { s[0].swap_bytes() } else { s[0] };

		// High-surrogate flag on the first unit, with a trailing unit
		// available to pair with.
		let has_high = (lo & 0xfc00) == 0xd800 && s.len() != 1;
		let hi = {
			let raw = s[has_high as usize];
			if FOREIGN { raw.swap_bytes() } else { raw }
		};

		// Fold both surrogate biases into one constant; the single-unit case
		// requires no surrogate adjustment.
		let cp = if has_high {
			(hi as u32)
				.wrapping_sub(0xdc00)
				.wrapping_add(0x10000)
				.wrapping_sub(0xd800 << 10)
				.wrapping_add((lo as u32) << 10)
		} else {
			hi as u32
		};

		*input = &s[1 + has_high as usize..];
		cp
	}

	#[inline(always)]
	fn decode_back(input: &mut &[u16]) -> u32 {
		let s = *input;
		let last = s.len() - 1;
		let hi = if FOREIGN {
			s[last].swap_bytes()
		} else {
			s[last]
		};

		// Trailing low surrogate with a leading high surrogate to pair with;
		// anything else (including a lone surrogate) passes through.
		let has_pair = (hi & 0xfc00) == 0xdc00 && s.len() != 1 && {
			let raw = s[last - 1];
			let lo = if FOREIGN { raw.swap_bytes() } else { raw };
			(lo & 0xfc00) == 0xd800
		};
		if has_pair {
			let raw = s[last - 1];
			let lo = if FOREIGN { raw.swap_bytes() } else { raw };
			*input = &s[..last - 1];
			(hi as u32)
				.wrapping_sub(0xdc00)
				.wrapping_add(0x10000)
				.wrapping_sub(0xd800 << 10)
				.wrapping_add((lo as u32) << 10)
		} else {
			*input = &s[..last];
			hi as u32
		}
	}
}
