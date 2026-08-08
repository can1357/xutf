//! UTF-32 codec with optional foreign (byte-swapped) endianness.

use crate::encoding::{Encoding, Kind};

/// UTF-32. `FOREIGN` selects byte order opposite to native (see
/// [`Utf32Le`](crate::Utf32Le) / [`Utf32Be`](crate::Utf32Be)).
pub struct Utf32<const FOREIGN: bool = false>;

impl<const FOREIGN: bool> Encoding for Utf32<FOREIGN> {
	type Container = alloc::vec::Vec<u32>;
	type Unit = u32;

	const FOREIGN: bool = FOREIGN;
	const KIND: Kind = Kind::Utf32;
	const MAX_UNITS: usize = 1;

	#[inline(always)]
	fn run_length(_front: u32) -> usize {
		1
	}

	#[inline(always)]
	fn encoded_length(_cp: u32) -> usize {
		1
	}

	#[inline(always)]
	fn encode(cp: u32, out: &mut [u32]) -> usize {
		out[0] = if FOREIGN { cp.swap_bytes() } else { cp };
		1
	}

	#[inline(always)]
	fn decode(input: &mut &[u32]) -> u32 {
		let cp = input[0];
		*input = &input[1..];
		if FOREIGN { cp.swap_bytes() } else { cp }
	}

	#[inline(always)]
	fn decode_back(input: &mut &[u32]) -> u32 {
		let last = input.len() - 1;
		let cp = input[last];
		*input = &input[..last];
		if FOREIGN { cp.swap_bytes() } else { cp }
	}
}
