//! UTF-8 codec.
//!
//! ```text
//! 0xxxxxxx                            —  7 bits
//! 110xxxxx 10xxxxxx                   —  5+6 bits
//! 1110xxxx 10xxxxxx 10xxxxxx          —  4+6+6 bits
//! 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx —  3+6+6+6 bits
//! ```

use core::hint::assert_unchecked;

use crate::encoding::{Encoding, Kind};

/// UTF-8. Byte-oriented, so there is no foreign-endianness variant.
pub struct Utf8;

/// `count` one-bits at `offset` (`fill_bits`).
#[inline(always)]
const fn fill_bits(count: u32, offset: u32) -> u32 {
	((1u32 << count) - 1) << offset
}

/// Multi-byte scatter/gather mask: 3 codepoint bits in the lead byte slot,
/// 6 in each continuation slot (bytes ordered lead..tail, high to low).
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
const PDEP_MASKS: [u32; 5] = [
	0,
	0,
	0b00000000_00000000_00011111_00111111,
	0b00000000_00001111_00111111_00111111,
	0b00000111_00111111_00111111_00111111,
];

/// Encodes an `I`-byte sequence (`I` in `2..=4`), returning `I`.
#[inline(always)]
fn write<const I: usize>(cp: u32, out: &mut [u8]) -> usize {
	#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
	{
		// Scatter the codepoint bits, then OR in the continuation flags and
		// the lead-byte flag (positioned at byte I-1, the big end).
		let mut tmp = unsafe { core::arch::x86_64::_pdep_u32(cp, PDEP_MASKS[4]) };
		tmp |= 0x8080_8080 | (fill_bits(I as u32, 8 - I as u32) << (8 * (I - 1)));
		// Bytes lead..tail are tmp bytes I-1..0: emit via a byte swap.
		out[..I].copy_from_slice(&tmp.to_be_bytes()[4 - I..]);
		return I;
	}
	#[allow(
		unreachable_code,
		reason = "the BMI2 specialization returns before the portable fallback"
	)]
	{
		for (i, slot) in out[..I].iter_mut().enumerate() {
			let flag = if i == 0 {
				fill_bits(I as u32, 8 - I as u32)
			} else {
				0x80
			};
			let bits = if i == 0 { 7 - I as u32 } else { 6 };
			*slot = (flag | ((cp >> (6 * (I - 1 - i))) & fill_bits(bits, 0))) as u8;
		}
		I
	}
}

/// Decodes an `I`-byte sequence (`I` in `2..=4`). Short input consumes the
/// remainder and yields `0`.
#[inline(always)]
fn read<const I: usize>(input: &mut &[u8]) -> u32 {
	let s = *input;
	if s.len() < I {
		*input = &[];
		return 0;
	}
	*input = &s[I..];

	#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
	{
		// Assemble big-endian by hand (the C++ avoids BSWAP here to keep the
		// dependency chain off it before PEXT).
		let mut tmp = 0u32;
		for (i, &b) in s[..I].iter().enumerate() {
			tmp |= (b as u32) << (8 * (I - 1 - i));
		}
		return unsafe { core::arch::x86_64::_pext_u32(tmp, PDEP_MASKS[I]) };
	}
	#[allow(
		unreachable_code,
		reason = "the BMI2 specialization returns before the portable fallback"
	)]
	{
		let mut cp = ((s[0] & fill_bits(7 - I as u32, 0) as u8) as u32) << (6 * (I - 1));
		for (k, &b) in s[1..I].iter().enumerate() {
			cp |= ((b & 0x3f) as u32) << (6 * (I - 2 - k));
		}
		cp
	}
}

impl Encoding for Utf8 {
	type Unit = u8;

	const FOREIGN: bool = false;
	const KIND: Kind = Kind::Utf8;
	const MAX_UNITS: usize = 4;

	#[inline(always)]
	fn run_length(front: u8) -> usize {
		let mut n = (front >> 7) as usize + 1;
		n += (front >= 0b1110_0000) as usize;
		n += (front >= 0b1111_0000) as usize;
		// SAFETY: n is 1 + up to three 0/1 increments, and the last two imply
		// the first (front >= 0xE0 sets bit 7).
		unsafe { assert_unchecked((1..=Self::MAX_UNITS).contains(&n)) };
		n
	}

	#[inline(always)]
	fn encoded_length(cp: u32) -> usize {
		let mut n = 1;
		n += (cp >> 7 != 0) as usize;
		n += (cp >> (5 + 6) != 0) as usize;
		n += (cp >> (4 + 6 + 6) != 0) as usize;
		// SAFETY: monotone increments bounded by 4.
		unsafe { assert_unchecked((1..=Self::MAX_UNITS).contains(&n)) };
		n
	}

	#[inline(always)]
	fn encode(cp: u32, out: &mut [u8]) -> usize {
		if cp <= 0x7f {
			out[0] = cp as u8;
			return 1;
		}
		if (cp >> 6) <= fill_bits(5, 0) {
			write::<2>(cp, out)
		} else if (cp >> 12) <= fill_bits(4, 0) {
			write::<3>(cp, out)
		} else {
			write::<4>(cp, out)
		}
	}

	#[inline(always)]
	fn decode(input: &mut &[u8]) -> u32 {
		let front = input[0];
		if front < 0x80 {
			*input = &input[1..];
			return front as u32;
		}
		if front < 0b1110_0000 {
			read::<2>(input)
		} else if front < 0b1111_0000 {
			read::<3>(input)
		} else {
			read::<4>(input)
		}
	}
}
