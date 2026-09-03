//! Native-endian SIMD transcoders for the six cross-width encoding pairs.
//!
//! The generic converter remains the correctness backstop. These kernels add
//! wide ASCII paths, table-shuffled mixed-width blocks, and fixed
//! multibyte patterns inspired by simdutf. Unmatched blocks advance one scalar
//! codepoint and immediately retry SIMD, preserving permissive codec semantics.

use core::simd::{
	Mask, Select, Simd, ToBytes,
	cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd},
	num::SimdUint,
	simd_swizzle,
};

use crate::{Utf8, Utf16, Utf32, encoding::Encoding};

const ASCII_EXPAND_16: usize = 16;
const ASCII_EXPAND_32: usize = 16;
const ASCII_WIDE: usize = 32;

/// Loads `N` lanes from `data[at..]` without a bounds check.
///
/// # Safety
/// `at + N <= data.len()`.
#[inline(always)]
unsafe fn ld<T: core::simd::SimdElement, const N: usize>(data: &[T], at: usize) -> Simd<T, N> {
	debug_assert!(at + N <= data.len());
	// SAFETY: caller guarantees the range; unaligned reads are allowed.
	unsafe { core::ptr::read_unaligned(data.as_ptr().add(at).cast::<[T; N]>()) }.into()
}

/// Stores `N` lanes to `data[at..]` without a bounds check.
///
/// # Safety
/// `at + N <= data.len()`.
#[inline(always)]
unsafe fn st<T: core::simd::SimdElement, const N: usize>(data: &mut [T], at: usize, v: Simd<T, N>) {
	debug_assert!(at + N <= data.len());
	// SAFETY: caller guarantees the range; unaligned writes are allowed.
	unsafe { core::ptr::write_unaligned(data.as_mut_ptr().add(at).cast::<[T; N]>(), v.to_array()) }
}
#[inline(always)]
fn shuffle_u8x16(input: Simd<u8, 16>, indexes: [u8; 16]) -> Simd<u8, 16> {
	crate::kernel::shuffle_u8x16(input, indexes)
}

#[inline(always)]
fn scalar_step<F: Encoding, T: Encoding>(
	src: &[F::Unit],
	dst: &mut [T::Unit],
	read: &mut usize,
	written: &mut usize,
) -> bool {
	let mut rest = &src[*read..];
	if rest.is_empty() {
		return false;
	}
	let cp = F::decode(&mut rest);
	if dst.len() - *written < T::encoded_length(cp) {
		return false;
	}
	*written += T::encode(cp, &mut dst[*written..]);
	*read = src.len() - rest.len();
	true
}

#[inline(always)]
fn scalar_finish<F: Encoding, T: Encoding>(
	src: &[F::Unit],
	dst: &mut [T::Unit],
	mut read: usize,
	mut written: usize,
) -> (usize, usize) {
	while read < src.len() && scalar_step::<F, T>(src, dst, &mut read, &mut written) {}
	(read, written)
}

#[inline(always)]
fn ascii_u8_to_u16<const N: usize>(
	src: &[u8],
	dst: &mut [u16],
	i: usize,
	o: usize,
) -> Option<usize> {
	if src.len() - i < N || dst.len() - o < N {
		return None;
	}
	let v = Simd::<u8, N>::from_slice(&src[i..]);
	v.cast::<u16>().copy_to_slice(&mut dst[o..o + N]);
	Some(v.simd_gt(Simd::splat(0x7f)).first_set().unwrap_or(N))
}

#[inline(always)]
fn ascii_u8_to_u32(src: &[u8], dst: &mut [u32], i: usize, o: usize) -> Option<usize> {
	if src.len() - i < ASCII_EXPAND_32 || dst.len() - o < ASCII_EXPAND_32 {
		return None;
	}
	let v = Simd::<u8, ASCII_EXPAND_32>::from_slice(&src[i..]);
	v.cast::<u32>()
		.copy_to_slice(&mut dst[o..o + ASCII_EXPAND_32]);
	Some(
		v.simd_gt(Simd::splat(0x7f))
			.first_set()
			.unwrap_or(ASCII_EXPAND_32),
	)
}

/// End-of-codepoint index for the low twelve window bytes: bit `p` is set
/// when byte `p + 1` does not continue a sequence, i.e. byte `p` ends one.
#[inline(always)]
fn utf8_eoc_index(v: Simd<u8, 16>) -> usize {
	let cont = (v.simd_ge(Simd::splat(0x80)) & v.simd_lt(Simd::splat(0xc0))).to_bitmask();
	((!cont) >> 1) as usize & 0x0fff
}

/// Validates six one/two-byte sequences from a table-shuffled window.
#[inline(always)]
fn decode_utf8_1_2_row(v: Simd<u8, 16>, row: &crate::tables::Utf8OneTwo) -> Option<Simd<u16, 8>> {
	let words = Simd::<u16, 8>::from_le_bytes(shuffle_u8x16(v, row.shuffle));
	let low = words & Simd::splat(0x00ff);
	let high = words >> 8;
	let ascii = high.simd_eq(Simd::splat(0));
	let valid_ascii = low.simd_le(Simd::splat(0x7f));
	let valid_two = high.simd_ge(Simd::splat(0xc0))
		& high.simd_lt(Simd::splat(0xe0))
		& low.simd_ge(Simd::splat(0x80))
		& low.simd_lt(Simd::splat(0xc0));
	if !ascii.select(valid_ascii, valid_two).all() {
		return None;
	}
	Some((words & Simd::splat(0x007f)) | ((words & Simd::splat(0x1f00)) >> 2))
}

#[inline(always)]
fn decode_utf8_1_2(v: Simd<u8, 16>) -> Option<(Simd<u16, 8>, usize)> {
	let row = &crate::tables::UTF8_ONE_TWO[utf8_eoc_index(v)];
	if row.consumed == 0 {
		return None;
	}
	Some((decode_utf8_1_2_row(v, row)?, row.consumed as usize))
}

/// Validates four one-to-three-byte sequences from a table-shuffled window.
#[inline(always)]
fn decode_utf8_1_2_3_row(
	v: Simd<u8, 16>,
	row: &crate::tables::Utf8OneTwoThree,
) -> Option<Simd<u32, 4>> {
	// Lanes are [last, mid, lead, 0]; zero-fill marks missing bytes.
	let w = Simd::<u32, 4>::from_le_bytes(shuffle_u8x16(v, row.shuffle));
	let b0 = w & Simd::splat(0xff);
	let b1 = (w >> 8) & Simd::splat(0xff);
	let b2 = w >> 16;
	let has_mid = b1.simd_ne(Simd::splat(0));
	let has_hi = b2.simd_ne(Simd::splat(0));
	let cont0 = b0.simd_ge(Simd::splat(0x80)) & b0.simd_lt(Simd::splat(0xc0));
	let cont1 = b1.simd_ge(Simd::splat(0x80)) & b1.simd_lt(Simd::splat(0xc0));
	let lead2 = b1.simd_ge(Simd::splat(0xc0)) & b1.simd_lt(Simd::splat(0xe0));
	let lead3 = b2.simd_ge(Simd::splat(0xe0)) & b2.simd_lt(Simd::splat(0xf0));
	let valid = (!has_hi | lead3)
		& (!has_mid | (has_hi & cont1) | (!has_hi & lead2))
		& ((has_mid & cont0) | (!has_mid & b0.simd_le(Simd::splat(0x7f))));
	if !valid.all() {
		return None;
	}
	let low_mask = has_mid.select(Simd::splat(0x3f), Simd::splat(0x7f));
	let mid_mask = has_hi.select(Simd::splat(0x3f), Simd::splat(0x1f));
	Some((b0 & low_mask) | ((b1 & mid_mask) << 6) | ((b2 & Simd::splat(0x0f)) << 12))
}

/// Decodes four mixed one/two/three-byte sequences from a 16-byte window.
#[inline(always)]
fn decode_utf8_1_2_3(v: Simd<u8, 16>) -> Option<(Simd<u32, 4>, usize)> {
	let row = &crate::tables::UTF8_ONE_TWO_THREE[utf8_eoc_index(v)];
	if row.consumed == 0 {
		return None;
	}
	Some((decode_utf8_1_2_3_row(v, row)?, row.consumed as usize))
}

#[inline(always)]
fn decode_utf8_4(v: Simd<u8, 16>) -> Option<Simd<u32, 4>> {
	let lead: Simd<u8, 4> = simd_swizzle!(v, [0, 4, 8, 12]);
	if !lead.simd_ge(Simd::splat(0xf0)).all() {
		return None;
	}
	let second: Simd<u8, 4> = simd_swizzle!(v, [1, 5, 9, 13]);
	let third: Simd<u8, 4> = simd_swizzle!(v, [2, 6, 10, 14]);
	let fourth: Simd<u8, 4> = simd_swizzle!(v, [3, 7, 11, 15]);
	let cp = ((lead & Simd::splat(0x07)).cast::<u32>() << 18)
		| ((second & Simd::splat(0x3f)).cast::<u32>() << 12)
		| ((third & Simd::splat(0x3f)).cast::<u32>() << 6)
		| (fourth & Simd::splat(0x3f)).cast();
	cp.simd_gt(Simd::splat(0xffff)).all().then_some(cp)
}

#[inline(always)]
fn utf8_to_utf16(src: &[u8], dst: &mut [u16]) -> (usize, usize) {
	let (mut i, mut o) = (0, 0);
	while i < src.len() {
		let lead = src[i];
		if lead <= 0x7f {
			if src.len() - i >= 64
				&& dst.len() - o >= 64
				// SAFETY: the length checks cover the fixed kernel block.
				&& unsafe {
					crate::kernel::ascii_u8_to_u16(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 64;
				o += 64;
				// ASCII streak: stay in the block kernel while it holds.
				while src.len() - i >= 64
					&& dst.len() - o >= 64
					// SAFETY: the loop bounds cover the fixed kernel block.
					&& unsafe {
						crate::kernel::ascii_u8_to_u16(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
					} {
					i += 64;
					o += 64;
				}
				continue;
			}
			if let Some(n @ 1..) = ascii_u8_to_u16::<ASCII_EXPAND_16>(src, dst, i, o) {
				i += n;
				o += n;
				continue;
			}
			if src.len() - i >= 16 && dst.len() - o >= 16 {
				let v = Simd::<u8, 16>::from_slice(&src[i..]);
				if !v.simd_gt(Simd::splat(0x7f)).any() {
					v.cast::<u16>().copy_to_slice(&mut dst[o..o + 16]);
					i += 16;
					o += 16;
					continue;
				}
			}
			// Run end: flush an 8-unit ASCII tail if present, then fall
			// straight into the mixed streamer without a dispatch bounce.
			if src.len() - i >= 8 && dst.len() - o >= 8 {
				let v = Simd::<u8, 8>::from_slice(&src[i..]);
				if !v.simd_gt(Simd::splat(0x7f)).any() {
					v.cast::<u16>().copy_to_slice(&mut dst[o..o + 8]);
					i += 8;
					o += 8;
				}
			}
			if utf8_mixed_to_utf16(src, dst, &mut i, &mut o) {
				continue;
			}
		} else if lead < 0xe0 {
			if utf8_mixed_to_utf16(src, dst, &mut i, &mut o) {
				continue;
			}
		} else if lead < 0xf0 {
			if utf8_mixed_to_utf16(src, dst, &mut i, &mut o) {
				continue;
			}
			if src.len() - i >= 48
				&& dst.len() - o >= 16
				// SAFETY: the length checks cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_3x16_to_utf16(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 48;
				o += 16;
				continue;
			}
		} else {
			if src.len() - i >= 64
				&& dst.len() - o >= 32
				// SAFETY: the length checks cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_4x16_to_utf16(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 64;
				o += 32;
				continue;
			}
			if src.len() - i >= 16
				&& dst.len() - o >= 8
				&& let Some(cp) = decode_utf8_4(Simd::from_slice(&src[i..]))
			{
				let adjusted = cp - Simd::splat(0x10000);
				let hi = ((adjusted >> 10).cast::<u16>()) | Simd::splat(0xd800);
				let lo = (adjusted.cast::<u16>() & Simd::splat(0x03ff)) | Simd::splat(0xdc00);
				let pairs: Simd<u16, 8> = simd_swizzle!(hi, lo, [0, 4, 1, 5, 2, 6, 3, 7]);
				pairs.copy_to_slice(&mut dst[o..o + 8]);
				i += 16;
				o += 8;
				continue;
			}
		}
		if !scalar_step::<Utf8, Utf16>(src, dst, &mut i, &mut o) {
			break;
		}
	}
	(i, o)
}

#[inline(always)]
fn utf8_to_utf32(src: &[u8], dst: &mut [u32]) -> (usize, usize) {
	let (mut i, mut o) = (0, 0);
	while i < src.len() {
		let lead = src[i];
		if lead <= 0x7f {
			if src.len() - i >= 64
				&& dst.len() - o >= 64
				// SAFETY: the length checks cover the fixed kernel block.
				&& unsafe {
					crate::kernel::ascii_u8_to_u32(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 64;
				o += 64;
				// ASCII streak: stay in the block kernel while it holds.
				while src.len() - i >= 64
					&& dst.len() - o >= 64
					// SAFETY: the loop bounds cover the fixed kernel block.
					&& unsafe {
						crate::kernel::ascii_u8_to_u32(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
					} {
					i += 64;
					o += 64;
				}
				continue;
			}
			if let Some(n @ 1..) = ascii_u8_to_u32(src, dst, i, o) {
				i += n;
				o += n;
				continue;
			}
			if src.len() - i >= 8 && dst.len() - o >= 8 {
				let v = Simd::<u8, 8>::from_slice(&src[i..]);
				if !v.simd_gt(Simd::splat(0x7f)).any() {
					v.cast::<u32>().copy_to_slice(&mut dst[o..o + 8]);
					i += 8;
					o += 8;
				}
			}
			if utf8_mixed_to_utf32(src, dst, &mut i, &mut o) {
				continue;
			}
		} else if lead < 0xe0 {
			if utf8_mixed_to_utf32(src, dst, &mut i, &mut o) {
				continue;
			}
		} else if lead < 0xf0 {
			if utf8_mixed_to_utf32(src, dst, &mut i, &mut o) {
				continue;
			}
			if src.len() - i >= 48
				&& dst.len() - o >= 16
				// SAFETY: the length checks cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_3x16_to_utf32(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 48;
				o += 16;
				continue;
			}
		} else {
			if src.len() - i >= 64
				&& dst.len() - o >= 16
				// SAFETY: the length checks cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_4x16_to_utf32(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 64;
				o += 16;
				continue;
			}
			if src.len() - i >= 16
				&& dst.len() - o >= 4
				&& let Some(cp) = decode_utf8_4(Simd::from_slice(&src[i..]))
			{
				cp.copy_to_slice(&mut dst[o..o + 4]);
				i += 16;
				o += 4;
				continue;
			}
		}
		if !scalar_step::<Utf8, Utf32>(src, dst, &mut i, &mut o) {
			break;
		}
	}
	(i, o)
}

#[inline(always)]
fn pack_2(cp: Simd<u16, 8>) -> Simd<u8, 16> {
	let packed =
		((cp >> 6) | Simd::splat(0x00c0)) | (((cp & Simd::splat(0x003f)) | Simd::splat(0x0080)) << 8);
	packed.to_le_bytes()
}

#[inline(always)]
fn pack_utf16_1_2(v: Simd<u16, 8>, ascii_mask: usize) -> (Simd<u8, 16>, usize) {
	let row = crate::tables::UTF16_ONE_TWO[ascii_mask];
	let ascii = v.simd_le(Simd::splat(0x7f));
	let pairs = ascii.select(v, Simd::<u16, 8>::from_le_bytes(pack_2(v)));
	(shuffle_u8x16(pairs.to_le_bytes(), row.shuffle), row.written as usize)
}

#[inline(always)]
fn pack_4(cp: Simd<u32, 4>) -> Simd<u8, 16> {
	let packed = ((cp >> 18) | Simd::splat(0xf0))
		| ((((cp >> 12) & Simd::splat(0x3f)) | Simd::splat(0x80)) << 8)
		| ((((cp >> 6) & Simd::splat(0x3f)) | Simd::splat(0x80)) << 16)
		| (((cp & Simd::splat(0x3f)) | Simd::splat(0x80)) << 24);
	packed.to_le_bytes()
}

/// Interleaves four lane bits into the even positions of an 8-bit index.
#[inline(always)]
const fn spread_width_bits(mask: usize) -> usize {
	let mask = (mask | (mask << 2)) & 0x33;
	(mask | (mask << 1)) & 0x55
}

/// Encodes four codepoints (widths 1-4) into compacted UTF-8 bytes.
///
/// Callers pre-exclude values whose scalar encoding differs from the packed
/// forms (UTF-16 surrogate halves stay valid: they encode as three bytes in
/// both paths).
#[inline(always)]
fn pack_utf8_1_2_3_4(cp: Simd<u32, 4>) -> (Simd<u8, 16>, usize) {
	let above_ascii = cp.simd_gt(Simd::splat(0x7f));
	let above_two = cp.simd_gt(Simd::splat(0x07ff));
	let above_three = cp.simd_gt(Simd::splat(0xffff));
	// Per-lane width minus one in binary: bit0 = a^b^c, bit1 = b.
	let low_bits = above_ascii.to_bitmask() ^ above_two.to_bitmask() ^ above_three.to_bitmask();
	let widths = spread_width_bits(low_bits as usize)
		| (spread_width_bits(above_two.to_bitmask() as usize) << 1);
	let row = crate::tables::UTF8_FOUR[widths];

	let two =
		((cp >> 6) | Simd::splat(0xc0)) | (((cp & Simd::splat(0x3f)) | Simd::splat(0x80)) << 8);
	let three = ((cp >> 12) | Simd::splat(0xe0))
		| ((((cp >> 6) & Simd::splat(0x3f)) | Simd::splat(0x80)) << 8)
		| (((cp & Simd::splat(0x3f)) | Simd::splat(0x80)) << 16);
	let four = ((cp >> 18) | Simd::splat(0xf0))
		| ((((cp >> 12) & Simd::splat(0x3f)) | Simd::splat(0x80)) << 8)
		| ((((cp >> 6) & Simd::splat(0x3f)) | Simd::splat(0x80)) << 16)
		| (((cp & Simd::splat(0x3f)) | Simd::splat(0x80)) << 24);
	let words = above_two.select(above_three.select(four, three), two);
	let words = above_ascii.select(words, cp);
	(shuffle_u8x16(words.to_le_bytes(), row.shuffle), row.written as usize)
}

/// Packs eight BMP non-surrogate UTF-16 units into two compacted UTF-8
/// halves, using simdutf's `dup_even` construction: each unit expands into a
/// four-byte zip group `[ascii, cont, lead3, lead2/mid]` from which a
/// 256-row shuffle table picks the live bytes.
#[inline(always)]
fn pack_utf16_1_2_3(v: Simd<u16, 8>) -> (Simd<u8, 16>, usize, Simd<u8, 16>, usize) {
	crate::kernel::pack_utf16_1_2_3(v)
}
/// Converts a 64-byte block of one-to-three-byte sequences with only the
/// block-level structure check, mirroring simdutf's validate-once design.
///
/// The caller's continuation mask proves the
/// `continuations == leads-demand` identity, which pins every window's
/// widths to its lead bytes; garbage and straddling sequences fail it and
/// return `None`. Every table step below then runs without per-window range
/// validation: the scalar decoder's formulas match the table arithmetic, so
/// permissive output is identical (overlong and surrogate forms decode the
/// same way on both paths).
#[inline(always)]
unsafe fn utf8_block_unchecked_to_utf16(
	src: &[u8],
	dst: &mut [u16],
	at: usize,
	out: usize,
	cont: u64,
	leads: u64,
	three: u64,
) -> Option<(usize, usize)> {
	// A lead in the last byte (or a three-byte lead in the last two) makes
	// demands past the block that the shifts below truncate; reject so the
	// checked fallback re-windows the straddler.
	if leads & (1 << 63) != 0 || three & 0xc000_0000_0000_0000 != 0 {
		return None;
	}
	if (leads << 1) | (three << 2) != cont {
		return None;
	}
	let eoc = !cont >> 1;
	let (mut pos, mut o) = (0usize, 0usize);
	// A step may start at byte 48: its 16-byte load stays within the
	// classified block, while every non-ASCII table path consumes at
	// most 12 bytes.
	while pos <= 48 {
		let slice = eoc >> pos;
		// SAFETY: pos <= 48 and the caller's 64-byte block leaves the
		// 16-byte window in bounds.
		let w: Simd<u8, 16> = unsafe { ld(src, at + pos) };
		if (slice & 0xffff) == 0xffff {
			// The identity plus boundary alignment prove all sixteen bytes
			// are ASCII: any lead inside would zero its end-of-codepoint bit.
			// SAFETY: 16 output units fit the caller's 64-unit headroom.
			unsafe { st(dst, out + o, w.cast::<u16>()) };
			o += 16;
			pos += 16;
			continue;
		}
		if (slice & 0x0fff) == 0x0fff {
			// The structure identity proves these twelve endpoint-aligned bytes
			// are ASCII; the trailing lanes are permitted scratch output.
			// SAFETY: sixteen scratch output units fit the caller's headroom.
			unsafe { st(dst, out + o, w.cast::<u16>()) };
			o += 12;
			pos += 12;
			continue;
		}
		if (slice & 0x0fff) == 0x924 {
			// Two-continuation demands pin the leads to the three-byte
			// class; the constant advance keeps the stride chain free.
			let cp = crate::kernel::decode_utf8_3x4(w);
			// SAFETY: 4 output units fit the destination check.
			unsafe { st(dst, out + o, cp) };
			o += 4;
			pos += 12;
			continue;
		}
		let m = (slice & 0x0fff) as usize;
		let row = &crate::tables::UTF8_ONE_TWO_WIDE[m];
		if row.written != 0 {
			let cp_lo = crate::kernel::decode_utf8_1_2(w, row.lo);
			// SAFETY: eight scratch output units fit the caller's headroom.
			unsafe { st(dst, out + o, cp_lo) };
			if row.written > 8 {
				let cp_hi = crate::kernel::decode_utf8_1_2(w, row.hi);
				// SAFETY: the second eight-unit scratch store fits the headroom.
				unsafe { st(dst, out + o + 8, cp_hi) };
			}
			o += row.written as usize;
			pos += 64 - (m as u64).leading_zeros() as usize;
			continue;
		}
		let row = &crate::tables::UTF8_ONE_TWO_THREE[m];
		if row.consumed == 0 {
			break;
		}
		let w32 = Simd::<u32, 4>::from_le_bytes(shuffle_u8x16(w, row.shuffle));
		let b0 = w32 & Simd::splat(0xff);
		let b1 = (w32 >> 8) & Simd::splat(0xff);
		let b2 = w32 >> 16;
		let has_mid = b1.simd_ne(Simd::splat(0));
		let has_hi = b2.simd_ne(Simd::splat(0));
		let low_mask = has_mid.select(Simd::splat(0x3f), Simd::splat(0x7f));
		let mid_mask = has_hi.select(Simd::splat(0x3f), Simd::splat(0x1f));
		let cp = (b0 & low_mask) | ((b1 & mid_mask) << 6) | ((b2 & Simd::splat(0x0f)) << 12);
		// SAFETY: 4 output units fit the destination check.
		unsafe { st(dst, out + o, cp.cast::<u16>()) };
		o += 4;
		pos += row.consumed as usize;
	}
	(pos != 0).then_some((pos, o))
}

/// Streams 64-byte UTF-8 blocks of one-to-three-byte sequences.
///
/// Classification runs once per block: the range masks prove the
/// `continuations == leads-demand` identity, which pins every window's
/// widths to its lead bytes. The x86 compactor retains a maximal complete
/// prefix when the final sequence straddles the block; garbage falls back to
/// the checked window loop, so table steps run without per-window validation.
#[inline(always)]
fn utf8_blocks_to_utf16(src: &[u8], dst: &mut [u16], i: &mut usize, o: &mut usize) -> bool {
	let mut advanced = false;
	#[cfg(target_arch = "x86_64")]
	{
		// SAFETY: the remaining slices provide the reported source and
		// destination ranges.
		if let Some((consumed, written)) = unsafe {
			crate::x86::utf8_mixed_prefix_to_utf16(
				src.as_ptr().add(*i),
				src.len() - *i,
				dst.as_mut_ptr().add(*o),
				dst.len() - *o,
			)
		} {
			*i += consumed;
			*o += written;
			advanced |= consumed != 0;
		}
	}
	'blocks: while src.len() - *i >= 64 && dst.len() - *o >= 64 {
		// SAFETY: the loop condition covers one 64-byte block.
		let (cont, leads, three, has4) =
			// SAFETY: the loop condition covers one 64-byte block.
			unsafe { crate::kernel::classify_block_u8x64(src.as_ptr().add(*i)) };
		if has4 {
			break;
		}
		if cont == 0 {
			// No continuations: all-ASCII copies wide, lead-only garbage
			// falls back to the checked window loop.
			let v = Simd::<u8, 64>::from_slice(&src[*i..]);
			if v.simd_ge(Simd::splat(0x80)).any() {
				break;
			}
			v.cast::<u16>().copy_to_slice(&mut dst[*o..*o + 64]);
			*i += 64;
			*o += 64;
			advanced = true;
			continue 'blocks;
		}
		let eoc = !cont >> 1;
		if cont == 0xaaaa_aaaa_aaaa_aaaa {
			// Homogeneous two-byte runs stream in fixed 32-codepoint batches.
			let mut ran = false;
			while src.len() - *i >= 64
				&& dst.len() - *o >= 32
				// SAFETY: the loop bounds cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_2x32_to_utf16(src.as_ptr().add(*i), dst.as_mut_ptr().add(*o))
				} {
				*i += 64;
				*o += 32;
				ran = true;
			}
			if !ran {
				break 'blocks;
			}
			advanced = true;
			continue 'blocks;
		}
		if eoc & 0xffff_ffff_ffff == 0x9249_2492_4924 {
			// Homogeneous three-byte runs stream in fixed 16-codepoint batches.
			let mut ran = false;
			while src.len() - *i >= 48
				&& dst.len() - *o >= 16
				// SAFETY: the loop bounds cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_3x16_to_utf16(src.as_ptr().add(*i), dst.as_mut_ptr().add(*o))
				} {
				*i += 48;
				*o += 16;
				ran = true;
			}
			if !ran {
				break 'blocks;
			}
			advanced = true;
			continue 'blocks;
		}
		// The x86 implementation compacts arbitrary one-to-three-byte runs
		// directly from their end-of-codepoint mask. A terminal lead is left
		// for the next block so a valid straddle does not discard this block.
		// SAFETY: the enclosing loop provides a full 64-byte input block and
		// 64 UTF-16 output units; the classification masks describe that block.
		if let Some((consumed, written)) = unsafe {
			crate::kernel::utf8_block_to_utf16(
				src.as_ptr().add(*i),
				dst.as_mut_ptr().add(*o),
				cont,
				leads,
				three,
			)
		} {
			*i += consumed;
			*o += written;
			advanced = true;
			continue 'blocks;
		}
		// Validate once per block, then convert windows unchecked; the
		// checked loop below remains the garbage/straddle fallback.
		// SAFETY: the enclosing loop provides the full block and output
		// headroom required by the unchecked converter.
		let block = unsafe { utf8_block_unchecked_to_utf16(src, dst, *i, *o, cont, leads, three) };
		if let Some((consumed, written)) = block {
			*i += consumed;
			*o += written;
			advanced = true;
			continue 'blocks;
		}
		let mut pos = 0usize;
		// A step may start at byte 48: its 16-byte load stays within the
		// classified block, while every non-ASCII table path consumes at
		// most 12 bytes.
		while pos <= 48 {
			let slice = eoc >> pos;
			// SAFETY: pos <= 48 and the loop condition leaves 64 readable
			// bytes, so the 16-byte window is in bounds.
			let w: Simd<u8, 16> = unsafe { ld(src, *i + pos) };
			if (slice & 0xffff) == 0xffff {
				if w.simd_ge(Simd::splat(0x80)).any() {
					break;
				}
				// SAFETY: 16 output units fit the caller's 64-unit headroom.
				unsafe { st(dst, *o, w.cast::<u16>()) };
				*o += 16;
				pos += 16;
				continue;
			}
			if (slice & 0x0fff) == 0xfff {
				// Twelve one-byte sequences by boundary alignment, but this
				// loop also runs on blocks that failed the structure identity,
				// where a lead followed by a non-continuation ends a sequence
				// too; only genuine ASCII may take the raw widened store.
				if w.simd_ge(Simd::splat(0x80)).to_bitmask() & 0x0fff != 0 {
					break;
				}
				// Lanes past twelve are scratch (allowed past the written
				// count), mirroring simdutf's block fast path.
				// SAFETY: 16 output units fit the caller's 64-unit headroom.
				unsafe { st(dst, *o, w.cast::<u16>()) };
				*o += 12;
				pos += 12;
				continue;
			}
			if (slice & 0x0fff) == 0x924 {
				let a: Simd<u8, 4> = simd_swizzle!(w, [0, 3, 6, 9]);
				if !(a.simd_ge(Simd::splat(0xe0)) & a.simd_lt(Simd::splat(0xf0))).all() {
					break;
				}
				let cp = crate::kernel::decode_utf8_3x4(w);
				// SAFETY: 4 output units fit the destination check.
				unsafe { st(dst, *o, cp) };
				*o += 4;
				pos += 12;
				continue;
			}
			if (slice & 0xffff) == 0xaaaa {
				let lead: Simd<u8, 8> = simd_swizzle!(w, [0, 2, 4, 6, 8, 10, 12, 14]);
				if !(lead.simd_ge(Simd::splat(0xc0)) & lead.simd_lt(Simd::splat(0xe0))).all() {
					break;
				}
				let tail: Simd<u8, 8> = simd_swizzle!(w, [1, 3, 5, 7, 9, 11, 13, 15]);
				let cp =
					((lead & Simd::splat(0x1f)).cast::<u16>() << 6) | (tail & Simd::splat(0x3f)).cast();
				// SAFETY: 8 output units fit the destination check.
				unsafe { st(dst, *o, cp) };
				*o += 8;
				pos += 16;
				continue;
			}
			let m = (slice & 0x0fff) as usize;
			let row = &crate::tables::UTF8_ONE_TWO[m];
			if row.consumed != 0 {
				let Some(cp) = decode_utf8_1_2_row(w, row) else {
					break;
				};
				// SAFETY: 8 output units fit the destination check.
				unsafe { st(dst, *o, cp) };
				*o += 6;
				pos += row.consumed as usize;
				continue;
			}
			let row = &crate::tables::UTF8_ONE_TWO_THREE[m];
			if row.consumed == 0 {
				break;
			}
			let Some(cp) = decode_utf8_1_2_3_row(w, row) else {
				break;
			};
			// SAFETY: 4 output units fit the destination check.
			unsafe { st(dst, *o, cp.cast::<u16>()) };
			*o += 4;
			pos += row.consumed as usize;
		}
		if pos == 0 {
			break 'blocks;
		}
		*i += pos;
		advanced = true;
	}
	advanced
}

/// [`utf8_block_unchecked_to_utf16`] with UTF-32 output.
#[inline(always)]
unsafe fn utf8_block_unchecked_to_utf32(
	src: &[u8],
	dst: &mut [u32],
	at: usize,
	out: usize,
	cont: u64,
	leads: u64,
	three: u64,
) -> Option<(usize, usize)> {
	// A lead in the last byte (or a three-byte lead in the last two) makes
	// demands past the block that the shifts below truncate; reject so the
	// checked fallback re-windows the straddler.
	if leads & (1 << 63) != 0 || three & 0xc000_0000_0000_0000 != 0 {
		return None;
	}
	if (leads << 1) | (three << 2) != cont {
		return None;
	}
	let eoc = !cont >> 1;
	let (mut pos, mut o) = (0usize, 0usize);
	// A step may start at byte 48: its 16-byte load stays within the
	// classified block, while every non-ASCII table path consumes at
	// most 12 bytes.
	while pos <= 48 {
		let slice = eoc >> pos;
		// SAFETY: pos <= 48 and the caller's 64-byte block leaves the
		// 16-byte window in bounds.
		let w: Simd<u8, 16> = unsafe { ld(src, at + pos) };
		if (slice & 0xffff) == 0xffff {
			// The identity plus boundary alignment prove all sixteen bytes
			// are ASCII: any lead inside would zero its end-of-codepoint bit.
			// SAFETY: 16 output units fit the caller's 64-unit headroom.
			unsafe { st(dst, out + o, w.cast::<u32>()) };
			o += 16;
			pos += 16;
			continue;
		}
		if (slice & 0x0fff) == 0x0fff {
			// The structure identity proves these twelve endpoint-aligned bytes
			// are ASCII; the trailing lanes are permitted scratch output.
			// SAFETY: sixteen scratch output units fit the caller's headroom.
			unsafe { st(dst, out + o, w.cast::<u32>()) };
			o += 12;
			pos += 12;
			continue;
		}
		if (slice & 0x0fff) == 0x924 {
			// Two-continuation demands pin the leads to the three-byte
			// class; the constant advance keeps the stride chain free.
			let cp = crate::kernel::decode_utf8_3x4(w).cast::<u32>();
			// SAFETY: 4 output units fit the destination check.
			unsafe { st(dst, out + o, cp) };
			o += 4;
			pos += 12;
			continue;
		}
		let m = (slice & 0x0fff) as usize;
		let row = &crate::tables::UTF8_ONE_TWO_WIDE[m];
		if row.written != 0 {
			let cp_lo = crate::kernel::decode_utf8_1_2(w, row.lo);
			// SAFETY: eight scratch output units fit the caller's headroom.
			unsafe { st(dst, out + o, cp_lo.cast::<u32>()) };
			if row.written > 8 {
				let cp_hi = crate::kernel::decode_utf8_1_2(w, row.hi);
				// SAFETY: the second eight-unit scratch store fits the headroom.
				unsafe { st(dst, out + o + 8, cp_hi.cast::<u32>()) };
			}
			o += row.written as usize;
			pos += 64 - (m as u64).leading_zeros() as usize;
			continue;
		}
		let row = &crate::tables::UTF8_ONE_TWO_THREE[m];
		if row.consumed == 0 {
			break;
		}
		let w32 = Simd::<u32, 4>::from_le_bytes(shuffle_u8x16(w, row.shuffle));
		let b0 = w32 & Simd::splat(0xff);
		let b1 = (w32 >> 8) & Simd::splat(0xff);
		let b2 = w32 >> 16;
		let has_mid = b1.simd_ne(Simd::splat(0));
		let has_hi = b2.simd_ne(Simd::splat(0));
		let low_mask = has_mid.select(Simd::splat(0x3f), Simd::splat(0x7f));
		let mid_mask = has_hi.select(Simd::splat(0x3f), Simd::splat(0x1f));
		let cp = (b0 & low_mask) | ((b1 & mid_mask) << 6) | ((b2 & Simd::splat(0x0f)) << 12);
		// SAFETY: 4 output units fit the destination check.
		unsafe { st(dst, out + o, cp) };
		o += 4;
		pos += row.consumed as usize;
	}
	(pos != 0).then_some((pos, o))
}

/// [`utf8_blocks_to_utf16`] with UTF-32 output.
#[inline(always)]
fn utf8_blocks_to_utf32(src: &[u8], dst: &mut [u32], i: &mut usize, o: &mut usize) -> bool {
	let mut advanced = false;
	#[cfg(target_arch = "x86_64")]
	{
		// SAFETY: the remaining slices provide the reported source and
		// destination ranges.
		if let Some((consumed, written)) = unsafe {
			crate::x86::utf8_mixed_prefix_to_utf32(
				src.as_ptr().add(*i),
				src.len() - *i,
				dst.as_mut_ptr().add(*o),
				dst.len() - *o,
			)
		} {
			*i += consumed;
			*o += written;
			advanced |= consumed != 0;
		}
	}
	'blocks: while src.len() - *i >= 64 && dst.len() - *o >= 64 {
		// SAFETY: the loop condition covers one 64-byte block.
		let (cont, leads, three, has4) =
			// SAFETY: the loop condition covers one 64-byte block.
			unsafe { crate::kernel::classify_block_u8x64(src.as_ptr().add(*i)) };
		if has4 {
			break;
		}
		if cont == 0 {
			// No continuations: all-ASCII copies wide, lead-only garbage
			// falls back to the checked window loop.
			let v = Simd::<u8, 64>::from_slice(&src[*i..]);
			if v.simd_ge(Simd::splat(0x80)).any() {
				break;
			}
			v.cast::<u32>().copy_to_slice(&mut dst[*o..*o + 64]);
			*i += 64;
			*o += 64;
			advanced = true;
			continue 'blocks;
		}
		let eoc = !cont >> 1;
		if cont == 0xaaaa_aaaa_aaaa_aaaa {
			// Homogeneous two-byte runs stream in fixed 32-codepoint batches.
			let mut ran = false;
			while src.len() - *i >= 64
				&& dst.len() - *o >= 32
				// SAFETY: the loop bounds cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_2x32_to_utf32(src.as_ptr().add(*i), dst.as_mut_ptr().add(*o))
				} {
				*i += 64;
				*o += 32;
				ran = true;
			}
			if !ran {
				break 'blocks;
			}
			advanced = true;
			continue 'blocks;
		}
		if eoc & 0xffff_ffff_ffff == 0x9249_2492_4924 {
			// Homogeneous three-byte runs stream in fixed 16-codepoint batches.
			let mut ran = false;
			while src.len() - *i >= 48
				&& dst.len() - *o >= 16
				// SAFETY: the loop bounds cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf8_3x16_to_utf32(src.as_ptr().add(*i), dst.as_mut_ptr().add(*o))
				} {
				*i += 48;
				*o += 16;
				ran = true;
			}
			if !ran {
				break 'blocks;
			}
			advanced = true;
			continue 'blocks;
		}
		// The x86 implementation compacts arbitrary one-to-three-byte runs
		// directly from their end-of-codepoint mask.
		// SAFETY: the enclosing loop provides a full 64-byte input block and
		// 64 UTF-32 output units; the classification masks describe that block.
		if let Some(written) = unsafe {
			crate::kernel::utf8_block_to_utf32(
				src.as_ptr().add(*i),
				dst.as_mut_ptr().add(*o),
				cont,
				leads,
				three,
			)
		} {
			*i += 64;
			*o += written;
			advanced = true;
			continue 'blocks;
		}
		// Validate once per block, then convert windows unchecked; the
		// checked loop below remains the garbage/straddle fallback.
		// SAFETY: the enclosing loop provides the full block and output
		// headroom required by the unchecked converter.
		let block = unsafe { utf8_block_unchecked_to_utf32(src, dst, *i, *o, cont, leads, three) };
		if let Some((consumed, written)) = block {
			*i += consumed;
			*o += written;
			advanced = true;
			continue 'blocks;
		}
		let mut pos = 0usize;
		// A step may start at byte 48: its 16-byte load stays within the
		// classified block, while every non-ASCII table path consumes at
		// most 12 bytes.
		while pos <= 48 {
			let slice = eoc >> pos;
			// SAFETY: pos <= 48 and the loop condition leaves 64 readable
			// bytes, so the 16-byte window is in bounds.
			let w: Simd<u8, 16> = unsafe { ld(src, *i + pos) };
			if (slice & 0xffff) == 0xffff {
				if w.simd_ge(Simd::splat(0x80)).any() {
					break;
				}
				// SAFETY: 16 output units fit the caller's 64-unit headroom.
				unsafe { st(dst, *o, w.cast::<u32>()) };
				*o += 16;
				pos += 16;
				continue;
			}
			if (slice & 0x0fff) == 0xfff {
				// See the UTF-16 loop: boundary alignment alone does not prove
				// ASCII once the block-level identity has failed.
				if w.simd_ge(Simd::splat(0x80)).to_bitmask() & 0x0fff != 0 {
					break;
				}
				// Lanes past twelve are scratch (allowed past the written
				// count), mirroring simdutf's block fast path.
				// SAFETY: 16 output units fit the caller's 64-unit headroom.
				unsafe { st(dst, *o, w.cast::<u32>()) };
				*o += 12;
				pos += 12;
				continue;
			}
			if (slice & 0x0fff) == 0x924 {
				let a: Simd<u8, 4> = simd_swizzle!(w, [0, 3, 6, 9]);
				if !(a.simd_ge(Simd::splat(0xe0)) & a.simd_lt(Simd::splat(0xf0))).all() {
					break;
				}
				let cp = crate::kernel::decode_utf8_3x4(w).cast::<u32>();
				// SAFETY: 4 output units fit the destination check.
				unsafe { st(dst, *o, cp) };
				*o += 4;
				pos += 12;
				continue;
			}
			if (slice & 0xffff) == 0xaaaa {
				let lead: Simd<u8, 8> = simd_swizzle!(w, [0, 2, 4, 6, 8, 10, 12, 14]);
				if !(lead.simd_ge(Simd::splat(0xc0)) & lead.simd_lt(Simd::splat(0xe0))).all() {
					break;
				}
				let tail: Simd<u8, 8> = simd_swizzle!(w, [1, 3, 5, 7, 9, 11, 13, 15]);
				let cp =
					((lead & Simd::splat(0x1f)).cast::<u32>() << 6) | (tail & Simd::splat(0x3f)).cast();
				// SAFETY: 8 output units fit the destination check.
				unsafe { st(dst, *o, cp) };
				*o += 8;
				pos += 16;
				continue;
			}
			let m = (slice & 0x0fff) as usize;
			let row = &crate::tables::UTF8_ONE_TWO[m];
			if row.consumed != 0 {
				let Some(cp) = decode_utf8_1_2_row(w, row) else {
					break;
				};
				// SAFETY: 8 output units fit the destination check.
				unsafe { st(dst, *o, cp.cast::<u32>()) };
				*o += 6;
				pos += row.consumed as usize;
				continue;
			}
			let row = &crate::tables::UTF8_ONE_TWO_THREE[m];
			if row.consumed == 0 {
				break;
			}
			let Some(cp) = decode_utf8_1_2_3_row(w, row) else {
				break;
			};
			// SAFETY: 4 output units fit the destination check.
			unsafe { st(dst, *o, cp) };
			*o += 4;
			pos += row.consumed as usize;
		}
		if pos == 0 {
			break 'blocks;
		}
		*i += pos;
		advanced = true;
	}
	advanced
}

/// Streams consecutive 16-byte mixed UTF-8 windows into UTF-16.
///
/// Well-formed one-to-three-byte content takes the block driver; the window
/// loop below backstops garbage, four-byte pockets, and tails with full
/// validation. Three consecutive pure-ASCII windows hand control back to the
/// caller's wide copy path.
#[inline(always)]
fn utf8_mixed_to_utf16(src: &[u8], dst: &mut [u16], i: &mut usize, o: &mut usize) -> bool {
	let mut advanced = utf8_blocks_to_utf16(src, dst, i, o);
	let mut ascii_windows = 0;
	while src.len() - *i >= 16 && dst.len() - *o >= 16 {
		let v = Simd::<u8, 16>::from_slice(&src[*i..]);
		if !v.simd_gt(Simd::splat(0x7f)).any() {
			ascii_windows += 1;
			if ascii_windows >= 3 {
				break;
			}
			v.cast::<u16>().copy_to_slice(&mut dst[*o..*o + 16]);
			*i += 16;
			*o += 16;
		} else if let Some((cp, consumed)) = decode_utf8_1_2(v) {
			ascii_windows = 0;
			cp.copy_to_slice(&mut dst[*o..*o + 8]);
			*i += consumed;
			*o += 6;
		} else if let Some((cp, consumed)) = decode_utf8_1_2_3(v) {
			ascii_windows = 0;
			cp.cast::<u16>().copy_to_slice(&mut dst[*o..*o + 4]);
			*i += consumed;
			*o += 4;
		} else {
			break;
		}
		advanced = true;
	}
	advanced
}

/// [`utf8_mixed_to_utf16`] with UTF-32 output.
#[inline(always)]
fn utf8_mixed_to_utf32(src: &[u8], dst: &mut [u32], i: &mut usize, o: &mut usize) -> bool {
	let mut advanced = utf8_blocks_to_utf32(src, dst, i, o);
	let mut ascii_windows = 0;
	while src.len() - *i >= 16 && dst.len() - *o >= 16 {
		let v = Simd::<u8, 16>::from_slice(&src[*i..]);
		if !v.simd_gt(Simd::splat(0x7f)).any() {
			ascii_windows += 1;
			if ascii_windows >= 3 {
				break;
			}
			v.cast::<u32>().copy_to_slice(&mut dst[*o..*o + 16]);
			*i += 16;
			*o += 16;
		} else if let Some((cp, consumed)) = decode_utf8_1_2(v) {
			ascii_windows = 0;
			cp.cast::<u32>().copy_to_slice(&mut dst[*o..*o + 8]);
			*i += consumed;
			*o += 6;
		} else if let Some((cp, consumed)) = decode_utf8_1_2_3(v) {
			ascii_windows = 0;
			cp.copy_to_slice(&mut dst[*o..*o + 4]);
			*i += consumed;
			*o += 4;
		} else {
			break;
		}
		advanced = true;
	}
	advanced
}

/// Concatenates two truncating-narrowed eight-unit vectors into 16 bytes.
#[inline(always)]
fn narrow_16(a: Simd<u16, 8>, b: Simd<u16, 8>) -> Simd<u8, 16> {
	simd_swizzle!(a.cast::<u8>(), b.cast::<u8>(), [
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
	])
}

/// [`narrow_16`] for eight-unit UTF-32 vectors.
#[inline(always)]
fn narrow32_16(a: Simd<u32, 8>, b: Simd<u32, 8>) -> Simd<u8, 16> {
	simd_swizzle!(a.cast::<u8>(), b.cast::<u8>(), [
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
	])
}

/// Maxes of both eight-unit halves through one vector-to-scalar transfer:
/// pairwise folds leave the left maximum in the low sixteen bits and the
/// right maximum in the next sixteen, so callers classify each half without
/// paying a second reduction.
#[inline(always)]
fn max_pair_16(left: Simd<u16, 8>, right: Simd<u16, 8>) -> u32 {
	#[cfg(target_arch = "aarch64")]
	// SAFETY: AArch64 mandates NEON; the folds operate on vector values only.
	unsafe {
		use core::arch::aarch64::*;
		let pair_max = vpmaxq_u16(left.into(), right.into());
		let quad_max = vpmaxq_u16(pair_max, pair_max);
		let octet_max = vpmaxq_u16(quad_max, quad_max);
		vgetq_lane_u32::<0>(vreinterpretq_u32_u16(octet_max))
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		let hi = (left.reduce_max() as u32) | ((right.reduce_max() as u32) << 16);
		hi
	}
}

/// [`max_pair_16`] for eight-unit UTF-32 vectors: `max(a)` lands in the low
/// thirty-two bits and `max(b)` in the high thirty-two.
#[inline(always)]
fn max_pair_32(a: Simd<u32, 8>, b: Simd<u32, 8>) -> u64 {
	#[cfg(target_arch = "aarch64")]
	// SAFETY: AArch64 mandates NEON; the folds operate on vector values only.
	unsafe {
		use core::arch::aarch64::*;
		let a0: Simd<u32, 4> = simd_swizzle!(a, [0, 1, 2, 3]);
		let a1: Simd<u32, 4> = simd_swizzle!(a, [4, 5, 6, 7]);
		let b0: Simd<u32, 4> = simd_swizzle!(b, [0, 1, 2, 3]);
		let b1: Simd<u32, 4> = simd_swizzle!(b, [4, 5, 6, 7]);
		let pa = vpmaxq_u32(a0.into(), a1.into());
		let pb = vpmaxq_u32(b0.into(), b1.into());
		let q = vpmaxq_u32(pa, pb);
		let r = vpmaxq_u32(q, q);
		vgetq_lane_u64::<0>(vreinterpretq_u64_u32(r))
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		(a.reduce_max() as u64) | ((b.reduce_max() as u64) << 32)
	}
}

/// ASCII lane mask for two eight-unit vectors through one transfer: bit `k`
/// covers lane `k` of `a` and bit `k + 8` lane `k` of `b`.
#[inline(always)]
fn ascii_pair_mask(a: Simd<u16, 8>, b: Simd<u16, 8>) -> usize {
	let na = a.simd_le(Simd::splat(0x7f)).cast::<i8>().to_simd();
	let nb = b.simd_le(Simd::splat(0x7f)).cast::<i8>().to_simd();
	let joined: Simd<i8, 16> =
		simd_swizzle!(na, nb, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
	usize::try_from(Mask::<i8, 16>::from_simd(joined).to_bitmask()).unwrap()
}

/// UTF-16 to UTF-8 with a flat, simdutf-style classification loop.
///
/// Sixteen units load per iteration and every class test runs on registers
/// already in hand. Pure-width runs are recognized from byproducts of the
/// generic paths (ASCII lane masks, packed byte counts) and stream through
/// the fixed block kernels, which validate their own input, so no window is
/// ever probed twice.
#[inline(always)]
fn utf16_to_utf8(src: &[u16], dst: &mut [u8]) -> (usize, usize) {
	let accelerated: Option<(usize, usize)> = {
		#[cfg(target_arch = "x86_64")]
		{
			// SAFETY: the slices provide the reported source and destination
			// ranges.
			unsafe {
				crate::x86::utf16_to_utf8_prefix(src.as_ptr(), src.len(), dst.as_mut_ptr(), dst.len())
			}
		}
		#[cfg(not(target_arch = "x86_64"))]
		{
			None
		}
	};
	let (mut i, mut o) = if let Some(position) = accelerated {
		position
	} else {
		// SAFETY: the slices provide the reported source and destination ranges.
		unsafe {
			crate::kernel::utf16_fast_prefix(src.as_ptr(), src.len(), dst.as_mut_ptr(), dst.len())
		}
	};
	// Wide ASCII probes only pay off on runs beyond two windows, so the
	// streak fires once two consecutive ASCII windows are seen.
	let mut ascii_run = 0u32;
	while src.len() - i >= 16 && dst.len() - o >= 64 {
		// SAFETY: the loop condition covers sixteen input units.
		let v0: Simd<u16, 8> = unsafe { ld(src, i) };
		// SAFETY: the loop condition covers sixteen input units.
		let v1: Simd<u16, 8> = unsafe { ld(src, i + 8) };
		let both = max_pair_16(v0, v1);
		if both & 0xff80_ff80 == 0 {
			// SAFETY: 16 output bytes fit the 64-byte headroom.
			unsafe { st(dst, o, narrow_16(v0, v1)) };
			i += 16;
			o += 16;
			ascii_run += 1;
			if ascii_run >= 2 {
				while src.len() - i >= ASCII_WIDE && dst.len() - o >= ASCII_WIDE {
					// SAFETY: the loop condition covers the wide window.
					let wide: Simd<u16, ASCII_WIDE> = unsafe { ld(src, i) };
					if wide.simd_gt(Simd::splat(0x7f)).any() {
						break;
					}
					// SAFETY: the loop condition covers every written byte.
					unsafe { st(dst, o, wide.cast::<u8>()) };
					i += ASCII_WIDE;
					o += ASCII_WIDE;
				}
			}
			continue;
		}
		ascii_run = 0;
		let x0 = both & 0xffff;
		let x1 = both >> 16;
		if both & 0xf800_f800 == 0 {
			// Everything is one or two bytes wide.
			if x0 <= 0x7f {
				// ASCII low half: a free byte store plus one table pack.
				let m1 = v1.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
				let (b1, w1) = pack_utf16_1_2(v1, m1);
				// SAFETY: both stores fit the 64-byte headroom.
				unsafe {
					st(dst, o, v0.cast::<u8>());
					st(dst, o + 8, b1);
				}
				i += 16;
				o += 8 + w1;
				continue;
			}
			if x1 <= 0x7f {
				// ASCII high half: one table pack plus a free byte store.
				let m0 = v0.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
				let (b0, w0) = pack_utf16_1_2(v0, m0);
				// SAFETY: both stores fit the 64-byte headroom.
				unsafe {
					st(dst, o, b0);
					st(dst, o + w0, v1.cast::<u8>());
				}
				i += 16;
				o += w0 + 8;
				continue;
			}
			let pair = ascii_pair_mask(v0, v1);
			if pair == 0 {
				// SAFETY: 32 output bytes fit the 64-byte headroom.
				unsafe {
					st(dst, o, pack_2(v0));
					st(dst, o + 16, pack_2(v1));
				}
				i += 16;
				o += 32;
				// Pure two-byte streak; the kernel revalidates each block.
				while src.len() - i >= 32
					&& dst.len() - o >= 64
					// SAFETY: the loop bounds cover the fixed kernel block.
					&& unsafe {
						crate::kernel::utf16_2x32_to_utf8(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
					} {
					i += 32;
					o += 64;
				}
				continue;
			}
			let (b0, w0) = pack_utf16_1_2(v0, pair & 0xff);
			let (b1, w1) = pack_utf16_1_2(v1, pair >> 8);
			// SAFETY: both 16-byte scratch stores fit the 64-byte headroom.
			unsafe {
				st(dst, o, b0);
				st(dst, o + w0, b1);
			}
			i += 16;
			o += w0 + w1;
			// Mixed one/two-byte streak: text like Cyrillic stays in this
			// class for long stretches, so skip the outer re-classification
			// until the class breaks.
			while src.len() - i >= 16 && dst.len() - o >= 64 {
				// SAFETY: the loop condition covers sixteen input units.
				let u0: Simd<u16, 8> = unsafe { ld(src, i) };
				// SAFETY: the loop condition covers sixteen input units.
				let u1: Simd<u16, 8> = unsafe { ld(src, i + 8) };
				if max_pair_16(u0, u1) & 0xf800_f800 != 0 {
					break;
				}
				let pair = ascii_pair_mask(u0, u1);
				let (b0, w0) = pack_utf16_1_2(u0, pair & 0xff);
				let (b1, w1) = pack_utf16_1_2(u1, pair >> 8);
				// SAFETY: both 16-byte scratch stores fit the headroom.
				unsafe {
					st(dst, o, b0);
					st(dst, o + w0, b1);
				}
				i += 16;
				o += w0 + w1;
			}
			continue;
		}
		// At least one half is above U+07FF. Values below U+D800 can never
		// be surrogates, so the vector check runs only when a max demands it.
		let has_surrogate = x0.max(x1) >= 0xd800
			&& ((v0 & Simd::splat(0xf800)).simd_eq(Simd::splat(0xd800))
				| (v1 & Simd::splat(0xf800)).simd_eq(Simd::splat(0xd800)))
			.any();
		if !has_surrogate {
			if v0.simd_min(v1).reduce_min() >= 0x800 {
				// Pure three-byte window: jump straight into the streaming
				// kernel; its first pass converts this very window, whose
				// purity is already proven.
				while src.len() - i >= 16
					&& dst.len() - o >= 48
					// SAFETY: the loop bounds cover the fixed kernel block.
					&& unsafe {
						crate::kernel::utf16_3x16_to_utf8(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
					} {
					i += 16;
					o += 48;
				}
				continue;
			}
			if x0 <= 0x7f {
				// ASCII low half before heavy content: a free byte store
				// plus one dup_even pack keeps table work off ASCII units.
				let (lo, a, hi, b) = pack_utf16_1_2_3(v1);
				// SAFETY: all three scratch stores fit the 64-byte headroom.
				unsafe {
					st(dst, o, v0.cast::<u8>());
					st(dst, o + 8, lo);
					st(dst, o + 8 + a, hi);
				}
				i += 16;
				o += 8 + a + b;
				continue;
			}
			// Mixed window: convert the low half alone, picking the cheaper
			// pack when it stays below U+0800; the next window re-classifies
			// the rest, which frequently returns to a cheaper class.
			if x0 <= 0x7ff {
				let m0 = v0.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
				let (b0, w0) = pack_utf16_1_2(v0, m0);
				// SAFETY: the 16-byte scratch store fits the headroom.
				unsafe { st(dst, o, b0) };
				i += 8;
				o += w0;
				continue;
			}
			let (lo, a, hi, b) = pack_utf16_1_2_3(v0);
			// SAFETY: both 16-byte scratch stores fit the 64-byte headroom.
			unsafe {
				st(dst, o, lo);
				st(dst, o + a, hi);
			}
			i += 8;
			o += a + b;
			if a + b >= 23 {
				// At most one unit fell below three bytes: the run is
				// effectively pure, so try streaming the continuation.
				while src.len() - i >= 16
					&& dst.len() - o >= 48
					// SAFETY: the loop bounds cover the fixed kernel block.
					&& unsafe {
						crate::kernel::utf16_3x16_to_utf8(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
					} {
					i += 16;
					o += 48;
				}
			}
			continue;
		}
		// Surrogates present: convert the clean low half alone so pairs sit
		// at the window start, then stream pair-only blocks.
		let s0 = (v0 & Simd::splat(0xf800)).simd_eq(Simd::splat(0xd800));
		if !s0.any() {
			if x0 <= 0x7ff {
				let m0 = v0.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
				let (b0, w0) = pack_utf16_1_2(v0, m0);
				// SAFETY: the 16-byte scratch store fits the headroom.
				unsafe { st(dst, o, b0) };
				i += 8;
				o += w0;
				continue;
			}
			let (lo, a, hi, b) = pack_utf16_1_2_3(v0);
			// SAFETY: both 16-byte scratch stores fit the 64-byte headroom.
			unsafe {
				st(dst, o, lo);
				st(dst, o + a, hi);
			}
			i += 8;
			o += a + b;
			continue;
		}
		let hi4: Simd<u16, 4> = simd_swizzle!(v0, [0, 2, 4, 6]);
		let lo4: Simd<u16, 4> = simd_swizzle!(v0, [1, 3, 5, 7]);
		if ((hi4 & Simd::splat(0xfc00)).simd_eq(Simd::splat(0xd800))
			& (lo4 & Simd::splat(0xfc00)).simd_eq(Simd::splat(0xdc00)))
		.all()
		{
			let cp = ((hi4.cast::<u32>() - Simd::splat(0xd800)) << 10)
				+ (lo4.cast::<u32>() - Simd::splat(0xdc00))
				+ Simd::splat(0x10000);
			// SAFETY: 16 output bytes fit the 64-byte headroom.
			unsafe { st(dst, o, pack_4(cp)) };
			i += 8;
			o += 16;
			// Pure surrogate-pair streak; the kernel revalidates each block.
			while src.len() - i >= 32
				&& dst.len() - o >= 64
				// SAFETY: the loop bounds cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf16_4x16_to_utf8(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 32;
				o += 64;
			}
			continue;
		}
		if !scalar_step::<Utf16, Utf8>(src, dst, &mut i, &mut o) {
			return (i, o);
		}
	}
	scalar_finish::<Utf16, Utf8>(src, dst, i, o)
}

/// UTF-32 to UTF-8 with the same flat loop shape as [`utf16_to_utf8`].
///
/// Surrogate codepoints need no special casing below the BMP boundary: the
/// permissive scalar encoder emits them as three-byte forms, exactly like
/// the vector packers.
#[inline(always)]
fn utf32_to_utf8(src: &[u32], dst: &mut [u8]) -> (usize, usize) {
	let accelerated: Option<(usize, usize)> = {
		#[cfg(target_arch = "x86_64")]
		{
			// SAFETY: the slices provide the reported source and destination
			// ranges.
			unsafe {
				crate::x86::utf32_to_utf8_prefix(src.as_ptr(), src.len(), dst.as_mut_ptr(), dst.len())
			}
		}
		#[cfg(not(target_arch = "x86_64"))]
		{
			None
		}
	};
	let (mut i, mut o) = accelerated.unwrap_or((0, 0));
	while src.len() - i >= 16 && dst.len() - o >= 64 {
		// SAFETY: the loop condition covers sixteen input units.
		let v0: Simd<u32, 8> = unsafe { ld(src, i) };
		// SAFETY: the loop condition covers sixteen input units.
		let v1: Simd<u32, 8> = unsafe { ld(src, i + 8) };
		let both = max_pair_32(v0, v1);
		if both & 0xffff_ff80_ffff_ff80 == 0 {
			// SAFETY: 16 output bytes fit the 64-byte headroom.
			unsafe { st(dst, o, narrow32_16(v0, v1)) };
			i += 16;
			o += 16;
			// ASCII streak: confirm one more window before the wide blocks,
			// so a lone ASCII window never pays a wasted wide probe.
			if src.len() - i >= 16 && dst.len() - o >= 16 {
				// SAFETY: the length check covers sixteen input units.
				let u0: Simd<u32, 8> = unsafe { ld(src, i) };
				// SAFETY: the length check covers sixteen input units.
				let u1: Simd<u32, 8> = unsafe { ld(src, i + 8) };
				if max_pair_32(u0, u1) & 0xffff_ff80_ffff_ff80 == 0 {
					// SAFETY: the length check covers sixteen output bytes.
					unsafe { st(dst, o, narrow32_16(u0, u1)) };
					i += 16;
					o += 16;
					while src.len() - i >= ASCII_WIDE && dst.len() - o >= ASCII_WIDE {
						// SAFETY: the loop condition covers the wide window.
						let wide: Simd<u32, ASCII_WIDE> = unsafe { ld(src, i) };
						if wide.simd_gt(Simd::splat(0x7f)).any() {
							break;
						}
						// SAFETY: the loop condition covers every written byte.
						unsafe { st(dst, o, wide.cast::<u8>()) };
						i += ASCII_WIDE;
						o += ASCII_WIDE;
					}
				}
			}
			continue;
		}
		let x0 = (both & 0xffff_ffff) as u32;
		let x1 = (both >> 32) as u32;
		if x0 <= 0x7f {
			// ASCII low half: flush it for free, then convert the high half
			// alone so table work never covers ASCII units.
			// SAFETY: 8 output bytes fit the 64-byte headroom.
			unsafe { st(dst, o, v0.cast::<u8>()) };
			i += 8;
			o += 8;
			if x1 <= 0x7ff {
				let n1: Simd<u16, 8> = v1.cast();
				let m1 = n1.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
				let (b1, w1) = pack_utf16_1_2(n1, m1);
				// SAFETY: the 16-byte scratch store fits the headroom.
				unsafe { st(dst, o, b1) };
				i += 8;
				o += w1;
				continue;
			}
			if x1 <= 0xffff {
				let (lo, a, hi, b) = pack_utf16_1_2_3(v1.cast());
				// SAFETY: both 16-byte scratch stores fit the headroom.
				unsafe {
					st(dst, o, lo);
					st(dst, o + a, hi);
				}
				i += 8;
				o += a + b;
			}
			// A supplementary value right after an ASCII run: leave it at
			// the front of the next window for the four-byte tier below.
			continue;
		}
		if x1 <= 0x7f && x0 <= 0x7ff {
			// ASCII high half: one table pack plus a free byte store.
			let n0: Simd<u16, 8> = v0.cast();
			let m0 = n0.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
			let (b0, w0) = pack_utf16_1_2(n0, m0);
			// SAFETY: both stores fit the 64-byte headroom.
			unsafe {
				st(dst, o, b0);
				st(dst, o + w0, v1.cast::<u8>());
			}
			i += 16;
			o += w0 + 8;
			continue;
		}
		if both & 0xffff_f800_ffff_f800 == 0 {
			let n0: Simd<u16, 8> = v0.cast();
			let n1: Simd<u16, 8> = v1.cast();
			let m0 = n0.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
			let m1 = n1.simd_le(Simd::splat(0x7f)).to_bitmask() as usize;
			if (m0 | m1) == 0 {
				// SAFETY: 32 output bytes fit the 64-byte headroom.
				unsafe {
					st(dst, o, pack_2(n0));
					st(dst, o + 16, pack_2(n1));
				}
				i += 16;
				o += 32;
				// Pure two-byte streak; the kernel revalidates each block.
				while src.len() - i >= 16
					&& dst.len() - o >= 32
					// SAFETY: the loop bounds cover the fixed kernel block.
					&& unsafe {
						crate::kernel::utf32_2x16_to_utf8(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
					} {
					i += 16;
					o += 32;
				}
				continue;
			}
			let (b0, w0) = pack_utf16_1_2(n0, m0);
			let (b1, w1) = pack_utf16_1_2(n1, m1);
			// SAFETY: both 16-byte scratch stores fit the 64-byte headroom.
			unsafe {
				st(dst, o, b0);
				st(dst, o + w0, b1);
			}
			i += 16;
			o += w0 + w1;
			continue;
		}
		if both & 0xffff_0000_ffff_0000 == 0 {
			// Both halves stay in the BMP; surrogate values pass through as
			// three-byte forms exactly like the permissive scalar encoder.
			if v0.simd_min(v1).reduce_min() >= 0x800 {
				// Pure three-byte window: jump straight into the streaming
				// kernel; its first pass converts this very window, whose
				// purity is already proven.
				while src.len() - i >= 16
					&& dst.len() - o >= 48
					// SAFETY: the loop bounds cover the fixed kernel block.
					&& unsafe {
						crate::kernel::utf32_3x16_to_utf8(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
					} {
					i += 16;
					o += 48;
				}
				continue;
			}
			// Per-half register kernel: BMP semantics are identical to the
			// UTF-16 encoder here (surrogate values already pass through).
			// SAFETY: the enclosing loop supplies sixteen input units and
			// 64 output bytes; the masks restrict both halves to BMP values.
			let written = unsafe {
				crate::kernel::utf16_block_to_utf8(
					v0.cast(),
					v1.cast(),
					x0 as u16,
					x1 as u16,
					dst.as_mut_ptr().add(o),
				)
			};
			i += 16;
			o += written;
			continue;
		}
		if x0.max(x1) <= 0x10ffff && v0.simd_min(v1).reduce_min() >= 0x10000 {
			// SAFETY: 64 output bytes fit the 64-byte headroom.
			unsafe {
				st(dst, o, pack_4(simd_swizzle!(v0, [0, 1, 2, 3])));
				st(dst, o + 16, pack_4(simd_swizzle!(v0, [4, 5, 6, 7])));
				st(dst, o + 32, pack_4(simd_swizzle!(v1, [0, 1, 2, 3])));
				st(dst, o + 48, pack_4(simd_swizzle!(v1, [4, 5, 6, 7])));
			}
			i += 16;
			o += 64;
			// Pure four-byte streak; the kernel revalidates each block.
			while src.len() - i >= 16
				&& dst.len() - o >= 64
				// SAFETY: the loop bounds cover the fixed kernel block.
				&& unsafe {
					crate::kernel::utf32_4x16_to_utf8(src.as_ptr().add(i), dst.as_mut_ptr().add(o))
				} {
				i += 16;
				o += 64;
			}
			continue;
		}
		// Widths straddle the BMP boundary: take the low half through the
		// width-table packer, or one permissive scalar step when a lane is
		// beyond U+10FFFF.
		if x0 <= 0x10ffff {
			let (lo, lo_n) = pack_utf8_1_2_3_4(simd_swizzle!(v0, [0, 1, 2, 3]));
			let (hi, hi_n) = pack_utf8_1_2_3_4(simd_swizzle!(v0, [4, 5, 6, 7]));
			// SAFETY: both 16-byte scratch stores fit the 64-byte headroom.
			unsafe {
				st(dst, o, lo);
				st(dst, o + lo_n, hi);
			}
			i += 8;
			o += lo_n + hi_n;
			continue;
		}
		if !scalar_step::<Utf32, Utf8>(src, dst, &mut i, &mut o) {
			return (i, o);
		}
	}
	scalar_finish::<Utf32, Utf8>(src, dst, i, o)
}
#[inline(always)]
fn utf16_to_utf32(src: &[u16], dst: &mut [u32]) -> (usize, usize) {
	let (mut i, mut o) = (0, 0);
	while src.len() - i >= 16 && dst.len() - o >= 16 {
		let v = Simd::<u16, 16>::from_slice(&src[i..]);
		if (v & Simd::splat(0xfc00)).simd_ne(Simd::splat(0xd800)).all() {
			v.cast::<u32>().copy_to_slice(&mut dst[o..o + 16]);
			i += 16;
			o += 16;
		} else {
			break;
		}
	}
	scalar_finish::<Utf16, Utf32>(src, dst, i, o)
}

#[inline(always)]
fn utf32_to_utf16(src: &[u32], dst: &mut [u16]) -> (usize, usize) {
	let (mut i, mut o) = (0, 0);
	while src.len() - i >= 16 && dst.len() - o >= 16 {
		let v = Simd::<u32, 16>::from_slice(&src[i..]);
		if v.simd_le(Simd::splat(0xffff)).all() {
			v.cast::<u16>().copy_to_slice(&mut dst[o..o + 16]);
			i += 16;
			o += 16;
		} else {
			break;
		}
	}
	scalar_finish::<Utf32, Utf16>(src, dst, i, o)
}

#[inline(always)]
const unsafe fn as_units<T, U>(slice: &[T]) -> &[U] {
	// SAFETY: callers dispatch on sealed Encoding::KIND, which uniquely fixes
	// the unit primitive. Length is in units, so it remains unchanged.
	unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
}

#[inline(always)]
const unsafe fn as_units_mut<T, U>(slice: &mut [T]) -> &mut [U] {
	// SAFETY: same invariant as `as_units`, with unique access from `&mut`.
	unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast(), slice.len()) }
}

/// Dispatch table for one native-endian specialized pair.
#[inline(always)]
fn transcode_impl<F: Encoding, T: Encoding>(
	src: &[F::Unit],
	dst: &mut [T::Unit],
) -> Option<(usize, usize)> {
	match (F::KIND, T::KIND) {
		(crate::Kind::Utf8, crate::Kind::Utf16) => {
			// SAFETY: sealed kinds pin units to u8/u16.
			Some(utf8_to_utf16(unsafe { as_units(src) }, unsafe { as_units_mut(dst) }))
		},
		(crate::Kind::Utf8, crate::Kind::Utf32) => {
			// SAFETY: sealed kinds pin units to u8/u32.
			Some(utf8_to_utf32(unsafe { as_units(src) }, unsafe { as_units_mut(dst) }))
		},
		(crate::Kind::Utf16, crate::Kind::Utf8) => {
			// SAFETY: sealed kinds pin units to u16/u8.
			Some(utf16_to_utf8(unsafe { as_units(src) }, unsafe { as_units_mut(dst) }))
		},
		(crate::Kind::Utf16, crate::Kind::Utf32) => {
			// SAFETY: sealed kinds pin units to u16/u32.
			Some(utf16_to_utf32(unsafe { as_units(src) }, unsafe { as_units_mut(dst) }))
		},
		(crate::Kind::Utf32, crate::Kind::Utf8) => {
			// SAFETY: sealed kinds pin units to u32/u8.
			Some(utf32_to_utf8(unsafe { as_units(src) }, unsafe { as_units_mut(dst) }))
		},
		(crate::Kind::Utf32, crate::Kind::Utf16) => {
			// SAFETY: sealed kinds pin units to u32/u16.
			Some(utf32_to_utf16(unsafe { as_units(src) }, unsafe { as_units_mut(dst) }))
		},
		_ => None,
	}
}

/// Runs a native-endian preserving specialized pair, or returns `None` for
/// same-width/foreign pairs handled better by the generic converter.
///
/// Portable-SIMD codegen is frozen at compile time, so on x86-64 the whole
/// driver tree is compiled once per feature level and one detection branch
/// per call picks the widest: the AVX-512 copy lowers the 64-lane masks to
/// `vpmovb2m` and the table shuffles to `vpshufb`, which the baseline build
/// cannot express.
#[inline(always)]
pub fn transcode<F: Encoding, T: Encoding>(
	src: &[F::Unit],
	dst: &mut [T::Unit],
) -> Option<(usize, usize)> {
	if F::FOREIGN || T::FOREIGN {
		return None;
	}
	#[cfg(target_arch = "x86_64")]
	{
		#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2")]
		fn wide512<F: Encoding, T: Encoding>(
			src: &[F::Unit],
			dst: &mut [T::Unit],
		) -> Option<(usize, usize)> {
			transcode_impl::<F, T>(src, dst)
		}
		#[target_feature(enable = "avx2,bmi2")]
		fn wide256<F: Encoding, T: Encoding>(
			src: &[F::Unit],
			dst: &mut [T::Unit],
		) -> Option<(usize, usize)> {
			transcode_impl::<F, T>(src, dst)
		}
		if std::arch::is_x86_feature_detected!("avx512vbmi2")
			&& std::arch::is_x86_feature_detected!("avx512vl")
			&& std::arch::is_x86_feature_detected!("avx512bw")
			&& std::arch::is_x86_feature_detected!("bmi2")
		{
			// SAFETY: runtime detection proved every enabled feature.
			return unsafe { wide512::<F, T>(src, dst) };
		}
		if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi2")
		{
			// SAFETY: runtime detection proved every enabled feature.
			return unsafe { wide256::<F, T>(src, dst) };
		}
	}
	transcode_impl::<F, T>(src, dst)
}
