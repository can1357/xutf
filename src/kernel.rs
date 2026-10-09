//! Fixed-pattern transcode kernels expressed only with portable SIMD vectors.
//!
//! This module is the complete native-endian kernel implementation. It has no
//! architecture dispatch: LLVM lowers the same vector operations for every
//! target.

use core::simd::{
	Select, Simd, ToBytes,
	cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd},
	num::SimdUint,
	simd_swizzle,
};

#[inline(always)]
const unsafe fn load_u8x16(src: *const u8) -> Simd<u8, 16> {
	// SAFETY: callers guarantee sixteen readable bytes.
	unsafe { Simd::from_slice(core::slice::from_raw_parts(src, 16)) }
}

#[inline(always)]
const unsafe fn load_u16x8(src: *const u16) -> Simd<u16, 8> {
	// SAFETY: callers guarantee eight readable units.
	unsafe { Simd::from_slice(core::slice::from_raw_parts(src, 8)) }
}

#[inline(always)]
const unsafe fn load_u32x4(src: *const u32) -> Simd<u32, 4> {
	// SAFETY: callers guarantee four readable units.
	unsafe { Simd::from_slice(core::slice::from_raw_parts(src, 4)) }
}

#[inline(always)]
const unsafe fn store_u16x8(dst: *mut u16, value: Simd<u16, 8>) {
	// SAFETY: callers guarantee eight writable units.
	unsafe { dst.cast::<[u16; 8]>().write_unaligned(*value.as_array()) }
}

#[inline(always)]
const unsafe fn store_u16x16(dst: *mut u16, value: Simd<u16, 16>) {
	// SAFETY: callers guarantee sixteen writable units.
	unsafe { dst.cast::<[u16; 16]>().write_unaligned(*value.as_array()) }
}

#[inline(always)]
const unsafe fn store_u32x4(dst: *mut u32, value: Simd<u32, 4>) {
	// SAFETY: callers guarantee four writable units.
	unsafe { dst.cast::<[u32; 4]>().write_unaligned(*value.as_array()) }
}

#[inline(always)]
const unsafe fn store_u32x16(dst: *mut u32, value: Simd<u32, 16>) {
	// SAFETY: callers guarantee sixteen writable units.
	unsafe { dst.cast::<[u32; 16]>().write_unaligned(*value.as_array()) }
}

#[inline(always)]
const unsafe fn store_u8x16(dst: *mut u8, value: Simd<u8, 16>) {
	// SAFETY: callers guarantee sixteen writable bytes.
	unsafe { dst.cast::<[u8; 16]>().write_unaligned(*value.as_array()) }
}

#[inline(always)]
fn combine_u8x8(low: Simd<u8, 8>, high: Simd<u8, 8>) -> Simd<u8, 16> {
	simd_swizzle!(low, high, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn split_u8x16(value: Simd<u8, 16>) -> (Simd<u8, 8>, Simd<u8, 8>) {
	(
		simd_swizzle!(value, [0, 1, 2, 3, 4, 5, 6, 7]),
		simd_swizzle!(value, [8, 9, 10, 11, 12, 13, 14, 15]),
	)
}

#[inline(always)]
fn combine_u16x4(low: Simd<u16, 4>, high: Simd<u16, 4>) -> Simd<u16, 8> {
	simd_swizzle!(low, high, [0, 1, 2, 3, 4, 5, 6, 7])
}

#[inline(always)]
unsafe fn store_utf8_2x16(lead: Simd<u8, 16>, tail: Simd<u8, 16>, dst: *mut u8) {
	let bytes: Simd<u8, 32> = simd_swizzle!(lead, tail, [
		0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28,
		13, 29, 14, 30, 15, 31,
	]);
	// SAFETY: callers guarantee 32 writable bytes.
	unsafe { dst.cast::<[u8; 32]>().write_unaligned(*bytes.as_array()) }
}

#[inline(always)]
unsafe fn store_utf8_3x8(lead: Simd<u8, 8>, middle: Simd<u8, 8>, tail: Simd<u8, 8>, dst: *mut u8) {
	let lm: Simd<u8, 16> =
		simd_swizzle!(lead, middle, [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]);
	let tt: Simd<u8, 16> =
		simd_swizzle!(tail, tail, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
	let head: Simd<u8, 16> =
		simd_swizzle!(lm, tt, [0, 1, 16, 2, 3, 17, 4, 5, 18, 6, 7, 19, 8, 9, 20, 10]);
	let rest: Simd<u8, 8> = simd_swizzle!(lm, tt, [11, 21, 12, 13, 22, 14, 15, 23]);
	// SAFETY: callers guarantee 24 writable bytes.
	unsafe {
		dst.cast::<[u8; 16]>().write_unaligned(*head.as_array());
		dst.add(16)
			.cast::<[u8; 8]>()
			.write_unaligned(*rest.as_array());
	}
}

#[inline(always)]
unsafe fn store_utf8_3x16(
	lead: Simd<u8, 16>,
	middle: Simd<u8, 16>,
	tail: Simd<u8, 16>,
	dst: *mut u8,
) {
	#[cfg(target_arch = "aarch64")]
	{
		use core::arch::aarch64::{uint8x16x3_t, vst3q_u8};

		// SAFETY: callers guarantee 48 writable bytes.
		unsafe { vst3q_u8(dst, uint8x16x3_t(lead.into(), middle.into(), tail.into())) }
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		let (lead_lo, lead_hi) = split_u8x16(lead);
		let (middle_lo, middle_hi) = split_u8x16(middle);
		let (tail_lo, tail_hi) = split_u8x16(tail);
		// SAFETY: callers guarantee 48 writable bytes.
		unsafe {
			store_utf8_3x8(lead_lo, middle_lo, tail_lo, dst);
			store_utf8_3x8(lead_hi, middle_hi, tail_hi, dst.add(24));
		}
	}
}

/// Shuffles one byte vector using portable SIMD indices.
#[inline(always)]
pub fn shuffle_u8x16(input: Simd<u8, 16>, indexes: [u8; 16]) -> Simd<u8, 16> {
	#[cfg(target_arch = "aarch64")]
	{
		use core::arch::aarch64::{vld1q_u8, vqtbl1q_u8};

		// SAFETY: AArch64 mandates NEON and both values span sixteen bytes.
		unsafe { Simd::from(vqtbl1q_u8(input.into(), vld1q_u8(indexes.as_ptr()))) }
	}
	#[cfg(target_arch = "x86_64")]
	{
		#[target_feature(enable = "ssse3")]
		fn shuffle_ssse3(input: Simd<u8, 16>, indexes: [u8; 16]) -> Simd<u8, 16> {
			use core::arch::x86_64::{_mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128};

			let mut output = [0; 16];
			// SAFETY: each pointer spans exactly one 128-bit vector.
			unsafe {
				let input = _mm_loadu_si128(input.as_array().as_ptr().cast());
				let indexes = _mm_loadu_si128(indexes.as_ptr().cast());
				_mm_storeu_si128(output.as_mut_ptr().cast(), _mm_shuffle_epi8(input, indexes));
			}
			Simd::from_array(output)
		}
		if std::arch::is_x86_feature_detected!("ssse3") {
			// SAFETY: the feature test proved SSSE3; without it `swizzle_dyn`
			// lowers to a scalar byte loop that cripples every table path.
			return unsafe { shuffle_ssse3(input, indexes) };
		}
		input.swizzle_dyn(Simd::from_array(indexes))
	}
	#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
	{
		input.swizzle_dyn(Simd::from_array(indexes))
	}
}

#[inline(always)]
fn max_u16x8(value: Simd<u16, 8>) -> u32 {
	u32::from(value.reduce_max())
}

#[inline(always)]
fn sum_u16x8(value: Simd<u16, 8>) -> u32 {
	u32::from(value.reduce_sum())
}

/// Packs eight BMP non-surrogate UTF-16 units into two compact UTF-8 halves.
#[inline(always)]
pub fn pack_utf16_1_2_3(v: Simd<u16, 8>) -> (Simd<u8, 16>, usize, Simd<u8, 16>, usize) {
	let t2 = (v & Simd::splat(0x7f)) | ((v & Simd::splat(0x3f)) << 8) | Simd::splat(0x8000);
	let one_or_two = v.simd_le(Simd::splat(0x07ff));
	let s3 = (v >> 12) | ((v & Simd::splat(0x0fc0)) << 2) | Simd::splat(0xc0e0);
	let s4 = s3 ^ one_or_two.select(Simd::splat(0), Simd::splat(0x4000));
	let z0: Simd<u16, 8> = simd_swizzle!(t2, s4, [0, 8, 1, 9, 2, 10, 3, 11]);
	let z1: Simd<u16, 8> = simd_swizzle!(t2, s4, [4, 12, 5, 13, 6, 14, 7, 15]);
	const ONE_W: Simd<u16, 8> =
		Simd::from_array([0x0001, 0x0004, 0x0010, 0x0040, 0x0100, 0x0400, 0x1000, 0x4000]);
	const TWO_W: Simd<u16, 8> =
		Simd::from_array([0x0002, 0x0008, 0x0020, 0x0080, 0x0200, 0x0800, 0x2000, 0x8000]);
	let one = v.simd_le(Simd::splat(0x7f));
	let mask =
		sum_u16x8(one.select(ONE_W, Simd::splat(0)) | one_or_two.select(TWO_W, Simd::splat(0)))
			as usize;
	let row0 = &crate::tables::UTF8_ONE_TWO_THREE_PACK[mask & 0xff];
	let row1 = &crate::tables::UTF8_ONE_TWO_THREE_PACK[mask >> 8];
	(
		shuffle_u8x16(z0.to_le_bytes(), row0.shuffle),
		usize::from(row0.written),
		shuffle_u8x16(z1.to_le_bytes(), row1.shuffle),
		usize::from(row1.written),
	)
}

/// Decodes up to eight adjacent one-or-two-byte UTF-8 sequences selected by
/// `indexes`.
#[inline(always)]
pub fn decode_utf8_1_2(input: Simd<u8, 16>, indexes: [u8; 16]) -> Simd<u16, 8> {
	let words = Simd::<u16, 8>::from_le_bytes(shuffle_u8x16(input, indexes));
	(words & Simd::splat(0x007f)) | ((words & Simd::splat(0x1f00)) >> 2)
}

/// Decodes four adjacent three-byte UTF-8 sequences.
#[inline(always)]
pub fn decode_utf8_3x4(input: Simd<u8, 16>) -> Simd<u16, 4> {
	#[cfg(target_arch = "aarch64")]
	{
		use core::arch::aarch64::{
			vget_high_u8, vget_low_u8, vld1q_u8, vqtbl1q_u8, vreinterpret_u16_u8, vrev16_u8,
			vsli_n_u16,
		};

		// SAFETY: AArch64 mandates NEON and the shuffle reads vector values only.
		unsafe {
			let shuffle = vld1q_u8([0, 2, 3, 5, 6, 8, 9, 11, 1, 1, 4, 4, 7, 7, 10, 10].as_ptr());
			let permuted = vqtbl1q_u8(input.into(), shuffle);
			let low = vget_low_u8(permuted);
			let high = vget_high_u8(permuted);
			let middle = vreinterpret_u16_u8(high);
			let lead = vreinterpret_u16_u8(low);
			let middle_lead = vsli_n_u16::<6>(middle, lead);
			let tail = vreinterpret_u16_u8(vrev16_u8(low));
			Simd::from(vsli_n_u16::<6>(tail, middle_lead))
		}
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		let lead: Simd<u8, 4> = simd_swizzle!(input, [0, 3, 6, 9]);
		let middle: Simd<u8, 4> = simd_swizzle!(input, [1, 4, 7, 10]);
		let tail: Simd<u8, 4> = simd_swizzle!(input, [2, 5, 8, 11]);
		((lead & Simd::splat(0x0f)).cast::<u16>() << 12)
			| ((middle & Simd::splat(0x3f)).cast::<u16>() << 6)
			| (tail & Simd::splat(0x3f)).cast()
	}
}

/// Packs four 16-lane byte masks into one 64-bit bitmask, lane 0 in bit 0,
/// using simdutf's three-`vpaddq` reduction (NEON has no movemask, and the
/// generic `to_bitmask` extracts lanes one by one).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn bitmask64(
	m0: core::arch::aarch64::uint8x16_t,
	m1: core::arch::aarch64::uint8x16_t,
	m2: core::arch::aarch64::uint8x16_t,
	m3: core::arch::aarch64::uint8x16_t,
) -> u64 {
	use core::arch::aarch64::*;

	// SAFETY: operates only on vector values; AArch64 mandates NEON.
	unsafe {
		let bit: uint8x16_t = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
		let s0 = vpaddq_u8(vandq_u8(m0, bit), vandq_u8(m1, bit));
		let s1 = vpaddq_u8(vandq_u8(m2, bit), vandq_u8(m3, bit));
		let s2 = vpaddq_u8(s0, s1);
		vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(s2, s2)))
	}
}

/// Returns UTF-8 continuation, lead, and three-byte-lead masks for 64 bytes.
#[inline(always)]
pub unsafe fn classify_block_u8x64(src: *const u8) -> (u64, u64, u64, bool) {
	#[cfg(target_arch = "aarch64")]
	// SAFETY: callers guarantee 64 readable bytes; AArch64 mandates NEON.
	unsafe {
		use core::arch::aarch64::*;

		let a = vld1q_u8(src);
		let b = vld1q_u8(src.add(16));
		let c = vld1q_u8(src.add(32));
		let d = vld1q_u8(src.add(48));
		let prefix = vdupq_n_u8(0xc0);
		let continuation_bits = vdupq_n_u8(0x80);
		let continuation = bitmask64(
			vceqq_u8(vandq_u8(a, prefix), continuation_bits),
			vceqq_u8(vandq_u8(b, prefix), continuation_bits),
			vceqq_u8(vandq_u8(c, prefix), continuation_bits),
			vceqq_u8(vandq_u8(d, prefix), continuation_bits),
		);
		let leads = bitmask64(
			vcgeq_u8(a, prefix),
			vcgeq_u8(b, prefix),
			vcgeq_u8(c, prefix),
			vcgeq_u8(d, prefix),
		);
		let lead3 = vdupq_n_u8(0xe0);
		let three =
			bitmask64(vcgeq_u8(a, lead3), vcgeq_u8(b, lead3), vcgeq_u8(c, lead3), vcgeq_u8(d, lead3));
		let has4 =
			vmaxvq_u8(vcgeq_u8(vmaxq_u8(vmaxq_u8(a, b), vmaxq_u8(c, d)), vdupq_n_u8(0xf0))) != 0;
		(continuation, leads, three, has4)
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		#[cfg(target_arch = "x86_64")]
		if let Some(classified) = unsafe { crate::x86::classify_block(src) } {
			return classified;
		}
		// SAFETY: callers guarantee 64 readable bytes.
		let input = unsafe { Simd::<u8, 64>::from_slice(core::slice::from_raw_parts(src, 64)) };
		let continuation = (input & Simd::splat(0xc0))
			.simd_eq(Simd::splat(0x80))
			.to_bitmask();
		let leads = input.simd_ge(Simd::splat(0xc0)).to_bitmask();
		let three = input.simd_ge(Simd::splat(0xe0)).to_bitmask();
		(continuation, leads, three, input.simd_ge(Simd::splat(0xf0)).any())
	}
}

/// Masks of 64 bytes that decide which four-byte UTF-8 sequences decode to a
/// supplementary-plane codepoint (two UTF-16 units): the four-byte leads
/// (`>= 0xf0`), those of them with lead bits `0b111` set, and every byte
/// with bits `0b11_0000` set. A lead at `p` is supplementary when it is in
/// the second mask or byte `p + 1` is in the third, so overlong and `0xf8..`
/// leads stay one unit, as their permissive decode does.
#[inline(always)]
pub unsafe fn utf8_four_u8x64(src: *const u8) -> (u64, u64, u64) {
	#[cfg(target_arch = "aarch64")]
	// SAFETY: callers guarantee 64 readable bytes; AArch64 mandates NEON.
	unsafe {
		use core::arch::aarch64::*;

		let a = vld1q_u8(src);
		let b = vld1q_u8(src.add(16));
		let c = vld1q_u8(src.add(32));
		let d = vld1q_u8(src.add(48));
		let lead = vdupq_n_u8(0xf0);
		let four =
			bitmask64(vcgeq_u8(a, lead), vcgeq_u8(b, lead), vcgeq_u8(c, lead), vcgeq_u8(d, lead));
		let high = vdupq_n_u8(0x07);
		let plane =
			bitmask64(vtstq_u8(a, high), vtstq_u8(b, high), vtstq_u8(c, high), vtstq_u8(d, high));
		let low = vdupq_n_u8(0x30);
		let next = bitmask64(vtstq_u8(a, low), vtstq_u8(b, low), vtstq_u8(c, low), vtstq_u8(d, low));
		(four, four & plane, next)
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		// SAFETY: callers guarantee 64 readable bytes.
		let input = unsafe { Simd::<u8, 64>::from_slice(core::slice::from_raw_parts(src, 64)) };
		let four = input.simd_ge(Simd::splat(0xf0)).to_bitmask();
		let plane = (input & Simd::splat(0x07))
			.simd_ne(Simd::splat(0))
			.to_bitmask();
		let next = (input & Simd::splat(0x30))
			.simd_ne(Simd::splat(0))
			.to_bitmask();
		(four, four & plane, next)
	}
}

/// Position of set bit `k` of `x`, counting from the lowest (`k = 0`); `x`
/// must have more than `k` set bits.
#[inline]
pub fn nth_set_bit(x: u64, k: u32) -> u32 {
	debug_assert!(x.count_ones() > k);
	#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
	{
		// SAFETY: compiled only with BMI2 enabled.
		unsafe { core::arch::x86_64::_pdep_u64(1 << k, x) }.trailing_zeros()
	}
	#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
	{
		// Whole bytes by popcount, then at most seven bits by clearing.
		let (mut x, mut k, mut base) = (x, k, 0);
		loop {
			let ones = (x & 0xff).count_ones();
			if k < ones {
				break;
			}
			k -= ones;
			x >>= 8;
			base += 8;
		}
		for _ in 0..k {
			x &= x - 1;
		}
		base + x.trailing_zeros()
	}
}

/// Widens 64 ASCII bytes to UTF-16.
#[inline(always)]
pub unsafe fn ascii_u8_to_u16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: callers guarantee 64 readable bytes and 64 writable units.
	unsafe {
		let a = load_u8x16(src);
		let b = load_u8x16(src.add(16));
		let c = load_u8x16(src.add(32));
		let d = load_u8x16(src.add(48));
		if (a | b | c | d).simd_gt(Simd::splat(0x7f)).any() {
			return false;
		}
		store_u16x16(dst, a.cast());
		store_u16x16(dst.add(16), b.cast());
		store_u16x16(dst.add(32), c.cast());
		store_u16x16(dst.add(48), d.cast());
	}
	true
}

/// Widens 64 ASCII bytes to UTF-32.
#[inline(always)]
pub unsafe fn ascii_u8_to_u32(src: *const u8, dst: *mut u32) -> bool {
	// SAFETY: callers guarantee 64 readable bytes and 64 writable units.
	unsafe {
		let a = load_u8x16(src);
		let b = load_u8x16(src.add(16));
		let c = load_u8x16(src.add(32));
		let d = load_u8x16(src.add(48));
		if (a | b | c | d).simd_gt(Simd::splat(0x7f)).any() {
			return false;
		}
		store_u32x16(dst, a.cast());
		store_u32x16(dst.add(16), b.cast());
		store_u32x16(dst.add(32), c.cast());
		store_u32x16(dst.add(48), d.cast());
	}
	true
}

#[inline(always)]
fn deinterleave_utf8_2(a: Simd<u8, 16>, b: Simd<u8, 16>) -> (Simd<u8, 16>, Simd<u8, 16>) {
	(
		simd_swizzle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]),
		simd_swizzle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]),
	)
}

#[inline(always)]
fn decode_utf8_2x16(leads: Simd<u8, 16>, tails: Simd<u8, 16>) -> Simd<u16, 16> {
	((leads & Simd::splat(0x1f)).cast::<u16>() << 6) | (tails & Simd::splat(0x3f)).cast()
}

/// Converts 32 consecutive two-byte UTF-8 sequences to UTF-16.
#[inline(always)]
pub unsafe fn utf8_2x32_to_utf16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: callers guarantee 64 readable bytes and 32 writable units.
	unsafe {
		let (lead0, tail0) = deinterleave_utf8_2(load_u8x16(src), load_u8x16(src.add(16)));
		let (lead1, tail1) = deinterleave_utf8_2(load_u8x16(src.add(32)), load_u8x16(src.add(48)));
		let range = (lead0 - Simd::splat(0x80)) | (lead1 - Simd::splat(0x80));
		if range.simd_gt(Simd::splat(0x5f)).any() {
			return false;
		}
		store_u16x16(dst, decode_utf8_2x16(lead0, tail0));
		store_u16x16(dst.add(16), decode_utf8_2x16(lead1, tail1));
	}
	true
}

/// Converts 32 consecutive two-byte UTF-8 sequences to UTF-32.
#[inline(always)]
pub unsafe fn utf8_2x32_to_utf32(src: *const u8, dst: *mut u32) -> bool {
	// SAFETY: callers guarantee 64 readable bytes and 32 writable units.
	unsafe {
		let (lead0, tail0) = deinterleave_utf8_2(load_u8x16(src), load_u8x16(src.add(16)));
		let (lead1, tail1) = deinterleave_utf8_2(load_u8x16(src.add(32)), load_u8x16(src.add(48)));
		let range = (lead0 - Simd::splat(0x80)) | (lead1 - Simd::splat(0x80));
		if range.simd_gt(Simd::splat(0x5f)).any() {
			return false;
		}
		store_u32x16(dst, decode_utf8_2x16(lead0, tail0).cast());
		store_u32x16(dst.add(16), decode_utf8_2x16(lead1, tail1).cast());
	}
	true
}

#[inline(always)]
unsafe fn load_utf8_3x16(src: *const u8) -> (Simd<u8, 16>, Simd<u8, 16>, Simd<u8, 16>) {
	// SAFETY: callers guarantee 48 readable bytes.
	let (a, b, c) = unsafe { (load_u8x16(src), load_u8x16(src.add(16)), load_u8x16(src.add(32))) };
	let lead_lo: Simd<u8, 8> = simd_swizzle!(a, b, [0, 3, 6, 9, 12, 15, 18, 21]);
	let lead_hi: Simd<u8, 8> = simd_swizzle!(b, c, [8, 11, 14, 17, 20, 23, 26, 29]);
	let middle_lo: Simd<u8, 8> = simd_swizzle!(a, b, [1, 4, 7, 10, 13, 16, 19, 22]);
	let middle_hi: Simd<u8, 8> = simd_swizzle!(b, c, [9, 12, 15, 18, 21, 24, 27, 30]);
	let tail_lo: Simd<u8, 8> = simd_swizzle!(a, b, [2, 5, 8, 11, 14, 17, 20, 23]);
	let tail_hi: Simd<u8, 8> = simd_swizzle!(b, c, [10, 13, 16, 19, 22, 25, 28, 31]);
	(
		combine_u8x8(lead_lo, lead_hi),
		combine_u8x8(middle_lo, middle_hi),
		combine_u8x8(tail_lo, tail_hi),
	)
}

#[inline(always)]
fn valid_utf8_3x16(leads: Simd<u8, 16>, middles: Simd<u8, 16>, tails: Simd<u8, 16>) -> bool {
	(leads - Simd::splat(0xe0)).simd_le(Simd::splat(0x0f)).all()
		&& (middles & Simd::splat(0xc0))
			.simd_eq(Simd::splat(0x80))
			.all()
		&& (tails & Simd::splat(0xc0)).simd_eq(Simd::splat(0x80)).all()
}

#[inline(always)]
fn decode_utf8_3x16(
	leads: Simd<u8, 16>,
	middles: Simd<u8, 16>,
	tails: Simd<u8, 16>,
) -> Simd<u16, 16> {
	((leads & Simd::splat(0x0f)).cast::<u16>() << 12)
		| ((middles & Simd::splat(0x3f)).cast::<u16>() << 6)
		| (tails & Simd::splat(0x3f)).cast()
}

/// Converts 16 consecutive three-byte UTF-8 sequences to UTF-16.
#[inline(always)]
pub unsafe fn utf8_3x16_to_utf16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: callers guarantee 48 readable bytes and 16 writable units.
	let (leads, middles, tails) = unsafe { load_utf8_3x16(src) };
	if !valid_utf8_3x16(leads, middles, tails) {
		return false;
	}
	// SAFETY: the destination has sixteen writable units.
	unsafe { store_u16x16(dst, decode_utf8_3x16(leads, middles, tails)) };
	true
}

/// Converts 16 consecutive three-byte UTF-8 sequences to UTF-32.
#[inline(always)]
pub unsafe fn utf8_3x16_to_utf32(src: *const u8, dst: *mut u32) -> bool {
	// SAFETY: callers guarantee 48 readable bytes and 16 writable units.
	let (leads, middles, tails) = unsafe { load_utf8_3x16(src) };
	if !valid_utf8_3x16(leads, middles, tails) {
		return false;
	}
	// SAFETY: the destination has sixteen writable units.
	unsafe { store_u32x16(dst, decode_utf8_3x16(leads, middles, tails).cast()) };
	true
}

#[inline(always)]
fn decode_utf8_4x4(input: Simd<u8, 16>) -> Option<Simd<u32, 4>> {
	let lead: Simd<u8, 4> = simd_swizzle!(input, [0, 4, 8, 12]);
	if !lead.simd_ge(Simd::splat(0xf0)).all() {
		return None;
	}
	let second: Simd<u8, 4> = simd_swizzle!(input, [1, 5, 9, 13]);
	let third: Simd<u8, 4> = simd_swizzle!(input, [2, 6, 10, 14]);
	let tail: Simd<u8, 4> = simd_swizzle!(input, [3, 7, 11, 15]);
	Some(
		((lead & Simd::splat(0x07)).cast::<u32>() << 18)
			| ((second & Simd::splat(0x3f)).cast::<u32>() << 12)
			| ((third & Simd::splat(0x3f)).cast::<u32>() << 6)
			| (tail & Simd::splat(0x3f)).cast(),
	)
}

#[inline(always)]
unsafe fn store_surrogates(cp: Simd<u32, 4>, dst: *mut u16) {
	let adjusted = cp - Simd::splat(0x10000);
	let hi = ((adjusted >> 10) | Simd::splat(0xd800)).cast::<u16>();
	let lo = ((adjusted & Simd::splat(0x03ff)) | Simd::splat(0xdc00)).cast::<u16>();
	let pairs: Simd<u16, 8> = simd_swizzle!(hi, lo, [0, 4, 1, 5, 2, 6, 3, 7]);
	// SAFETY: callers guarantee eight writable UTF-16 units.
	unsafe { store_u16x8(dst, pairs) }
}

/// Converts 16 consecutive four-byte UTF-8 sequences to UTF-16.
#[inline(always)]
pub unsafe fn utf8_4x16_to_utf16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: callers guarantee 64 readable bytes and 32 writable units.
	let (a, b, c, d) = unsafe {
		(load_u8x16(src), load_u8x16(src.add(16)), load_u8x16(src.add(32)), load_u8x16(src.add(48)))
	};
	let (Some(ca), Some(cb), Some(cc), Some(cd)) =
		(decode_utf8_4x4(a), decode_utf8_4x4(b), decode_utf8_4x4(c), decode_utf8_4x4(d))
	else {
		return false;
	};
	if ca.simd_min(cb).simd_min(cc.simd_min(cd)).reduce_min() <= 0xffff {
		return false;
	}
	// SAFETY: each call writes eight units within the destination.
	unsafe {
		store_surrogates(ca, dst);
		store_surrogates(cb, dst.add(8));
		store_surrogates(cc, dst.add(16));
		store_surrogates(cd, dst.add(24));
	}
	true
}

/// Converts 16 consecutive four-byte UTF-8 sequences to UTF-32.
#[inline(always)]
pub unsafe fn utf8_4x16_to_utf32(src: *const u8, dst: *mut u32) -> bool {
	// SAFETY: callers guarantee 64 readable bytes and 16 writable units.
	let (a, b, c, d) = unsafe {
		(load_u8x16(src), load_u8x16(src.add(16)), load_u8x16(src.add(32)), load_u8x16(src.add(48)))
	};
	let (Some(ca), Some(cb), Some(cc), Some(cd)) =
		(decode_utf8_4x4(a), decode_utf8_4x4(b), decode_utf8_4x4(c), decode_utf8_4x4(d))
	else {
		return false;
	};
	if ca.simd_min(cb).simd_min(cc.simd_min(cd)).reduce_min() <= 0xffff {
		return false;
	}
	// SAFETY: each call writes four units within the destination.
	unsafe {
		store_u32x4(dst, ca);
		store_u32x4(dst.add(4), cb);
		store_u32x4(dst.add(8), cc);
		store_u32x4(dst.add(12), cd);
	}
	true
}

#[inline(always)]
fn pack_utf8_2x16(a: Simd<u16, 8>, b: Simd<u16, 8>) -> (Simd<u8, 16>, Simd<u8, 16>) {
	let lead =
		combine_u8x8(((a >> 6) | Simd::splat(0xc0)).cast(), ((b >> 6) | Simd::splat(0xc0)).cast());
	let tail = combine_u8x8(
		((a & Simd::splat(0x3f)) | Simd::splat(0x80)).cast(),
		((b & Simd::splat(0x3f)) | Simd::splat(0x80)).cast(),
	);
	(lead, tail)
}

#[inline(always)]
fn pack_utf8_3x16(a: Simd<u16, 8>, b: Simd<u16, 8>) -> (Simd<u8, 16>, Simd<u8, 16>, Simd<u8, 16>) {
	let lead =
		combine_u8x8(((a >> 12) | Simd::splat(0xe0)).cast(), ((b >> 12) | Simd::splat(0xe0)).cast());
	let middle = combine_u8x8(
		(((a >> 6) & Simd::splat(0x3f)) | Simd::splat(0x80)).cast(),
		(((b >> 6) & Simd::splat(0x3f)) | Simd::splat(0x80)).cast(),
	);
	let tail = combine_u8x8(
		((a & Simd::splat(0x3f)) | Simd::splat(0x80)).cast(),
		((b & Simd::splat(0x3f)) | Simd::splat(0x80)).cast(),
	);
	(lead, middle, tail)
}

#[inline(always)]
fn pack_utf8_4x4(cp: Simd<u32, 4>) -> Simd<u8, 16> {
	let lead = (cp >> 18) | Simd::splat(0xf0);
	let second = (((cp >> 12) & Simd::splat(0x3f)) | Simd::splat(0x80)) << 8;
	let third = (((cp >> 6) & Simd::splat(0x3f)) | Simd::splat(0x80)) << 16;
	let tail = ((cp & Simd::splat(0x3f)) | Simd::splat(0x80)) << 24;
	(lead | second | third | tail).to_le_bytes()
}

/// Converts 32 codepoints in U+0080..=U+07FF to two-byte UTF-8.
#[inline(always)]
pub unsafe fn utf16_2x32_to_utf8(src: *const u16, dst: *mut u8) -> bool {
	// SAFETY: callers guarantee 32 readable units and 64 writable bytes.
	let (a, b, c, d) = unsafe {
		(load_u16x8(src), load_u16x8(src.add(8)), load_u16x8(src.add(16)), load_u16x8(src.add(24)))
	};
	let min = a.simd_min(b).simd_min(c.simd_min(d)).reduce_min();
	let max = a.simd_max(b).simd_max(c.simd_max(d)).reduce_max();
	if min < 0x80 || max > 0x07ff {
		return false;
	}
	let (lead0, tail0) = pack_utf8_2x16(a, b);
	let (lead1, tail1) = pack_utf8_2x16(c, d);
	// SAFETY: each call writes 32 bytes within the destination.
	unsafe {
		store_utf8_2x16(lead0, tail0, dst);
		store_utf8_2x16(lead1, tail1, dst.add(32));
	}
	true
}

#[inline(always)]
unsafe fn utf16_one_two_half_to_utf8(input: Simd<u16, 8>, dst: *mut u8) -> usize {
	let ascii = input.simd_le(Simd::splat(0x7f));
	let encoded = (input >> 6) | ((input & Simd::splat(0x003f)) << 8) | Simd::splat(0x80c0);
	let pairs = ascii.select(input, encoded);
	let mask = ascii.to_bitmask() as usize;
	// SAFETY: an eight-lane bitmask indexes the 256-row table.
	let row = unsafe { crate::tables::UTF16_ONE_TWO.get_unchecked(mask) };
	let bytes = shuffle_u8x16(pairs.to_le_bytes(), row.shuffle);
	// SAFETY: callers guarantee sixteen writable scratch bytes.
	unsafe { store_u8x16(dst, bytes) };
	usize::from(row.written)
}

#[inline(always)]
unsafe fn utf16_3_to_utf8(input: Simd<u16, 8>, dst: *mut u8) -> usize {
	let (out0, written0, out1, written1) = pack_utf16_1_2_3(input);
	// SAFETY: callers guarantee 32 writable scratch bytes.
	unsafe {
		store_u8x16(dst, out0);
		store_u8x16(dst.add(written0), out1);
	}
	written0 + written1
}

/// Encodes eight codepoints already known to occupy three UTF-8 bytes.
#[inline(always)]
pub unsafe fn utf16_3x8_to_utf8(input: Simd<u16, 8>, dst: *mut u8) {
	let lead = ((input >> 12) | Simd::splat(0xe0)).cast();
	let middle = (((input >> 6) & Simd::splat(0x3f)) | Simd::splat(0x80)).cast();
	let tail = ((input & Simd::splat(0x3f)) | Simd::splat(0x80)).cast();
	// SAFETY: callers guarantee 24 writable bytes.
	unsafe { store_utf8_3x8(lead, middle, tail, dst) };
}

#[inline(always)]
unsafe fn utf16_bmp_half_to_utf8(input: Simd<u16, 8>, max: u16, dst: *mut u8) -> usize {
	// SAFETY: callers guarantee eight non-surrogate units and 32 scratch bytes.
	unsafe {
		if max <= 0x7f {
			dst.cast::<[u8; 8]>()
				.write_unaligned(*input.cast::<u8>().as_array());
			return 8;
		}
		if max <= 0x07ff {
			return utf16_one_two_half_to_utf8(input, dst);
		}
		if input.reduce_min() >= 0x0800 {
			utf16_3x8_to_utf8(input, dst);
			return 24;
		}
		utf16_3_to_utf8(input, dst)
	}
}

/// Converts one validated non-surrogate UTF-16 block already in registers.
#[inline(always)]
pub unsafe fn utf16_block_to_utf8(
	a: Simd<u16, 8>,
	b: Simd<u16, 8>,
	max_a: u16,
	max_b: u16,
	dst: *mut u8,
) -> usize {
	// SAFETY: callers guarantee non-surrogate values and 64 scratch bytes.
	unsafe {
		let written = utf16_bmp_half_to_utf8(a, max_a, dst);
		let tail = utf16_bmp_half_to_utf8(b, max_b, dst.add(written));
		written + tail
	}
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn utf16_is_sparse_three(src: *const u16) -> bool {
	// SAFETY: callers guarantee 64 readable units.
	let (a, b, c, d, e, f, g, h) = unsafe {
		(
			load_u16x8(src),
			load_u16x8(src.add(8)),
			load_u16x8(src.add(16)),
			load_u16x8(src.add(24)),
			load_u16x8(src.add(32)),
			load_u16x8(src.add(40)),
			load_u16x8(src.add(48)),
			load_u16x8(src.add(56)),
		)
	};
	let below_three = a.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones()
		+ b.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones()
		+ c.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones()
		+ d.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones()
		+ e.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones()
		+ f.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones()
		+ g.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones()
		+ h.simd_lt(Simd::splat(0x0800)).to_bitmask().count_ones();
	below_three <= 8
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn utf16_sparse_three_prefix(src: *const u16, len: usize, dst: *mut u8) -> (usize, usize) {
	let (mut i, mut o) = (0, 0);
	while len - i >= 8 {
		if len - i >= 16 {
			// SAFETY: the loop bounds cover sixteen units and 48 output bytes.
			if unsafe { utf16_3x16_to_utf8(src.add(i), dst.add(o)) } {
				i += 16;
				o += 48;
				continue;
			}
		}
		// SAFETY: the loop bounds cover eight input units.
		let units = unsafe { load_u16x8(src.add(i)) };
		let max = max_u16x8(units);
		if max >= 0xd800
			&& (units & Simd::splat(0xf800))
				.simd_eq(Simd::splat(0xd800))
				.any()
		{
			break;
		}
		if units.reduce_min() >= 0x0800 {
			// SAFETY: every lane has three-byte width.
			unsafe { utf16_3x8_to_utf8(units, dst.add(o)) };
			o += 24;
		} else {
			// SAFETY: the surrogate check proves the mixed BMP block is supported.
			o += unsafe { utf16_3_to_utf8(units, dst.add(o)) };
		}
		i += 8;
	}
	(i, o)
}

#[inline(always)]
unsafe fn utf16_one_two_prefix(src: *const u16, len: usize, dst: *mut u8) -> (usize, usize) {
	let (mut i, mut o) = (0, 0);
	while len - i >= 64 {
		// SAFETY: the loop bound covers the fixed source block.
		let (a, b, c, d, e, f, g, h) = unsafe {
			(
				load_u16x8(src.add(i)),
				load_u16x8(src.add(i + 8)),
				load_u16x8(src.add(i + 16)),
				load_u16x8(src.add(i + 24)),
				load_u16x8(src.add(i + 32)),
				load_u16x8(src.add(i + 40)),
				load_u16x8(src.add(i + 48)),
				load_u16x8(src.add(i + 56)),
			)
		};
		let max = a
			.simd_max(b)
			.simd_max(c.simd_max(d))
			.simd_max(e.simd_max(f).simd_max(g.simd_max(h)))
			.reduce_max();
		if max > 0x07ff {
			break;
		}
		// SAFETY: classification proves every lane has one- or two-byte width,
		// and the caller provides scratch space for each compacting store.
		unsafe {
			o += utf16_one_two_half_to_utf8(a, dst.add(o));
			o += utf16_one_two_half_to_utf8(b, dst.add(o));
			o += utf16_one_two_half_to_utf8(c, dst.add(o));
			o += utf16_one_two_half_to_utf8(d, dst.add(o));
			o += utf16_one_two_half_to_utf8(e, dst.add(o));
			o += utf16_one_two_half_to_utf8(f, dst.add(o));
			o += utf16_one_two_half_to_utf8(g, dst.add(o));
			o += utf16_one_two_half_to_utf8(h, dst.add(o));
		}
		i += 64;
	}
	(i, o)
}

/// Converts non-surrogate UTF-16 blocks until the first unsupported block.
#[inline(always)]
pub unsafe fn utf16_fast_prefix(
	src: *const u16,
	len: usize,
	dst: *mut u8,
	dst_len: usize,
) -> (usize, usize) {
	let Some(required) = len.checked_mul(3).and_then(|bytes| bytes.checked_add(16)) else {
		return (0, 0);
	};
	if dst_len < required {
		return (0, 0);
	}
	#[cfg(target_arch = "x86_64")]
	if crate::x86::available() {
		return (0, 0);
	}
	// SAFETY: callers supply the readable source and writable destination
	// ranges.
	unsafe {
		if len >= 64 {
			let a = load_u16x8(src);
			let b = load_u16x8(src.add(8));
			let c = load_u16x8(src.add(16));
			let d = load_u16x8(src.add(24));
			let e = load_u16x8(src.add(32));
			let f = load_u16x8(src.add(40));
			let g = load_u16x8(src.add(48));
			let h = load_u16x8(src.add(56));
			let min = a
				.simd_min(b)
				.simd_min(c.simd_min(d))
				.simd_min(e.simd_min(f).simd_min(g.simd_min(h)))
				.reduce_min();
			let max = a
				.simd_max(b)
				.simd_max(c.simd_max(d))
				.simd_max(e.simd_max(f).simd_max(g.simd_max(h)))
				.reduce_max();
			if max <= 0x07ff {
				if min >= 0x80 {
					return (0, 0);
				}
				let non_ascii = a.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones()
					+ b.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones()
					+ c.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones()
					+ d.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones()
					+ e.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones()
					+ f.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones()
					+ g.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones()
					+ h.simd_gt(Simd::splat(0x7f)).to_bitmask().count_ones();
				if non_ascii >= 32 {
					return utf16_one_two_prefix(src, len, dst);
				}
			}
			if min >= 0x0800 && (max < 0xd800 || min > 0xdfff) {
				return (0, 0);
			}
			#[cfg(target_arch = "aarch64")]
			if max >= 0x0800 && utf16_is_sparse_three(src) {
				return utf16_sparse_three_prefix(src, len, dst);
			}
		}
		let (mut i, mut o) = (0usize, 0usize);
		while len - i >= 8 {
			let mut units = load_u16x8(src.add(i));
			let mut max = max_u16x8(units);
			if max & 0xff80 == 0 {
				if len - i < 16 {
					dst.add(o)
						.cast::<[u8; 8]>()
						.write_unaligned(*units.cast::<u8>().as_array());
					i += 8;
					o += 8;
					continue;
				}
				let next = load_u16x8(src.add(i + 8));
				let next_max = max_u16x8(next);
				if next_max & 0xff80 == 0 {
					let bytes = combine_u8x8(units.cast(), next.cast());
					store_u8x16(dst.add(o), bytes);
					i += 16;
					o += 16;
					continue;
				}
				dst.add(o)
					.cast::<[u8; 8]>()
					.write_unaligned(*units.cast::<u8>().as_array());
				i += 8;
				o += 8;
				units = next;
				max = next_max;
			}
			if max & 0xf800 == 0 {
				o += utf16_one_two_half_to_utf8(units, dst.add(o));
				i += 8;
				continue;
			}
			if max < 0xd800 {
				o += utf16_3_to_utf8(units, dst.add(o));
				i += 8;
				continue;
			}
			if (units & Simd::splat(0xf800))
				.simd_eq(Simd::splat(0xd800))
				.any()
			{
				break;
			}
			o += utf16_3_to_utf8(units, dst.add(o));
			i += 8;
		}
		(i, o)
	}
}

/// Converts 16 non-surrogate BMP UTF-16 codepoints to three-byte UTF-8.
#[inline(always)]
pub unsafe fn utf16_3x16_to_utf8(src: *const u16, dst: *mut u8) -> bool {
	// SAFETY: callers guarantee 16 readable units and 48 writable bytes.
	let (a, b) = unsafe { (load_u16x8(src), load_u16x8(src.add(8))) };
	let range = (a - Simd::splat(0x0800)).simd_max(b - Simd::splat(0x0800));
	let surrogate = (a & Simd::splat(0xf800)).simd_eq(Simd::splat(0xd800))
		| (b & Simd::splat(0xf800)).simd_eq(Simd::splat(0xd800));
	if range.simd_gt(Simd::splat(0xf7ff)).any() || surrogate.any() {
		return false;
	}
	let (lead, middle, tail) = pack_utf8_3x16(a, b);
	// SAFETY: the destination has 48 writable bytes.
	unsafe { store_utf8_3x16(lead, middle, tail, dst) };
	true
}

/// Converts 16 UTF-16 surrogate pairs to four-byte UTF-8.
#[inline(always)]
pub unsafe fn utf16_4x16_to_utf8(src: *const u16, dst: *mut u8) -> bool {
	// SAFETY: callers guarantee 32 readable units and 64 writable bytes.
	let (a, b, c, d) = unsafe {
		(load_u16x8(src), load_u16x8(src.add(8)), load_u16x8(src.add(16)), load_u16x8(src.add(24)))
	};
	let parts = |input: Simd<u16, 8>| {
		(simd_swizzle!(input, [0, 2, 4, 6]), simd_swizzle!(input, [1, 3, 5, 7]))
	};
	let (ah, al) = parts(a);
	let (bh, bl) = parts(b);
	let (ch, cl) = parts(c);
	let (dh, dl) = parts(d);
	let valid = |hi: Simd<u16, 4>, lo: Simd<u16, 4>| {
		(hi & Simd::splat(0xfc00)).simd_eq(Simd::splat(0xd800))
			& (lo & Simd::splat(0xfc00)).simd_eq(Simd::splat(0xdc00))
	};
	if !(valid(ah, al) & valid(bh, bl) & valid(ch, cl) & valid(dh, dl)).all() {
		return false;
	}
	let decode = |hi: Simd<u16, 4>, lo: Simd<u16, 4>| {
		((hi.cast::<u32>() - Simd::splat(0xd800)) << 10)
			+ (lo.cast::<u32>() - Simd::splat(0xdc00))
			+ Simd::splat(0x10000)
	};
	// SAFETY: each store writes sixteen bytes within the destination.
	unsafe {
		store_u8x16(dst, pack_utf8_4x4(decode(ah, al)));
		store_u8x16(dst.add(16), pack_utf8_4x4(decode(bh, bl)));
		store_u8x16(dst.add(32), pack_utf8_4x4(decode(ch, cl)));
		store_u8x16(dst.add(48), pack_utf8_4x4(decode(dh, dl)));
	}
	true
}

/// Converts a heterogeneous non-surrogate UTF-16 block to UTF-8.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn utf16_mixed_32_to_utf8(src: *const u16, dst: *mut u8) -> Option<usize> {
	#[cfg(target_arch = "x86_64")]
	// SAFETY: callers guarantee the fixed source and destination bounds.
	if let Some(written) = unsafe { crate::x86::utf16_mixed_32_to_utf8(src, dst) } {
		return Some(written);
	}
	// SAFETY: callers guarantee 32 readable units and 96 writable bytes.
	let (a, b, c, d) = unsafe {
		(load_u16x8(src), load_u16x8(src.add(8)), load_u16x8(src.add(16)), load_u16x8(src.add(24)))
	};
	let surrogate = |value: Simd<u16, 8>| (value & Simd::splat(0xf800)).simd_eq(Simd::splat(0xd800));
	if (surrogate(a) | surrogate(b) | surrogate(c) | surrogate(d)).any() {
		return None;
	}
	let min = a.simd_min(b).simd_min(c.simd_min(d)).reduce_min();
	let max = a.simd_max(b).simd_max(c.simd_max(d)).reduce_max();
	if max <= 0x7f || (min > 0x7f && max <= 0x07ff) || min > 0x07ff {
		return None;
	}
	// SAFETY: each helper writes at most 32 bytes inside the 96-byte scratch
	// area.
	unsafe {
		let mut written = utf16_3_to_utf8(a, dst);
		written += utf16_3_to_utf8(b, dst.add(written));
		written += utf16_3_to_utf8(c, dst.add(written));
		written += utf16_3_to_utf8(d, dst.add(written));
		Some(written)
	}
}

/// Converts 16 UTF-32 codepoints in U+0080..=U+07FF to UTF-8.
#[inline(always)]
pub unsafe fn utf32_2x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
	// SAFETY: callers guarantee 16 readable units and 32 writable bytes.
	let (a, b, c, d) = unsafe {
		(load_u32x4(src), load_u32x4(src.add(4)), load_u32x4(src.add(8)), load_u32x4(src.add(12)))
	};
	let min = a.simd_min(b).simd_min(c.simd_min(d)).reduce_min();
	let max = a.simd_max(b).simd_max(c.simd_max(d)).reduce_max();
	if min < 0x80 || max > 0x07ff {
		return false;
	}
	let lo = combine_u16x4(a.cast(), b.cast());
	let hi = combine_u16x4(c.cast(), d.cast());
	let (lead, tail) = pack_utf8_2x16(lo, hi);
	// SAFETY: the destination has 32 writable bytes.
	unsafe { store_utf8_2x16(lead, tail, dst) };
	true
}

/// Converts 16 UTF-32 codepoints in U+0800..=U+FFFF to UTF-8.
#[inline(always)]
pub unsafe fn utf32_3x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
	// SAFETY: callers guarantee 16 readable units and 48 writable bytes.
	let (a, b, c, d) = unsafe {
		(load_u32x4(src), load_u32x4(src.add(4)), load_u32x4(src.add(8)), load_u32x4(src.add(12)))
	};
	let min = a.simd_min(b).simd_min(c.simd_min(d)).reduce_min();
	let max = a.simd_max(b).simd_max(c.simd_max(d)).reduce_max();
	if min < 0x0800 || max > 0xffff {
		return false;
	}
	let lo = combine_u16x4(a.cast(), b.cast());
	let hi = combine_u16x4(c.cast(), d.cast());
	let (lead, middle, tail) = pack_utf8_3x16(lo, hi);
	// SAFETY: the destination has 48 writable bytes.
	unsafe { store_utf8_3x16(lead, middle, tail, dst) };
	true
}

/// Converts 16 UTF-32 codepoints in U+10000..=U+10FFFF to UTF-8.
#[inline(always)]
pub unsafe fn utf32_4x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
	// SAFETY: callers guarantee 16 readable units and 64 writable bytes.
	let (a, b, c, d) = unsafe {
		(load_u32x4(src), load_u32x4(src.add(4)), load_u32x4(src.add(8)), load_u32x4(src.add(12)))
	};
	let min = a.simd_min(b).simd_min(c.simd_min(d)).reduce_min();
	let max = a.simd_max(b).simd_max(c.simd_max(d)).reduce_max();
	if min < 0x10000 || max > 0x10ffff {
		return false;
	}
	// SAFETY: each store writes sixteen bytes within the destination.
	unsafe {
		store_u8x16(dst, pack_utf8_4x4(a));
		store_u8x16(dst.add(16), pack_utf8_4x4(b));
		store_u8x16(dst.add(32), pack_utf8_4x4(c));
		store_u8x16(dst.add(48), pack_utf8_4x4(d));
	}
	true
}

/// Converts one UTF-32 block with AVX-512 byte compaction when available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn utf32_mixed_16_to_utf8(src: *const u32, dst: *mut u8) -> Option<usize> {
	// SAFETY: callers guarantee the fixed source and destination bounds.
	unsafe { crate::x86::utf32_mixed_16_to_utf8(src, dst) }
}

/// Decodes an arbitrary one-to-three-byte UTF-8 block when the target has a
/// profitable compaction primitive.
#[allow(
	clippy::missing_const_for_fn,
	reason = "const only on non-x86_64; the x86_64 body calls runtime kernels"
)]
#[inline(always)]
pub unsafe fn utf8_block_to_utf16(
	src: *const u8,
	dst: *mut u16,
	continuation: u64,
	leads: u64,
	three: u64,
) -> Option<(usize, usize)> {
	#[cfg(target_arch = "x86_64")]
	{
		// SAFETY: the caller guarantees one readable block, output headroom,
		// and masks classified from this source block.
		unsafe { crate::x86::utf8_block_to_utf16(src, dst, continuation, leads, three) }
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		let _ = (src, dst, continuation, leads, three);
		None
	}
}

/// Decodes an arbitrary one-to-three-byte UTF-8 block when the target has a
/// profitable compaction primitive.
#[allow(
	clippy::missing_const_for_fn,
	reason = "const only on non-x86_64; the x86_64 body calls runtime kernels"
)]
#[inline(always)]
pub unsafe fn utf8_block_to_utf32(
	src: *const u8,
	dst: *mut u32,
	continuation: u64,
	leads: u64,
	three: u64,
) -> Option<usize> {
	#[cfg(target_arch = "x86_64")]
	{
		// SAFETY: the caller guarantees one readable block, output headroom,
		// and masks classified from this source block.
		unsafe { crate::x86::utf8_block_to_utf32(src, dst, continuation, leads, three) }
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		let _ = (src, dst, continuation, leads, three);
		None
	}
}
