//! `AArch64` NEON primitives for fixed-pattern UTF blocks.
//!
//! Only transforms with no portable-SIMD equivalent live here: interleaved
//! loads and stores (`ld2/ld3/st2/st3`), the `tbl` byte shuffle, the
//! shift-insert three-byte decoder, and the `vpaddq` mask reduction. All
//! mixed-width table logic is generic in [`crate::kernel`].

use core::{arch::aarch64::*, simd::Simd};

use crate::kernel::KernelSet;

/// AArch64 implementation of the native transcode kernel set.
pub(crate) struct Neon;

#[inline(always)]
unsafe fn load_u8x16(src: *const u8) -> uint8x16_t {
	// SAFETY: caller guarantees 16 readable bytes.
	unsafe { vld1q_u8(src) }
}

/// Packs four 16-lane byte masks (0x00/0xFF lanes) into one 64-bit bitmask,
/// lane 0 in bit 0, using simdutf's three-`vpaddq` reduction.
#[inline(always)]
pub unsafe fn bitmask64(m0: uint8x16_t, m1: uint8x16_t, m2: uint8x16_t, m3: uint8x16_t) -> u64 {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let bit: uint8x16_t = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
		let s0 = vpaddq_u8(vandq_u8(m0, bit), vandq_u8(m1, bit));
		let s1 = vpaddq_u8(vandq_u8(m2, bit), vandq_u8(m3, bit));
		let s2 = vpaddq_u8(s0, s1);
		vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(s2, s2)))
	}
}

/// Continuation and lead-byte positions plus a four-byte/garbage flag for
/// one 64-byte block: three compare families over one set of loads.
#[inline(always)]
pub unsafe fn classify_block_u8x64(src: *const u8) -> (u64, u64, u64, bool) {
	// SAFETY: caller guarantees 64 readable bytes.
	unsafe {
		let a = vld1q_u8(src);
		let b = vld1q_u8(src.add(16));
		let c = vld1q_u8(src.add(32));
		let d = vld1q_u8(src.add(48));
		let prefix = vdupq_n_u8(0xc0);
		let continuation = vdupq_n_u8(0x80);
		let cont = bitmask64(
			vceqq_u8(vandq_u8(a, prefix), continuation),
			vceqq_u8(vandq_u8(b, prefix), continuation),
			vceqq_u8(vandq_u8(c, prefix), continuation),
			vceqq_u8(vandq_u8(d, prefix), continuation),
		);
		let leads = bitmask64(
			vcgeq_u8(a, prefix),
			vcgeq_u8(b, prefix),
			vcgeq_u8(c, prefix),
			vcgeq_u8(d, prefix),
		);
		let lead3 = vdupq_n_u8(0xe0);
		let three = bitmask64(
			vcgeq_u8(a, lead3),
			vcgeq_u8(b, lead3),
			vcgeq_u8(c, lead3),
			vcgeq_u8(d, lead3),
		);
		let has4 =
			vmaxvq_u8(vcgeq_u8(vmaxq_u8(vmaxq_u8(a, b), vmaxq_u8(c, d)), vdupq_n_u8(0xf0))) != 0;
		(cont, leads, three, has4)
	}
}

#[inline(always)]
unsafe fn decode_2x16_unchecked(leads: uint8x16_t, tails: uint8x16_t) -> (uint16x8_t, uint16x8_t) {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let lead_mask = vdupq_n_u16(0x1f);
		let lead_lo = vandq_u16(vmovl_u8(vget_low_u8(leads)), lead_mask);
		let lead_hi = vandq_u16(vmovl_high_u8(leads), lead_mask);
		let tail_lo = vmovl_u8(vget_low_u8(tails));
		let tail_hi = vmovl_high_u8(tails);
		(vsliq_n_u16::<6>(tail_lo, lead_lo), vsliq_n_u16::<6>(tail_hi, lead_hi))
	}
}

#[inline(always)]
unsafe fn leads_4(input: uint8x16_t) -> uint8x8_t {
	// SAFETY: AArch64 mandates NEON; the local array remains alive for the load.
	unsafe {
		let indexes = vld1_u8([0, 4, 8, 12, 0, 4, 8, 12].as_ptr());
		vqtbl1_u8(input, indexes)
	}
}

#[inline(always)]
unsafe fn decode_4_unchecked(input: uint8x16_t) -> uint32x4_t {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let words = vreinterpretq_u32_u8(input);
		let lead = vandq_u32(words, vdupq_n_u32(0x07));
		let second = vandq_u32(vshrq_n_u32::<8>(words), vdupq_n_u32(0x3f));
		let third = vandq_u32(vshrq_n_u32::<16>(words), vdupq_n_u32(0x3f));
		let tail = vandq_u32(vshrq_n_u32::<24>(words), vdupq_n_u32(0x3f));
		vorrq_u32(
			vorrq_u32(vshlq_n_u32::<18>(lead), vshlq_n_u32::<12>(second)),
			vorrq_u32(vshlq_n_u32::<6>(third), tail),
		)
	}
}

#[inline(always)]
unsafe fn store_surrogates(cp: uint32x4_t, dst: *mut u16) {
	// SAFETY: AArch64 mandates NEON, and the caller provides space for eight
	// UTF-16 units.
	unsafe {
		let adjusted = vsubq_u32(cp, vdupq_n_u32(0x10000));
		let hi = vmovn_u32(vorrq_u32(vshrq_n_u32::<10>(adjusted), vdupq_n_u32(0xd800)));
		let lo = vmovn_u32(vorrq_u32(vandq_u32(adjusted, vdupq_n_u32(0x03ff)), vdupq_n_u32(0xdc00)));
		let pairs = vzip1q_u16(vcombine_u16(hi, hi), vcombine_u16(lo, lo));
		vst1q_u16(dst, pairs);
	}
}

/// Widens 64 ASCII bytes to UTF-16.
#[inline(always)]
pub unsafe fn ascii_u8_to_u16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: caller guarantees 64 readable bytes and 64 writable UTF-16 units.
	unsafe {
		let a = load_u8x16(src);
		let b = load_u8x16(src.add(16));
		let c = load_u8x16(src.add(32));
		let d = load_u8x16(src.add(48));
		if vmaxvq_u8(vcgtq_u8(vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d)), vdupq_n_u8(0x7f))) != 0 {
			return false;
		}
		macro_rules! store {
			($v:expr, $at:expr) => {{
				let v = $v;
				vst1q_u16(dst.add($at), vmovl_u8(vget_low_u8(v)));
				vst1q_u16(dst.add($at + 8), vmovl_high_u8(v));
			}};
		}
		store!(a, 0);
		store!(b, 16);
		store!(c, 32);
		store!(d, 48);
		true
	}
}

/// Widens 64 ASCII bytes to UTF-32.
#[inline(always)]
pub unsafe fn ascii_u8_to_u32(src: *const u8, dst: *mut u32) -> bool {
	// SAFETY: caller guarantees 64 readable bytes and 64 writable UTF-32 units.
	unsafe {
		let a = load_u8x16(src);
		let b = load_u8x16(src.add(16));
		let c = load_u8x16(src.add(32));
		let d = load_u8x16(src.add(48));
		let max = vmaxq_u8(vmaxq_u8(a, b), vmaxq_u8(c, d));
		if vmaxvq_u8(max) > 0x7f {
			return false;
		}
		let widen = #[inline(always)]
		|input: uint8x16_t, at: usize| {
			let lo = vmovl_u8(vget_low_u8(input));
			let hi = vmovl_high_u8(input);
			vst1q_u32(dst.add(at), vmovl_u16(vget_low_u16(lo)));
			vst1q_u32(dst.add(at + 4), vmovl_high_u16(lo));
			vst1q_u32(dst.add(at + 8), vmovl_u16(vget_low_u16(hi)));
			vst1q_u32(dst.add(at + 12), vmovl_high_u16(hi));
		};
		widen(a, 0);
		widen(b, 16);
		widen(c, 32);
		widen(d, 48);
		true
	}
}

#[inline(always)]
unsafe fn store_2x32_u16(ab: uint8x16x2_t, cd: uint8x16x2_t, dst: *mut u16) {
	// SAFETY: caller provides valid vector registers and writable output bounds.
	unsafe {
		let (a, b) = decode_2x16_unchecked(ab.0, ab.1);
		let (c, d) = decode_2x16_unchecked(cd.0, cd.1);
		vst1q_u16(dst, a);
		vst1q_u16(dst.add(8), b);
		vst1q_u16(dst.add(16), c);
		vst1q_u16(dst.add(24), d);
	}
}

#[inline(always)]
unsafe fn store_2x32_u32(ab: uint8x16x2_t, cd: uint8x16x2_t, dst: *mut u32) {
	// SAFETY: caller provides valid vector registers and writable output bounds.
	unsafe {
		let (a, b) = decode_2x16_unchecked(ab.0, ab.1);
		let (c, d) = decode_2x16_unchecked(cd.0, cd.1);
		vst1q_u32(dst, vmovl_u16(vget_low_u16(a)));
		vst1q_u32(dst.add(4), vmovl_high_u16(a));
		vst1q_u32(dst.add(8), vmovl_u16(vget_low_u16(b)));
		vst1q_u32(dst.add(12), vmovl_high_u16(b));
		vst1q_u32(dst.add(16), vmovl_u16(vget_low_u16(c)));
		vst1q_u32(dst.add(20), vmovl_high_u16(c));
		vst1q_u32(dst.add(24), vmovl_u16(vget_low_u16(d)));
		vst1q_u32(dst.add(28), vmovl_high_u16(d));
	}
}

/// Converts 32 consecutive two-byte UTF-8 sequences to UTF-16.
#[inline(always)]
pub unsafe fn utf8_2x32_to_utf16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: caller guarantees 64 readable bytes and 32 writable UTF-16 units.
	unsafe {
		let ab = vld2q_u8(src);
		let cd = vld2q_u8(src.add(32));
		let lead_range = vmaxq_u8(vsubq_u8(ab.0, vdupq_n_u8(0x80)), vsubq_u8(cd.0, vdupq_n_u8(0x80)));
		if vmaxvq_u8(vcgtq_u8(lead_range, vdupq_n_u8(0x5f))) != 0 {
			return false;
		}
		store_2x32_u16(ab, cd, dst);
	}
	true
}

/// Converts 32 consecutive two-byte UTF-8 sequences to UTF-32.
#[inline(always)]
pub unsafe fn utf8_2x32_to_utf32(src: *const u8, dst: *mut u32) -> bool {
	// SAFETY: caller guarantees 64 readable bytes and 32 writable UTF-32 units.
	unsafe {
		let ab = vld2q_u8(src);
		let cd = vld2q_u8(src.add(32));
		let lead_range = vmaxq_u8(vsubq_u8(ab.0, vdupq_n_u8(0x80)), vsubq_u8(cd.0, vdupq_n_u8(0x80)));
		if vmaxvq_u8(vcgtq_u8(lead_range, vdupq_n_u8(0x5f))) != 0 {
			return false;
		}
		store_2x32_u32(ab, cd, dst);
	}
	true
}

/// Converts four 3-byte UTF-8 sequences to UTF-16 using 5 NEON instructions.
#[inline(always)]
pub unsafe fn convert_utf8_3_byte_to_utf16(in_vec: uint8x16_t) -> uint16x4_t {
	// SAFETY: AArch64 mandates NEON; the shuffle constant stays alive for
	// the load.
	unsafe {
		let sh = vld1q_u8([0, 2, 3, 5, 6, 8, 9, 11, 1, 1, 4, 4, 7, 7, 10, 10].as_ptr());
		let perm = vqtbl1q_u8(in_vec, sh);
		let perm_low = vget_low_u8(perm);
		let perm_high = vget_high_u8(perm);
		let mid = vreinterpret_u16_u8(perm_high);
		let high = vreinterpret_u16_u8(perm_low);
		let mid_high = vsli_n_u16::<6>(mid, high);
		let low = vreinterpret_u16_u8(vrev16_u8(perm_low));
		vsli_n_u16::<6>(low, mid_high)
	}
}

#[inline(always)]
unsafe fn valid_3x16(input: uint8x16x3_t) -> bool {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let lead = vcleq_u8(vsubq_u8(input.0, vdupq_n_u8(0xe0)), vdupq_n_u8(0x0f));
		let middle = vceqq_u8(vandq_u8(input.1, vdupq_n_u8(0xc0)), vdupq_n_u8(0x80));
		let tail = vceqq_u8(vandq_u8(input.2, vdupq_n_u8(0xc0)), vdupq_n_u8(0x80));
		vminvq_u8(vandq_u8(vandq_u8(lead, middle), tail)) != 0
	}
}

#[inline(always)]
unsafe fn decode_3x16(input: uint8x16x3_t) -> (uint16x8_t, uint16x8_t) {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let decode = |lead: uint8x8_t, middle: uint8x8_t, tail: uint8x8_t| {
			let lead = vmovl_u8(vand_u8(lead, vdup_n_u8(0x0f)));
			let middle = vmovl_u8(vand_u8(middle, vdup_n_u8(0x3f)));
			let tail = vmovl_u8(vand_u8(tail, vdup_n_u8(0x3f)));
			vorrq_u16(vorrq_u16(vshlq_n_u16::<12>(lead), vshlq_n_u16::<6>(middle)), tail)
		};
		(
			decode(vget_low_u8(input.0), vget_low_u8(input.1), vget_low_u8(input.2)),
			decode(vget_high_u8(input.0), vget_high_u8(input.1), vget_high_u8(input.2)),
		)
	}
}

/// Converts 16 consecutive three-byte UTF-8 sequences to UTF-16 after full
/// validation.
#[inline(always)]
pub unsafe fn utf8_3x16_to_utf16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: caller guarantees 48 readable bytes and 16 writable UTF-16 units.
	unsafe {
		let input = vld3q_u8(src);
		if !valid_3x16(input) {
			return false;
		}
		let (lo, hi) = decode_3x16(input);
		vst1q_u16(dst, lo);
		vst1q_u16(dst.add(8), hi);
	}
	true
}

/// Converts 16 consecutive three-byte UTF-8 sequences to UTF-32 after full
/// validation.
#[inline(always)]
pub unsafe fn utf8_3x16_to_utf32(src: *const u8, dst: *mut u32) -> bool {
	// SAFETY: caller guarantees 48 readable bytes and 16 writable UTF-32 units.
	unsafe {
		let input = vld3q_u8(src);
		if !valid_3x16(input) {
			return false;
		}
		let (lo, hi) = decode_3x16(input);
		vst1q_u32(dst, vmovl_u16(vget_low_u16(lo)));
		vst1q_u32(dst.add(4), vmovl_high_u16(lo));
		vst1q_u32(dst.add(8), vmovl_u16(vget_low_u16(hi)));
		vst1q_u32(dst.add(12), vmovl_high_u16(hi));
	}
	true
}

/// Converts 16 consecutive four-byte UTF-8 sequences to UTF-16.
#[inline(always)]
pub unsafe fn utf8_4x16_to_utf16(src: *const u8, dst: *mut u16) -> bool {
	// SAFETY: caller guarantees 64 readable bytes and 32 writable UTF-16 units.
	unsafe {
		let a = load_u8x16(src);
		let b = load_u8x16(src.add(16));
		let c = load_u8x16(src.add(32));
		let d = load_u8x16(src.add(48));
		let leads = vmin_u8(vmin_u8(leads_4(a), leads_4(b)), vmin_u8(leads_4(c), leads_4(d)));
		if vminv_u8(leads) < 0xf0 {
			return false;
		}
		let ca = decode_4_unchecked(a);
		let cb = decode_4_unchecked(b);
		let cc = decode_4_unchecked(c);
		let cd = decode_4_unchecked(d);
		let min = vminq_u32(vminq_u32(ca, cb), vminq_u32(cc, cd));
		if vminvq_u32(min) <= 0xffff {
			return false;
		}
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
	// SAFETY: caller guarantees 64 readable bytes and 16 writable UTF-32 units.
	unsafe {
		let a = load_u8x16(src);
		let b = load_u8x16(src.add(16));
		let c = load_u8x16(src.add(32));
		let d = load_u8x16(src.add(48));
		let leads = vmin_u8(vmin_u8(leads_4(a), leads_4(b)), vmin_u8(leads_4(c), leads_4(d)));
		if vminv_u8(leads) < 0xf0 {
			return false;
		}
		let ca = decode_4_unchecked(a);
		let cb = decode_4_unchecked(b);
		let cc = decode_4_unchecked(c);
		let cd = decode_4_unchecked(d);
		let min = vminq_u32(vminq_u32(ca, cb), vminq_u32(cc, cd));
		if vminvq_u32(min) <= 0xffff {
			return false;
		}
		vst1q_u32(dst, ca);
		vst1q_u32(dst.add(4), cb);
		vst1q_u32(dst.add(8), cc);
		vst1q_u32(dst.add(12), cd);
	}
	true
}

#[inline(always)]
unsafe fn pack_2x16_u16(a: uint16x8_t, b: uint16x8_t) -> uint8x16x2_t {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let lead_a = vmovn_u16(vorrq_u16(vshrq_n_u16::<6>(a), vdupq_n_u16(0xc0)));
		let lead_b = vmovn_u16(vorrq_u16(vshrq_n_u16::<6>(b), vdupq_n_u16(0xc0)));
		let tail_a = vmovn_u16(vorrq_u16(vandq_u16(a, vdupq_n_u16(0x3f)), vdupq_n_u16(0x80)));
		let tail_b = vmovn_u16(vorrq_u16(vandq_u16(b, vdupq_n_u16(0x3f)), vdupq_n_u16(0x80)));
		uint8x16x2_t(vcombine_u8(lead_a, lead_b), vcombine_u8(tail_a, tail_b))
	}
}

/// Converts 32 codepoints in U+0080..=U+07FF to two-byte UTF-8.
#[inline(always)]
pub unsafe fn utf16_2x32_to_utf8(src: *const u16, dst: *mut u8) -> bool {
	// SAFETY: caller guarantees 32 readable UTF-16 units and 64 writable bytes.
	unsafe {
		let a = vld1q_u16(src);
		let b = vld1q_u16(src.add(8));
		let c = vld1q_u16(src.add(16));
		let d = vld1q_u16(src.add(24));
		let min_val = vminq_u16(vminq_u16(a, b), vminq_u16(c, d));
		let max_val = vmaxq_u16(vmaxq_u16(a, b), vmaxq_u16(c, d));
		if vminvq_u16(min_val) < 0x80 || vmaxvq_u16(vcgtq_u16(max_val, vdupq_n_u16(0x07ff))) != 0 {
			return false;
		}
		vst2q_u8(dst, pack_2x16_u16(a, b));
		vst2q_u8(dst.add(32), pack_2x16_u16(c, d));
	}
	true
}

#[inline(always)]
unsafe fn pack_3x16_u16(a: uint16x8_t, b: uint16x8_t) -> uint8x16x3_t {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let lead_a = vmovn_u16(vorrq_u16(vshrq_n_u16::<12>(a), vdupq_n_u16(0xe0)));
		let lead_b = vmovn_u16(vorrq_u16(vshrq_n_u16::<12>(b), vdupq_n_u16(0xe0)));
		let middle_a =
			vmovn_u16(vorrq_u16(vandq_u16(vshrq_n_u16::<6>(a), vdupq_n_u16(0x3f)), vdupq_n_u16(0x80)));
		let middle_b =
			vmovn_u16(vorrq_u16(vandq_u16(vshrq_n_u16::<6>(b), vdupq_n_u16(0x3f)), vdupq_n_u16(0x80)));
		let tail_a = vmovn_u16(vorrq_u16(vandq_u16(a, vdupq_n_u16(0x3f)), vdupq_n_u16(0x80)));
		let tail_b = vmovn_u16(vorrq_u16(vandq_u16(b, vdupq_n_u16(0x3f)), vdupq_n_u16(0x80)));
		uint8x16x3_t(
			vcombine_u8(lead_a, lead_b),
			vcombine_u8(middle_a, middle_b),
			vcombine_u8(tail_a, tail_b),
		)
	}
}

#[inline(always)]
unsafe fn pack_4_u32(cp: uint32x4_t) -> uint8x16_t {
	// SAFETY: AArch64 mandates NEON, and this operates only on vector values.
	unsafe {
		let lead = vorrq_u32(vshrq_n_u32::<18>(cp), vdupq_n_u32(0xf0));
		let second = vshlq_n_u32::<8>(vorrq_u32(
			vandq_u32(vshrq_n_u32::<12>(cp), vdupq_n_u32(0x3f)),
			vdupq_n_u32(0x80),
		));
		let third = vshlq_n_u32::<16>(vorrq_u32(
			vandq_u32(vshrq_n_u32::<6>(cp), vdupq_n_u32(0x3f)),
			vdupq_n_u32(0x80),
		));
		let tail = vshlq_n_u32::<24>(vorrq_u32(vandq_u32(cp, vdupq_n_u32(0x3f)), vdupq_n_u32(0x80)));
		vreinterpretq_u8_u32(vorrq_u32(vorrq_u32(lead, second), vorrq_u32(third, tail)))
	}
}

/// Converts 16 non-surrogate BMP UTF-16 codepoints to three-byte UTF-8.
#[inline(always)]
pub unsafe fn utf16_3x16_to_utf8(src: *const u16, dst: *mut u8) -> bool {
	// SAFETY: caller guarantees 16 readable UTF-16 units and 48 writable bytes.
	unsafe {
		let a = vld1q_u16(src);
		let b = vld1q_u16(src.add(8));
		let range = vmaxq_u16(vsubq_u16(a, vdupq_n_u16(0x0800)), vsubq_u16(b, vdupq_n_u16(0x0800)));
		let surrogate = vorrq_u16(
			vceqq_u16(vandq_u16(a, vdupq_n_u16(0xf800)), vdupq_n_u16(0xd800)),
			vceqq_u16(vandq_u16(b, vdupq_n_u16(0xf800)), vdupq_n_u16(0xd800)),
		);
		let invalid = vorrq_u16(vcgtq_u16(range, vdupq_n_u16(0xf7ff)), surrogate);
		if vmaxvq_u16(invalid) != 0 {
			return false;
		}
		vst3q_u8(dst, pack_3x16_u16(a, b));
	}
	true
}

/// Converts 16 UTF-32 codepoints to two-byte UTF-8 after checking every
/// lane is in U+0080..=U+07FF.
#[inline(always)]
pub unsafe fn utf32_2x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
	// SAFETY: caller guarantees 16 readable input units and 32 writable bytes.
	unsafe {
		let a = vld1q_u32(src);
		let b = vld1q_u32(src.add(4));
		let c = vld1q_u32(src.add(8));
		let d = vld1q_u32(src.add(12));
		let min = vminq_u32(vminq_u32(a, b), vminq_u32(c, d));
		let max = vmaxq_u32(vmaxq_u32(a, b), vmaxq_u32(c, d));
		if vminvq_u32(min) < 0x80 || vmaxvq_u32(max) > 0x07ff {
			return false;
		}
		let input = vcombine_u16(vmovn_u32(a), vmovn_u32(b));
		let next = vcombine_u16(vmovn_u32(c), vmovn_u32(d));
		vst2q_u8(dst, pack_2x16_u16(input, next));
	}
	true
}

/// Converts 16 UTF-32 codepoints to three-byte UTF-8 after checking every
/// lane is in U+0800..=U+FFFF (surrogate values pass through permissively).
#[inline(always)]
pub unsafe fn utf32_3x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
	// SAFETY: caller guarantees 16 readable input units and 48 writable bytes.
	unsafe {
		let a = vld1q_u32(src);
		let b = vld1q_u32(src.add(4));
		let c = vld1q_u32(src.add(8));
		let d = vld1q_u32(src.add(12));
		let min = vminq_u32(vminq_u32(a, b), vminq_u32(c, d));
		let max = vmaxq_u32(vmaxq_u32(a, b), vmaxq_u32(c, d));
		if vminvq_u32(min) < 0x800 || vmaxvq_u32(max) > 0xffff {
			return false;
		}
		let lo = vcombine_u16(vmovn_u32(a), vmovn_u32(b));
		let hi = vcombine_u16(vmovn_u32(c), vmovn_u32(d));
		vst3q_u8(dst, pack_3x16_u16(lo, hi));
	}
	true
}

/// Converts 16 UTF-32 codepoints to four-byte UTF-8 after checking every
/// lane is in U+10000..=U+10FFFF.
#[inline(always)]
pub unsafe fn utf32_4x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
	// SAFETY: caller guarantees 16 readable input units and 64 writable bytes.
	unsafe {
		let a = vld1q_u32(src);
		let b = vld1q_u32(src.add(4));
		let c = vld1q_u32(src.add(8));
		let d = vld1q_u32(src.add(12));
		let min = vminq_u32(vminq_u32(a, b), vminq_u32(c, d));
		let max = vmaxq_u32(vmaxq_u32(a, b), vmaxq_u32(c, d));
		if vminvq_u32(min) < 0x10000 || vmaxvq_u32(max) > 0x10ffff {
			return false;
		}
		vst1q_u8(dst, pack_4_u32(a));
		vst1q_u8(dst.add(16), pack_4_u32(b));
		vst1q_u8(dst.add(32), pack_4_u32(c));
		vst1q_u8(dst.add(48), pack_4_u32(d));
	}
	true
}

/// Converts 16 UTF-16 surrogate pairs to four-byte UTF-8.
#[inline(always)]
pub unsafe fn utf16_4x16_to_utf8(src: *const u16, dst: *mut u8) -> bool {
	// SAFETY: caller guarantees 32 readable UTF-16 units and 64 writable bytes.
	unsafe {
		let a = vld1q_u16(src);
		let b = vld1q_u16(src.add(8));
		let c = vld1q_u16(src.add(16));
		let d = vld1q_u16(src.add(24));
		let parts = |input: uint16x8_t| {
			(vget_low_u16(vuzp1q_u16(input, input)), vget_low_u16(vuzp2q_u16(input, input)))
		};
		let (ah, al) = parts(a);
		let (bh, bl) = parts(b);
		let (ch, cl) = parts(c);
		let (dh, dl) = parts(d);
		let valid = |hi: uint16x4_t, lo: uint16x4_t| {
			vand_u16(
				vceq_u16(vand_u16(hi, vdup_n_u16(0xfc00)), vdup_n_u16(0xd800)),
				vceq_u16(vand_u16(lo, vdup_n_u16(0xfc00)), vdup_n_u16(0xdc00)),
			)
		};
		let all_valid =
			vand_u16(vand_u16(valid(ah, al), valid(bh, bl)), vand_u16(valid(ch, cl), valid(dh, dl)));
		if vminv_u16(all_valid) != u16::MAX {
			return false;
		}
		let decode = |hi: uint16x4_t, lo: uint16x4_t| {
			vaddq_u32(
				vaddq_u32(
					vshlq_n_u32::<10>(vsubq_u32(vmovl_u16(hi), vdupq_n_u32(0xd800))),
					vsubq_u32(vmovl_u16(lo), vdupq_n_u32(0xdc00)),
				),
				vdupq_n_u32(0x10000),
			)
		};
		vst1q_u8(dst, pack_4_u32(decode(ah, al)));
		vst1q_u8(dst.add(16), pack_4_u32(decode(bh, bl)));
		vst1q_u8(dst.add(32), pack_4_u32(decode(ch, cl)));
		vst1q_u8(dst.add(48), pack_4_u32(decode(dh, dl)));
	}
	true
}

impl KernelSet for Neon {
	#[inline(always)]
	fn shuffle_u8x16(input: Simd<u8, 16>, indexes: [u8; 16]) -> Simd<u8, 16> {
		// SAFETY: AArch64 mandates NEON, and both vectors span sixteen bytes.
		unsafe { Simd::from(vqtbl1q_u8(input.into(), vld1q_u8(indexes.as_ptr()))) }
	}

	#[inline(always)]
	unsafe fn store_utf8_3x8(
		lead: Simd<u8, 8>,
		middle: Simd<u8, 8>,
		tail: Simd<u8, 8>,
		dst: *mut u8,
	) {
		// SAFETY: trait callers provide 24 writable bytes.
		unsafe { vst3_u8(dst, uint8x8x3_t(lead.into(), middle.into(), tail.into())) }
	}

	#[inline(always)]
	fn decode_utf8_3x4(input: Simd<u8, 16>) -> Simd<u16, 4> {
		// SAFETY: AArch64 mandates NEON; the converter reads vector registers only.
		unsafe { Simd::from(convert_utf8_3_byte_to_utf16(input.into())) }
	}

	#[inline(always)]
	unsafe fn ascii_u8_to_u16(src: *const u8, dst: *mut u16) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { ascii_u8_to_u16(src, dst) }
	}

	#[inline(always)]
	unsafe fn ascii_u8_to_u32(src: *const u8, dst: *mut u32) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { ascii_u8_to_u32(src, dst) }
	}

	#[inline(always)]
	unsafe fn classify_block_u8x64(src: *const u8) -> (u64, u64, u64, bool) {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { classify_block_u8x64(src) }
	}

	#[inline(always)]
	unsafe fn utf8_block_to_utf16(_: *const u8, _: *mut u16) -> Option<(usize, usize)> {
		None
	}

	#[inline(always)]
	unsafe fn utf8_block_to_utf32(_: *const u8, _: *mut u32) -> Option<usize> {
		None
	}

	#[inline(always)]
	unsafe fn utf8_2x32_to_utf16(src: *const u8, dst: *mut u16) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf8_2x32_to_utf16(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf8_2x32_to_utf32(src: *const u8, dst: *mut u32) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf8_2x32_to_utf32(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf8_3x16_to_utf16(src: *const u8, dst: *mut u16) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf8_3x16_to_utf16(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf8_3x16_to_utf32(src: *const u8, dst: *mut u32) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf8_3x16_to_utf32(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf8_4x16_to_utf16(src: *const u8, dst: *mut u16) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf8_4x16_to_utf16(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf8_4x16_to_utf32(src: *const u8, dst: *mut u32) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf8_4x16_to_utf32(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf16_2x32_to_utf8(src: *const u16, dst: *mut u8) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf16_2x32_to_utf8(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf16_3x16_to_utf8(src: *const u16, dst: *mut u8) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf16_3x16_to_utf8(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf16_4x16_to_utf8(src: *const u16, dst: *mut u8) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf16_4x16_to_utf8(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf16_mixed_32_to_utf8(_: *const u16, _: *mut u8) -> Option<usize> {
		None
	}

	#[inline(always)]
	unsafe fn utf32_2x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf32_2x16_to_utf8(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf32_3x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf32_3x16_to_utf8(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf32_4x16_to_utf8(src: *const u32, dst: *mut u8) -> bool {
		// SAFETY: trait callers uphold the fixed block bounds.
		unsafe { utf32_4x16_to_utf8(src, dst) }
	}

	#[inline(always)]
	unsafe fn utf32_mixed_16_to_utf8(_: *const u32, _: *mut u8) -> Option<usize> {
		None
	}
}
