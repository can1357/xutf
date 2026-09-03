//! Focused x86-64 kernels for operations portable SIMD cannot lower well.
//!
//! The generic conversion ladders stay in [`crate::native`] and
//! [`crate::kernel`]. This module only contains the AVX-512 VBMI2 compactor
//! used for heterogeneous UTF-8 blocks; without it those blocks fall through
//! to a branch-heavy 12-byte table loop.

use core::{
	arch::x86_64::*,
	sync::atomic::{AtomicU8, Ordering},
};

#[inline(always)]
const fn byte_indexes() -> [u8; 64] {
	let mut indexes = [0; 64];
	let mut lane = 0;
	while lane < indexes.len() {
		indexes[lane] = lane as u8;
		lane += 1;
	}
	indexes
}

const BYTE_INDEXES: [u8; 64] = byte_indexes();
const COMPRESS_TWO: u64 = 0x3333_3333_3333_3333;
const COMPRESS_THREE: u64 = 0x7777_7777_7777_7777;
static AVAILABLE: AtomicU8 = AtomicU8::new(0);

/// Reports whether the AVX-512 VBMI2/BMI2 compaction kernels can run.
#[inline(always)]
pub(crate) fn available() -> bool {
	if cfg!(all(
		target_feature = "avx2",
		target_feature = "bmi2",
		target_feature = "avx512f",
		target_feature = "avx512bw",
		target_feature = "avx512vbmi",
		target_feature = "avx512vbmi2"
	)) {
		return true;
	}
	let cached = AVAILABLE.load(Ordering::Relaxed);
	if cached != 0 {
		return cached == 2;
	}
	let detected = std::arch::is_x86_feature_detected!("avx2")
		&& std::arch::is_x86_feature_detected!("bmi2")
		&& std::arch::is_x86_feature_detected!("avx512f")
		&& std::arch::is_x86_feature_detected!("avx512bw")
		&& std::arch::is_x86_feature_detected!("avx512vbmi")
		&& std::arch::is_x86_feature_detected!("avx512vbmi2");
	AVAILABLE.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
	detected
}

#[inline(always)]
unsafe fn load_u8x64(src: *const u8) -> __m512i {
	// SAFETY: callers guarantee 64 readable bytes.
	unsafe { _mm512_loadu_si512(src.cast()) }
}
#[inline(always)]
unsafe fn load_u16x32(src: *const u16) -> __m512i {
	// SAFETY: callers guarantee 32 readable UTF-16 units.
	unsafe { _mm512_loadu_si512(src.cast()) }
}

#[inline(always)]
unsafe fn load_u32x16(src: *const u32) -> __m512i {
	// SAFETY: callers guarantee 16 readable UTF-32 units.
	unsafe { _mm512_loadu_si512(src.cast()) }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn load_constant(bytes: &[u8; 64]) -> __m512i {
	// SAFETY: the constant spans exactly one vector.
	unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) }
}

#[inline(always)]
fn low_mask32(count: usize) -> u32 {
	if count >= 32 {
		u32::MAX
	} else {
		(1_u32 << count) - 1
	}
}

#[inline(always)]
fn low_mask16(count: usize) -> u16 {
	if count >= 16 {
		u16::MAX
	} else {
		(1_u16 << count) - 1
	}
}

#[inline(always)]
fn mask16(mask: u64, shift: u32) -> u16 {
	u16::try_from((mask >> shift) & u64::from(u16::MAX)).unwrap()
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn store_utf32_half(decoded: __m512i, dst: *mut u32, count: usize) {
	let low = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(decoded));
	// SAFETY: the first mask caps the widening store at sixteen units.
	unsafe { _mm512_mask_storeu_epi32(dst.cast(), low_mask16(count), low) };
	if count > 16 {
		let high = _mm512_shuffle_i32x4::<0x4e>(decoded, decoded);
		let high = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(high));
		// SAFETY: the second mask contains only the remaining live units.
		unsafe { _mm512_mask_storeu_epi32(dst.add(16).cast(), low_mask16(count - 16), high) };
	}
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn decode_utf16_half(
	index_bytes: __m256i,
	cleared: __m512i,
	before_non_ascii: __m512i,
	third_leads: __m512i,
	has_three: bool,
) -> __m512i {
	let indexes = _mm512_cvtepu8_epi16(index_bytes);
	let last = _mm512_maskz_permutexvar_epi8(0x5555_5555_5555_5555, indexes, cleared);
	let previous = _mm512_add_epi16(indexes, _mm512_set1_epi16(-1));
	let second = _mm512_slli_epi16::<6>(_mm512_maskz_permutexvar_epi8(
		0x5555_5555_5555_5555,
		previous,
		before_non_ascii,
	));
	let decoded = _mm512_or_si512(last, second);
	if !has_three {
		return decoded;
	}
	let previous = _mm512_add_epi16(previous, _mm512_set1_epi16(-1));
	let third = _mm512_slli_epi16::<12>(_mm512_maskz_permutexvar_epi8(
		0x5555_5555_5555_5555,
		previous,
		third_leads,
	));
	_mm512_or_si512(decoded, third)
}
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn block_one_two_to_utf16(
	cleared: __m512i,
	before_non_ascii: __m512i,
	ends: u64,
	dst: *mut u16,
) {
	// SAFETY: the constant spans exactly one ZMM register.
	let indexes = _mm512_add_epi8(unsafe { load_constant(&BYTE_INDEXES) }, _mm512_set1_epi8(-1));
	let previous = _mm512_permutexvar_epi8(indexes, before_non_ascii);
	let low = _mm512_or_si512(
		_mm512_cvtepu8_epi16(_mm512_castsi512_si256(cleared)),
		_mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(previous))),
	);
	let low_mask = u32::try_from(ends & u64::from(u32::MAX)).unwrap();
	// SAFETY: the endpoint mask selects only live low-half codepoints.
	unsafe { _mm512_mask_compressstoreu_epi16(dst.cast(), low_mask, low) };
	let high_mask = u32::try_from(ends >> 32).unwrap();
	if high_mask != 0 {
		let cleared = _mm512_shuffle_i32x4::<0x4e>(cleared, cleared);
		let previous = _mm512_shuffle_i32x4::<0x4e>(previous, previous);
		let high = _mm512_or_si512(
			_mm512_cvtepu8_epi16(_mm512_castsi512_si256(cleared)),
			_mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(previous))),
		);
		// SAFETY: both compressed stores fit the caller's 64-unit output block.
		unsafe {
			_mm512_mask_compressstoreu_epi16(
				dst.add(low_mask.count_ones() as usize).cast(),
				high_mask,
				high,
			)
		};
	}
}
#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn block_capped_one_two_to_utf16(
	cleared: __m512i,
	before_non_ascii: __m512i,
	ends: u64,
	dst: *mut u16,
) {
	// SAFETY: the constant spans exactly one ZMM register.
	let indexes = _mm512_maskz_compress_epi8(ends, unsafe { load_constant(&BYTE_INDEXES) });
	let indexes = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(indexes));
	let last = _mm512_maskz_permutexvar_epi8(0x5555_5555_5555_5555, indexes, cleared);
	let previous = _mm512_add_epi16(indexes, _mm512_set1_epi16(-1));
	let second = _mm512_slli_epi16::<6>(_mm512_maskz_permutexvar_epi8(
		0x5555_5555_5555_5555,
		previous,
		before_non_ascii,
	));
	// SAFETY: capping selected exactly 32 endpoint indexes.
	unsafe { _mm512_storeu_si512(dst.cast(), _mm512_or_si512(last, second)) };
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn block_to_utf16(
	input: __m512i,
	dst: *mut u16,
	continuation: u64,
	leads: u64,
	three: u64,
) -> Option<(usize, usize)> {
	let complete = if leads & (1_u64 << 63) != 0 {
		(1_u64 << 63) - 1
	} else if three & (1_u64 << 62) != 0 {
		(1_u64 << 62) - 1
	} else {
		u64::MAX
	};
	if (continuation & complete) != (((leads << 1) | (three << 2)) & complete) {
		return None;
	}
	let mut ends = (!continuation >> 1) & complete;
	if complete == u64::MAX {
		ends |= 1_u64 << 63;
	}
	let active_three = three & complete;
	let total = ends.count_ones() as usize;
	let capped = active_three == 0 && (33..=40).contains(&total);
	if capped {
		ends = _pdep_u64(u64::from(u32::MAX), ends);
	}
	let written = ends.count_ones() as usize;
	let consumed = 64 - ends.leading_zeros() as usize;
	let not_ascii = continuation | leads;
	let non_ascii_tags = _mm512_maskz_mov_epi8(not_ascii, _mm512_set1_epi8(-64));
	let cleared = _mm512_andnot_si512(non_ascii_tags, input);
	let before_non_ascii = _mm512_maskz_mov_epi8(not_ascii >> 1, cleared);
	if capped {
		// SAFETY: validation proved a complete 32-codepoint one-or-two-byte
		// prefix.
		unsafe { block_capped_one_two_to_utf16(cleared, before_non_ascii, ends, dst) };
		return Some((consumed, written));
	}
	if active_three == 0 {
		// SAFETY: validation proved a complete one-or-two-byte block prefix.
		unsafe { block_one_two_to_utf16(cleared, before_non_ascii, ends, dst) };
		return Some((consumed, written));
	}
	// SAFETY: the constant spans exactly one ZMM register.
	let indexes = _mm512_maskz_compress_epi8(ends, unsafe { load_constant(&BYTE_INDEXES) });
	let third_leads = _mm512_maskz_mov_epi8(active_three, cleared);
	// SAFETY: the low half contains the first 32 compressed byte indexes.
	let decoded = unsafe {
		decode_utf16_half(
			_mm512_castsi512_si256(indexes),
			cleared,
			before_non_ascii,
			third_leads,
			active_three != 0,
		)
	};
	// SAFETY: the mask bounds the first store to the live codepoints.
	unsafe { _mm512_mask_storeu_epi16(dst.cast(), low_mask32(written), decoded) };
	if written > 32 {
		// Swap the 256-bit halves so the remaining indexes can use the same
		// widening decoder without requiring AVX-512DQ extraction.
		let high = _mm512_shuffle_i32x4::<0x4e>(indexes, indexes);
		// SAFETY: the low half now contains compressed indexes 32 through 63.
		let decoded = unsafe {
			decode_utf16_half(
				_mm512_castsi512_si256(high),
				cleared,
				before_non_ascii,
				third_leads,
				active_three != 0,
			)
		};
		// SAFETY: the second mask contains exactly the remaining codepoints.
		unsafe { _mm512_mask_storeu_epi16(dst.add(32).cast(), low_mask32(written - 32), decoded) };
	}
	Some((consumed, written))
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn block_to_utf32(
	input: __m512i,
	dst: *mut u32,
	continuation: u64,
	leads: u64,
	three: u64,
) -> Option<(usize, usize)> {
	let complete = if leads & (1_u64 << 63) != 0 {
		(1_u64 << 63) - 1
	} else if three & (1_u64 << 62) != 0 {
		(1_u64 << 62) - 1
	} else {
		u64::MAX
	};
	if (continuation & complete) != (((leads << 1) | (three << 2)) & complete) {
		return None;
	}
	let mut ends = (!continuation >> 1) & complete;
	if complete == u64::MAX {
		ends |= 1_u64 << 63;
	}
	let written = ends.count_ones() as usize;
	let consumed = 64 - ends.leading_zeros() as usize;
	// SAFETY: the constant spans exactly one ZMM register.
	let indexes = _mm512_maskz_compress_epi8(ends, unsafe { load_constant(&BYTE_INDEXES) });
	let not_ascii = continuation | leads;
	let non_ascii_tags = _mm512_maskz_mov_epi8(not_ascii, _mm512_set1_epi8(-64));
	let cleared = _mm512_andnot_si512(non_ascii_tags, input);
	let before_non_ascii = _mm512_maskz_mov_epi8(not_ascii >> 1, cleared);
	let third_leads = _mm512_maskz_mov_epi8(three & complete, cleared);
	// SAFETY: the low half contains the first 32 endpoint indexes.
	let decoded = unsafe {
		decode_utf16_half(
			_mm512_castsi512_si256(indexes),
			cleared,
			before_non_ascii,
			third_leads,
			three != 0,
		)
	};
	// SAFETY: the caller provides 64 writable UTF-32 units.
	unsafe { store_utf32_half(decoded, dst, written.min(32)) };
	if written > 32 {
		let high = _mm512_shuffle_i32x4::<0x4e>(indexes, indexes);
		// SAFETY: the low half now contains endpoint indexes 32 through 63.
		let decoded = unsafe {
			decode_utf16_half(
				_mm512_castsi512_si256(high),
				cleared,
				before_non_ascii,
				third_leads,
				three != 0,
			)
		};
		// SAFETY: the remaining count is at most 32.
		unsafe { store_utf32_half(decoded, dst.add(32), written - 32) };
	}
	Some((consumed, written))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn pack_two_u32(input: __m512i) -> __m512i {
	let lead = _mm512_or_si512(_mm512_srli_epi32::<6>(input), _mm512_set1_epi32(0x0000_00c0));
	let tail = _mm512_slli_epi32::<8>(_mm512_or_si512(
		_mm512_and_si512(input, _mm512_set1_epi32(0x0000_003f)),
		_mm512_set1_epi32(0x0000_0080),
	));
	_mm512_or_si512(lead, tail)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn pack_three_u32(input: __m512i) -> __m512i {
	let lead = _mm512_or_si512(_mm512_srli_epi32::<12>(input), _mm512_set1_epi32(0x0000_00e0));
	let middle = _mm512_slli_epi32::<8>(_mm512_or_si512(
		_mm512_and_si512(_mm512_srli_epi32::<6>(input), _mm512_set1_epi32(0x0000_003f)),
		_mm512_set1_epi32(0x0000_0080),
	));
	let tail = _mm512_slli_epi32::<16>(_mm512_or_si512(
		_mm512_and_si512(input, _mm512_set1_epi32(0x0000_003f)),
		_mm512_set1_epi32(0x0000_0080),
	));
	_mm512_or_si512(_mm512_or_si512(lead, middle), tail)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn pack_four_u32(input: __m512i) -> __m512i {
	let lead = _mm512_or_si512(
		_mm512_and_si512(_mm512_srli_epi32::<18>(input), _mm512_set1_epi32(0x0000_0007)),
		_mm512_set1_epi32(0x0000_00f0),
	);
	let second = _mm512_slli_epi32::<8>(_mm512_or_si512(
		_mm512_and_si512(_mm512_srli_epi32::<12>(input), _mm512_set1_epi32(0x0000_003f)),
		_mm512_set1_epi32(0x0000_0080),
	));
	let third = _mm512_slli_epi32::<16>(_mm512_or_si512(
		_mm512_and_si512(_mm512_srli_epi32::<6>(input), _mm512_set1_epi32(0x0000_003f)),
		_mm512_set1_epi32(0x0000_0080),
	));
	let tail = _mm512_slli_epi32::<24>(_mm512_or_si512(
		_mm512_and_si512(input, _mm512_set1_epi32(0x0000_003f)),
		_mm512_set1_epi32(0x0000_0080),
	));
	_mm512_or_si512(_mm512_or_si512(lead, second), _mm512_or_si512(third, tail))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn pack_mixed_bmp_u32(input: __m512i, ascii: u16, two: u16) -> __m512i {
	let groups = _mm512_multishift_epi64_epi8(_mm512_set1_epi64(0x2026_2c32_0006_0c12), input);
	let tags =
		_mm512_mask_mov_epi32(_mm512_set1_epi32(-0x7f7f_2000), two, _mm512_set1_epi32(-0x7f40_0000));
	let packed = _mm512_ternarylogic_epi32::<0xea>(groups, _mm512_set1_epi32(0x3f3f_3f3f), tags);
	_mm512_mask_slli_epi32::<24>(packed, ascii, input)
}
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn encode_utf16_one_two(input: __m512i, ascii: u32, dst: *mut u8) -> usize {
	let bits = _mm512_ternarylogic_epi32::<0xa8>(
		_mm512_slli_epi16::<8>(input),
		_mm512_srli_epi16::<6>(input),
		_mm512_set1_epi16(0x3f3f),
	);
	let packed = _mm512_mask_add_epi16(input, !ascii, bits, _mm512_set1_epi16(-32_576));
	let output = 0x5555_5555_5555_5555 | _pdep_u64(u64::from(!ascii), 0xaaaa_aaaa_aaaa_aaaa);
	// SAFETY: the output mask selects at most 64 bytes.
	unsafe { _mm512_mask_compressstoreu_epi8(dst.cast(), output, packed) };
	output.count_ones() as usize
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn encode_utf32_one_two(input: __m512i, ascii: u16, dst: *mut u8) -> usize {
	if ascii == u16::MAX {
		// SAFETY: narrowing writes exactly sixteen ASCII bytes.
		unsafe { _mm_storeu_si128(dst.cast(), _mm512_cvtepi32_epi8(input)) };
		return 16;
	}
	// SAFETY: the range check guarantees every non-ASCII lane uses two bytes.
	let packed = _mm512_mask_blend_epi32(ascii, unsafe { pack_two_u32(input) }, input);
	let lanes = 0x1111_1111_1111_1111;
	let output = lanes | (_pdep_u64(u64::from(!ascii), lanes) << 1);
	// SAFETY: the output mask selects at most 32 bytes.
	unsafe { _mm512_mask_compressstoreu_epi8(dst.cast(), output, packed) };
	output.count_ones() as usize
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn encode_utf32_bmp(input: __m512i, ascii: u16, two: u16, dst: *mut u8) -> usize {
	// SAFETY: the range check guarantees every lane uses at most three bytes.
	let packed = _mm512_mask_blend_epi32(two, unsafe { pack_three_u32(input) }, unsafe {
		pack_two_u32(input)
	});
	let packed = _mm512_mask_blend_epi32(ascii, packed, input);
	let lanes = 0x1111_1111_1111_1111;
	let output =
		lanes | (_pdep_u64(u64::from(!ascii), lanes) << 1) | (_pdep_u64(u64::from(!two), lanes) << 2);
	// SAFETY: the output mask selects at most 48 bytes.
	unsafe { _mm512_mask_compressstoreu_epi8(dst.cast(), output, packed) };
	output.count_ones() as usize
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn mixed_utf16_to_utf8(src: *const u16, dst: *mut u8) -> Option<usize> {
	// SAFETY: the public wrapper guarantees 32 readable UTF-16 units.
	let input = unsafe { load_u16x32(src) };
	let surrogate = _mm512_cmpgt_epu16_mask(input, _mm512_set1_epi16(-10241))
		& _mm512_cmplt_epu16_mask(input, _mm512_set1_epi16(-8192));
	if surrogate != 0 {
		return None;
	}
	let ascii = _mm512_cmplt_epu16_mask(input, _mm512_set1_epi16(0x0080));
	let two = _mm512_cmplt_epu16_mask(input, _mm512_set1_epi16(0x0800));
	if ascii == u32::MAX {
		// SAFETY: narrowing writes exactly 32 ASCII bytes.
		unsafe { _mm256_storeu_si256(dst.cast(), _mm512_cvtepi16_epi8(input)) };
		return Some(32);
	}
	if two == u32::MAX {
		// SAFETY: the range mask proves every lane uses one or two bytes.
		return Some(unsafe { encode_utf16_one_two(input, ascii, dst) });
	}
	let first = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(input));
	let high = _mm512_shuffle_i32x4::<0x4e>(input, input);
	let second = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(high));
	let first_ascii = mask16(u64::from(ascii), 0);
	let first_two = mask16(u64::from(two), 0);
	let second_ascii = mask16(u64::from(ascii), 16);
	let second_two = mask16(u64::from(two), 16);
	let first_ascii_bytes = _pdep_u64(u64::from(!first_ascii), 0x1111_1111_1111_1111);
	let first_two_bytes = _pdep_u64(u64::from(!first_two), 0x1111_1111_1111_1111);
	let second_ascii_bytes = _pdep_u64(u64::from(!second_ascii), 0x1111_1111_1111_1111);
	let second_two_bytes = _pdep_u64(u64::from(!second_two), 0x1111_1111_1111_1111);
	let first_mask = 0x8888_8888_8888_8888 | (first_ascii_bytes << 2) | (first_two_bytes << 1);
	let second_mask = 0x8888_8888_8888_8888 | (second_ascii_bytes << 2) | (second_two_bytes << 1);
	let first_written = first_mask.count_ones() as usize;
	let second_written = second_mask.count_ones() as usize;
	// SAFETY: both stores remain inside the caller's 96-byte scratch block.
	unsafe {
		_mm512_mask_compressstoreu_epi8(
			dst.cast(),
			first_mask,
			pack_mixed_bmp_u32(first, first_ascii, first_two),
		);
		_mm512_mask_compressstoreu_epi8(
			dst.add(first_written).cast(),
			second_mask,
			pack_mixed_bmp_u32(second, second_ascii, second_two),
		);
	}
	Some(first_written + second_written)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn mixed_utf32_to_utf8(src: *const u32, dst: *mut u8) -> Option<usize> {
	// SAFETY: the public wrapper guarantees 16 readable UTF-32 units.
	let input = unsafe { load_u32x16(src) };
	let ascii = _mm512_cmplt_epu32_mask(input, _mm512_set1_epi32(0x0000_0080));
	let two = _mm512_cmplt_epu32_mask(input, _mm512_set1_epi32(0x0000_0800));
	let three = _mm512_cmplt_epu32_mask(input, _mm512_set1_epi32(0x0001_0000));
	if ascii == u16::MAX {
		// SAFETY: narrowing writes exactly sixteen ASCII bytes.
		unsafe { _mm_storeu_si128(dst.cast(), _mm512_cvtepi32_epi8(input)) };
		return Some(16);
	}
	if two == u16::MAX {
		// SAFETY: the range mask proves every lane uses one or two bytes.
		return Some(unsafe { encode_utf32_one_two(input, ascii, dst) });
	}
	if two == 0 && three == u16::MAX {
		// SAFETY: compression writes exactly three bytes per input unit.
		unsafe {
			_mm512_mask_compressstoreu_epi8(dst.cast(), COMPRESS_THREE, pack_three_u32(input));
		}
		return Some(48);
	}
	if three == u16::MAX {
		// SAFETY: the range mask proves every lane uses at most three bytes.
		return Some(unsafe { encode_utf32_bmp(input, ascii, two, dst) });
	}
	if three == 0 {
		// SAFETY: every lane owns one four-byte output word.
		unsafe { _mm512_storeu_si512(dst.cast(), pack_four_u32(input)) };
		return Some(64);
	}
	let one = _mm512_and_si512(input, _mm512_set1_epi32(0x0000_007f));
	// SAFETY: the final masks discard values outside each selected range.
	let two_byte = unsafe { pack_two_u32(input) };
	let three_byte = unsafe { pack_three_u32(input) };
	let four_byte = unsafe { pack_four_u32(input) };
	let packed = _mm512_mask_blend_epi32(three, four_byte, three_byte);
	let packed = _mm512_mask_blend_epi32(two, packed, two_byte);
	let packed = _mm512_mask_blend_epi32(ascii, packed, one);
	let lanes = 0x1111_1111_1111_1111;
	let output = lanes
		| (_pdep_u64(u64::from(!ascii), lanes) << 1)
		| (_pdep_u64(u64::from(!two), lanes) << 2)
		| (_pdep_u64(u64::from(!three), lanes) << 3);
	let written = output.count_ones() as usize;
	// SAFETY: the output mask selects at most 64 bytes.
	unsafe { _mm512_mask_compressstoreu_epi8(dst.cast(), output, packed) };
	Some(written)
}
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn encode_utf16_prefix(
	src: *const u16,
	len: usize,
	dst: *mut u8,
	dst_len: usize,
) -> (usize, usize) {
	let (mut consumed, mut written) = (0, 0);
	while len - consumed >= 64 && dst_len - written >= 64 {
		// SAFETY: the loop bounds cover 64 source units.
		let first = unsafe { load_u16x32(src.add(consumed)) };
		// SAFETY: the loop bounds cover the adjacent source block.
		let second = unsafe { load_u16x32(src.add(consumed + 32)) };
		let combined = _mm512_or_si512(first, second);
		if _mm512_cmplt_epu16_mask(combined, _mm512_set1_epi16(0x0080)) != u32::MAX {
			break;
		}
		// SAFETY: the destination bound covers 64 narrowed bytes.
		unsafe {
			_mm256_storeu_si256(dst.add(written).cast(), _mm512_cvtepi16_epi8(first));
			_mm256_storeu_si256(dst.add(written + 32).cast(), _mm512_cvtepi16_epi8(second));
		}
		consumed += 64;
		written += 64;
	}
	while len - consumed >= 64 && dst_len - written >= 128 {
		// SAFETY: the loop bounds cover 64 source units.
		let first = unsafe { load_u16x32(src.add(consumed)) };
		// SAFETY: the loop bounds cover the adjacent source block.
		let second = unsafe { load_u16x32(src.add(consumed + 32)) };
		let combined = _mm512_or_si512(first, second);
		if _mm512_cmplt_epu16_mask(combined, _mm512_set1_epi16(0x0800)) != u32::MAX {
			break;
		}
		let first_ascii = _mm512_cmplt_epu16_mask(first, _mm512_set1_epi16(0x0080));
		let second_ascii = _mm512_cmplt_epu16_mask(second, _mm512_set1_epi16(0x0080));
		// SAFETY: the combined range check covers both blocks.
		let first_written = unsafe { encode_utf16_one_two(first, first_ascii, dst.add(written)) };
		// SAFETY: the destination has room for both worst-case blocks.
		let second_written =
			unsafe { encode_utf16_one_two(second, second_ascii, dst.add(written + first_written)) };
		consumed += 64;
		written += first_written + second_written;
	}
	while len - consumed >= 32 && dst_len - written >= 96 {
		// SAFETY: the loop bounds cover one source block and its maximum output.
		let Some(block) = (unsafe { mixed_utf16_to_utf8(src.add(consumed), dst.add(written)) })
		else {
			break;
		};
		consumed += 32;
		written += block;
	}
	(consumed, written)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn encode_utf32_prefix(
	src: *const u32,
	len: usize,
	dst: *mut u8,
	dst_len: usize,
) -> (usize, usize) {
	let (mut consumed, mut written) = (0, 0);
	while len - consumed >= 64 && dst_len - written >= 64 {
		// SAFETY: the loop bounds cover four source vectors.
		let a = unsafe { load_u32x16(src.add(consumed)) };
		let b = unsafe { load_u32x16(src.add(consumed + 16)) };
		let c = unsafe { load_u32x16(src.add(consumed + 32)) };
		let d = unsafe { load_u32x16(src.add(consumed + 48)) };
		let combined = _mm512_or_si512(_mm512_or_si512(a, b), _mm512_or_si512(c, d));
		if _mm512_cmplt_epu32_mask(combined, _mm512_set1_epi32(0x0080)) != u16::MAX {
			break;
		}
		// SAFETY: the destination bound covers 64 narrowed bytes.
		unsafe {
			_mm_storeu_si128(dst.add(written).cast(), _mm512_cvtepi32_epi8(a));
			_mm_storeu_si128(dst.add(written + 16).cast(), _mm512_cvtepi32_epi8(b));
			_mm_storeu_si128(dst.add(written + 32).cast(), _mm512_cvtepi32_epi8(c));
			_mm_storeu_si128(dst.add(written + 48).cast(), _mm512_cvtepi32_epi8(d));
		}
		consumed += 64;
		written += 64;
	}
	while len - consumed >= 64 && dst_len - written >= 128 {
		// SAFETY: the loop bounds cover four source vectors.
		let a = unsafe { load_u32x16(src.add(consumed)) };
		let b = unsafe { load_u32x16(src.add(consumed + 16)) };
		let c = unsafe { load_u32x16(src.add(consumed + 32)) };
		let d = unsafe { load_u32x16(src.add(consumed + 48)) };
		let combined = _mm512_or_si512(_mm512_or_si512(a, b), _mm512_or_si512(c, d));
		if _mm512_cmplt_epu32_mask(combined, _mm512_set1_epi32(0x0800)) != u16::MAX {
			break;
		}
		let ascii = [
			_mm512_cmplt_epu32_mask(a, _mm512_set1_epi32(0x0080)),
			_mm512_cmplt_epu32_mask(b, _mm512_set1_epi32(0x0080)),
			_mm512_cmplt_epu32_mask(c, _mm512_set1_epi32(0x0080)),
			_mm512_cmplt_epu32_mask(d, _mm512_set1_epi32(0x0080)),
		];
		// SAFETY: the combined range check covers every block.
		let wa = unsafe { encode_utf32_one_two(a, ascii[0], dst.add(written)) };
		let wb = unsafe { encode_utf32_one_two(b, ascii[1], dst.add(written + wa)) };
		let wc = unsafe { encode_utf32_one_two(c, ascii[2], dst.add(written + wa + wb)) };
		let wd = unsafe { encode_utf32_one_two(d, ascii[3], dst.add(written + wa + wb + wc)) };
		consumed += 64;
		written += wa + wb + wc + wd;
	}
	while len - consumed >= 16 && dst_len - written >= 64 {
		// SAFETY: the loop bounds cover one source block and its maximum output.
		let Some(block) = (unsafe { mixed_utf32_to_utf8(src.add(consumed), dst.add(written)) })
		else {
			break;
		};
		consumed += 16;
		written += block;
	}
	(consumed, written)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn classify_masks(input: __m512i) -> (u64, u64, u64) {
	let ascii = _mm512_cmplt_epu8_mask(input, _mm512_set1_epi8(-128));
	let leads = _mm512_cmpge_epu8_mask(input, _mm512_set1_epi8(-64));
	let continuation = !(ascii | leads);
	let three = _mm512_cmpge_epu8_mask(input, _mm512_set1_epi8(-32));
	(continuation, leads, three)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn classify_input(input: __m512i) -> (u64, u64, u64, bool) {
	// SAFETY: this function enables the same target features.
	let (continuation, leads, three) = unsafe { classify_masks(input) };
	let has_four = _mm512_cmpgt_epu8_mask(input, _mm512_set1_epi8(-17)) != 0;
	(continuation, leads, three, has_four)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn classify(src: *const u8) -> (u64, u64, u64, bool) {
	// SAFETY: the public wrapper guarantees one complete source block.
	let input = unsafe { load_u8x64(src) };
	// SAFETY: this function enables the same target features.
	unsafe { classify_input(input) }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi,avx512vbmi2,bmi2")]
unsafe fn decode_utf8_prefix<const UTF32: bool>(
	src: *const u8,
	len: usize,
	dst: *mut u8,
	dst_len: usize,
) -> (usize, usize) {
	let (mut consumed, mut written) = (0, 0);
	let input_limit = len.min(dst_len);
	while input_limit - consumed >= 64 {
		// SAFETY: the loop bound covers one complete source block.
		let input = unsafe { load_u8x64(src.add(consumed)) };
		// SAFETY: this function enables the same target features.
		let (continuation, leads, three) = unsafe { classify_masks(input) };
		if continuation == 0
			|| (three != 0 && _mm512_cmpgt_epu8_mask(input, _mm512_set1_epi8(-17)) != 0)
		{
			break;
		}
		let block = if UTF32 {
			// SAFETY: the loop bounds cover the fixed source and destination
			// blocks.
			unsafe { block_to_utf32(input, dst.add(written * 4).cast(), continuation, leads, three) }
		} else {
			// SAFETY: the loop bounds cover the fixed source and destination
			// blocks.
			unsafe { block_to_utf16(input, dst.add(written * 2).cast(), continuation, leads, three) }
		};
		let Some((used, output)) = block else {
			break;
		};
		if used == 0 {
			break;
		}
		consumed += used;
		written += output;
	}
	(consumed, written)
}

/// Classifies one UTF-8 block when AVX-512 VBMI2 is available.
#[inline(always)]
pub(crate) unsafe fn classify_block(src: *const u8) -> Option<(u64, u64, u64, bool)> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature.
		Some(unsafe { classify(src) })
	} else {
		None
	}
}

/// Decodes a maximal complete one-to-three-byte prefix to UTF-16.
#[inline(always)]
pub(crate) unsafe fn utf8_block_to_utf16(
	src: *const u8,
	dst: *mut u16,
	continuation: u64,
	leads: u64,
	three: u64,
) -> Option<(usize, usize)> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature, and the
		// masks came from [`classify_block`] for this source block.
		let input = unsafe { load_u8x64(src) };
		unsafe { block_to_utf16(input, dst, continuation, leads, three) }
	} else {
		None
	}
}

/// Decodes one complete one-to-three-byte block to UTF-32.
#[inline(always)]
pub(crate) unsafe fn utf8_block_to_utf32(
	src: *const u8,
	dst: *mut u32,
	continuation: u64,
	leads: u64,
	three: u64,
) -> Option<usize> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature, and the
		// masks came from [`classify_block`] for this source block.
		let input = unsafe { load_u8x64(src) };
		let (consumed, written) = unsafe { block_to_utf32(input, dst, continuation, leads, three) }?;
		(consumed == 64).then_some(written)
	} else {
		None
	}
}
/// Decodes heterogeneous one-to-three-byte UTF-8 into UTF-16 with one dispatch.
#[inline(always)]
pub(crate) unsafe fn utf8_mixed_prefix_to_utf16(
	src: *const u8,
	len: usize,
	dst: *mut u16,
	dst_len: usize,
) -> Option<(usize, usize)> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature.
		Some(unsafe { decode_utf8_prefix::<false>(src, len, dst.cast(), dst_len) })
	} else {
		None
	}
}

/// Decodes heterogeneous one-to-three-byte UTF-8 into UTF-32 with one dispatch.
#[inline(always)]
pub(crate) unsafe fn utf8_mixed_prefix_to_utf32(
	src: *const u8,
	len: usize,
	dst: *mut u32,
	dst_len: usize,
) -> Option<(usize, usize)> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature.
		Some(unsafe { decode_utf8_prefix::<true>(src, len, dst.cast(), dst_len) })
	} else {
		None
	}
}

/// Encodes a maximal non-surrogate UTF-16 prefix with one AVX-512 dispatch.
#[inline(always)]
pub(crate) unsafe fn utf16_to_utf8_prefix(
	src: *const u16,
	len: usize,
	dst: *mut u8,
	dst_len: usize,
) -> Option<(usize, usize)> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature.
		Some(unsafe { encode_utf16_prefix(src, len, dst, dst_len) })
	} else {
		None
	}
}

/// Encodes a maximal in-range UTF-32 prefix with one AVX-512 dispatch.
#[inline(always)]
pub(crate) unsafe fn utf32_to_utf8_prefix(
	src: *const u32,
	len: usize,
	dst: *mut u8,
	dst_len: usize,
) -> Option<(usize, usize)> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature.
		Some(unsafe { encode_utf32_prefix(src, len, dst, dst_len) })
	} else {
		None
	}
}

/// Encodes one heterogeneous non-surrogate UTF-16 block with byte compaction.
#[inline(always)]
pub(crate) unsafe fn utf16_mixed_32_to_utf8(src: *const u16, dst: *mut u8) -> Option<usize> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature.
		unsafe { mixed_utf16_to_utf8(src, dst) }
	} else {
		None
	}
}

/// Encodes one heterogeneous UTF-32 block with byte compaction.
#[inline(always)]
pub(crate) unsafe fn utf32_mixed_16_to_utf8(src: *const u32, dst: *mut u8) -> Option<usize> {
	if available() {
		// SAFETY: runtime detection proved every enabled feature.
		unsafe { mixed_utf32_to_utf8(src, dst) }
	} else {
		None
	}
}
