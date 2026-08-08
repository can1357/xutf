//! Export-only transcoding entry points for `scripts/asm.py`.
//!
//! Each wrapper has the same raw-buffer shape as simdutf and uses the output
//! capacity from `benches/throughput.rs`. Keeping the call to
//! [`xutf::transcode_into`] here makes rustc emit the complete inlined xutf
//! path under the selected target CPU.

use core::slice;

use xutf::{AsciiCase, Encoding, Utf8, Utf16, Utf32, transcode_into};

/// Transcodes raw unit buffers for the assembly inspection wrappers.
#[inline(always)]
unsafe fn transcode<F: Encoding, T: Encoding>(
	src: *const F::Unit,
	src_len: usize,
	dst: *mut T::Unit,
	dst_len: usize,
) -> usize {
	// SAFETY: each exported wrapper documents the source buffer contract.
	let src = unsafe { slice::from_raw_parts(src, src_len) };
	// SAFETY: each exported wrapper documents the destination buffer contract.
	let dst = unsafe { slice::from_raw_parts_mut(dst, dst_len) };
	transcode_into::<F, T>(src, dst, AsciiCase::Preserve).1
}

/// Exposes the inlined UTF-8 to UTF-16 path for assembly inspection.
///
/// # Safety
///
/// `src` must point to `len` readable bytes and `dst` to `len + 16` writable
/// UTF-16 units. `len + 16` must not overflow `usize`.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn xutf_utf8_to_utf16(src: *const u8, len: usize, dst: *mut u16) -> usize {
	// SAFETY: upheld by this function's safety contract.
	unsafe { transcode::<Utf8, Utf16>(src, len, dst, len + 16) }
}

/// Exposes the inlined UTF-16 to UTF-8 path for assembly inspection.
///
/// # Safety
///
/// `src` must point to `len` readable UTF-16 units and `dst` to `len * 4 + 16`
/// writable bytes. `len * 4 + 16` must not overflow `usize`.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn xutf_utf16_to_utf8(src: *const u16, len: usize, dst: *mut u8) -> usize {
	// SAFETY: upheld by this function's safety contract.
	unsafe { transcode::<Utf16, Utf8>(src, len, dst, len * 4 + 16) }
}

/// Exposes the inlined UTF-8 to UTF-32 path for assembly inspection.
///
/// # Safety
///
/// `src` must point to `len` readable bytes and `dst` to `len + 16` writable
/// UTF-32 units. `len + 16` must not overflow `usize`.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn xutf_utf8_to_utf32(src: *const u8, len: usize, dst: *mut u32) -> usize {
	// SAFETY: upheld by this function's safety contract.
	unsafe { transcode::<Utf8, Utf32>(src, len, dst, len + 16) }
}

/// Exposes the inlined UTF-32 to UTF-8 path for assembly inspection.
///
/// # Safety
///
/// `src` must point to `len` readable UTF-32 units and `dst` to `len * 4 + 16`
/// writable bytes. `len * 4 + 16` must not overflow `usize`.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn xutf_utf32_to_utf8(src: *const u32, len: usize, dst: *mut u8) -> usize {
	// SAFETY: upheld by this function's safety contract.
	unsafe { transcode::<Utf32, Utf8>(src, len, dst, len * 4 + 16) }
}

const fn main() {}
