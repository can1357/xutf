//! SIMD ASCII fast-path kernels (`utf_ascii_enum` / `utf_ascii_cmp`).
//!
//! Both kernels use the overlapped-tail trick from the original: when fewer
//! than `N` units remain before `limit`, the window is moved back to
//! `limit - N` and the overlap is re-processed (loads are idempotent, stores
//! rewrite identical values).

use core::{
	mem::size_of,
	simd::{
		Simd,
		cmp::{SimdPartialEq, SimdPartialOrd},
	},
};

use crate::unit::Unit;

/// Wide transcode lane count: four NEON registers on ARM and two
/// register-resident x86 widening groups on Sapphire Rapids.
#[cfg(target_arch = "aarch64")]
pub const TRANS_WIDE: usize = 64;
#[cfg(target_arch = "x86_64")]
pub(crate) const TRANS_WIDE: usize = 64;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) const TRANS_WIDE: usize = 16;
pub const CMP_WIDE: usize = 16;
pub const NARROW: usize = 8;

/// Streams `N`-lane windows of `data[..limit]` through `f(window, offset)`
/// while they are all-ASCII, returning the length of the leading ASCII run.
///
/// `f` is also invoked on the window containing the first non-ASCII unit
/// (its clean prefix is live data; anything past the returned count is
/// scratch the caller must overwrite or discard).
#[inline(always)]
pub fn ascii_run<U: Unit, const N: usize>(
	data: &[U],
	limit: usize,
	mut f: impl FnMut(Simd<U, N>, usize),
) -> usize {
	debug_assert!(N <= limit && limit <= data.len());
	let mut i = 0;
	loop {
		let v = Simd::<U, N>::from_slice(&data[i..]);
		let bad = U::non_ascii_index(v);
		f(v, i);
		if let Some(p) = bad {
			return i + p;
		}
		i += N;
		if i + N > limit {
			if i == limit {
				return i;
			}
			i = limit - N;
		}
	}
}

/// Raw unit-slice equality tuned for long inputs: 256-byte blocks fold four
/// XOR/OR trees into a single reduction each, so the equal-buffer sweep
/// carries one branch per 256 bytes instead of a libc `memcmp` call's
/// per-chunk exits.
#[inline]
pub fn raw_eq<U: Unit>(left: &[U], right: &[U]) -> bool {
	if left.len() != right.len() {
		return false;
	}
	// SAFETY: unit types are plain integers; the byte view covers the same
	// memory with length scaled by the unit width.
	let left = unsafe { core::slice::from_raw_parts(left.as_ptr().cast::<u8>(), size_of_val(left)) };
	// SAFETY: as above.
	let right =
		unsafe { core::slice::from_raw_parts(right.as_ptr().cast::<u8>(), size_of_val(right)) };
	#[cfg(target_arch = "x86_64")]
	{
		// LLVM lowers equal byte slices to the platform `bcmp`/`memcmp`
		// implementation. That routine is runtime-multiversioned; a fixed
		// portable-SIMD loop is limited to the crate's baseline target and
		// loses roughly a third of throughput on AVX2/AVX-512 machines.
		left == right
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		let byte_len = left.len();
		let mut offset = 0;
		while byte_len - offset >= 256 {
			let difference = ((Simd::<u8, 64>::from_slice(&left[offset..])
				^ Simd::from_slice(&right[offset..]))
				| (Simd::<u8, 64>::from_slice(&left[offset + 64..])
					^ Simd::from_slice(&right[offset + 64..])))
				| ((Simd::<u8, 64>::from_slice(&left[offset + 128..])
					^ Simd::from_slice(&right[offset + 128..]))
					| (Simd::<u8, 64>::from_slice(&left[offset + 192..])
						^ Simd::from_slice(&right[offset + 192..])));
			if difference != Simd::splat(0) {
				return false;
			}
			offset += 256;
		}
		while byte_len - offset >= 128 {
			let difference = (Simd::<u8, 64>::from_slice(&left[offset..])
				^ Simd::from_slice(&right[offset..]))
				| (Simd::<u8, 64>::from_slice(&left[offset + 64..])
					^ Simd::from_slice(&right[offset + 64..]));
			if difference != Simd::splat(0) {
				return false;
			}
			offset += 128;
		}
		while byte_len - offset >= 16 {
			let difference =
				Simd::<u8, 16>::from_slice(&left[offset..]) ^ Simd::from_slice(&right[offset..]);
			if difference != Simd::splat(0) {
				return false;
			}
			offset += 16;
		}
		if offset == byte_len {
			return true;
		}
		if byte_len >= 16 {
			// Overlapped tail window; loads are idempotent.
			let difference = Simd::<u8, 16>::from_slice(&left[byte_len - 16..])
				^ Simd::from_slice(&right[byte_len - 16..]);
			return difference == Simd::splat(0);
		}
		left[offset..] == right[offset..]
	}
}
/// Tests raw code units for equality after folding ASCII uppercase units.
///
/// This is a complete equality shortcut, not an ASCII-prefix scan: non-ASCII
/// units are left unchanged. It is therefore sound for native same-encoding
/// strings, where an equal folded unit sequence necessarily decodes equally.
#[inline(always)]
pub fn raw_eq_ignore_ascii_case<U: Unit, const N: usize>(a: &[U], b: &[U]) -> bool {
	if a.len() != b.len() {
		return false;
	}
	if a.len() < N {
		return a.iter().zip(b).all(|(&x, &y)| {
			let x = x.to_u32();
			let y = y.to_u32();
			let fx = if x.wrapping_sub('A' as u32) <= 25 {
				x ^ 0x20
			} else {
				x
			};
			let fy = if y.wrapping_sub('A' as u32) <= 25 {
				y ^ 0x20
			} else {
				y
			};
			fx == fy
		});
	}

	let mut i = 0;
	loop {
		let va = U::fold_case(Simd::<U, N>::from_slice(&a[i..]), 'A' as u32);
		let vb = U::fold_case(Simd::<U, N>::from_slice(&b[i..]), 'A' as u32);
		if va != vb {
			return false;
		}
		i += N;
		if i + N > a.len() {
			if i == a.len() {
				return true;
			}
			i = a.len() - N;
		}
	}
}

/// Outcome of one [`ascii_cmp`] sweep.
pub enum CmpStep {
	/// Hit a non-ASCII unit; resume scalar at the advanced position.
	NonAscii,
	/// Both sides equal through `limit`.
	Equal,
	/// Mismatch: folded difference (or `1` when only equality was asked).
	Diff(i64),
}
/// Equality-only ASCII sweep specialized for native UTF-8 versus UTF-16.
///
/// Keeping the comparison in the `u16` domain avoids widening both operands
/// to `u32`, as the ordering-capable generic kernel must do.
#[inline(always)]
pub fn ascii_eq_u8_u16<const N: usize>(
	a: &[u8],
	b: &[u16],
	limit: usize,
	caseless: bool,
	advanced: &mut usize,
) -> CmpStep {
	debug_assert!(N <= limit && limit <= a.len() && limit <= b.len());
	let mut i = 0;
	let step = loop {
		let mut va = Simd::<u8, N>::from_slice(&a[i..]);
		let mut vb = Simd::<u16, N>::from_slice(&b[i..]);
		if u8::non_ascii_index(va).is_some() || u16::non_ascii_index(vb).is_some() {
			break CmpStep::NonAscii;
		}
		if caseless {
			va = u8::fold_case(va, 'A' as u32);
			vb = u16::fold_case(vb, 'A' as u32);
		}
		if u8::cast::<u16, N>(va).simd_ne(vb).any() {
			break CmpStep::Diff(1);
		}
		i += N;
		if i + N > limit {
			if i == limit {
				break CmpStep::Equal;
			}
			i = limit - N;
		}
	};
	*advanced = i;
	step
}

/// Compares `a[..limit]` to `b[..limit]` while both are ASCII, in `N`-lane
/// windows widened to `u32`. Writes the number of confirmed-equal leading
/// units to `advanced`.
#[inline(always)]
pub fn ascii_cmp<A: Unit, B: Unit, const N: usize>(
	a: &[A],
	b: &[B],
	limit: usize,
	caseless: bool,
	for_eq: bool,
	advanced: &mut usize,
) -> CmpStep {
	debug_assert!(N <= limit && limit <= a.len() && limit <= b.len());
	let mut i = 0;
	let step = loop {
		let va = A::widen(Simd::<A, N>::from_slice(&a[i..]));
		let vb = B::widen(Simd::<B, N>::from_slice(&b[i..]));

		// Unlike the C++ (which only checked the narrow side and could
		// false-match truncated wide units), require both sides ASCII.
		if (va | vb).simd_gt(Simd::splat(0x7f)).any() {
			break CmpStep::NonAscii;
		}

		let (fa, fb) = if caseless {
			(u32::fold_case(va, 'A' as u32), u32::fold_case(vb, 'A' as u32))
		} else {
			(va, vb)
		};

		let ne = fa.simd_ne(fb);
		if ne.any() {
			if for_eq {
				break CmpStep::Diff(1);
			}
			let p = ne.first_set().unwrap();
			break CmpStep::Diff(fa[p] as i64 - fb[p] as i64);
		}

		i += N;
		if i + N > limit {
			if i == limit {
				break CmpStep::Equal;
			}
			i = limit - N;
		}
	};
	*advanced = i;
	step
}

/// Length of the longest prefix of `data` whose units are all printable
/// single-cell ASCII (`0x20..=0x7E`).
#[inline]
pub fn plain_prefix<U: Unit>(data: &[U]) -> usize {
	// Scalar reject: cluster-scanning loops re-enter here once per cluster on
	// non-ASCII text; a head-unit check skips the vector setup entirely.
	if data
		.first()
		.is_none_or(|u| !(0x20..=0x7e).contains(&u.to_u32()))
	{
		return 0;
	}
	if size_of::<U>() == 1 {
		// A single-register probe catches the short ASCII spans in mixed text
		// without putting the long-run loop on a narrower stride. A clean
		// probe is deliberately re-read by the throughput path below.
		if data.len() >= CMP_WIDE {
			let v = Simd::<U, CMP_WIDE>::from_slice(data);
			if let Some(p) = U::non_plain_index(v) {
				return p;
			}
		}
		plain_prefix_n::<U, TRANS_WIDE>(data)
	} else {
		plain_prefix_n::<U, 16>(data)
	}
}

#[inline(always)]
fn plain_prefix_n<U: Unit, const N: usize>(data: &[U]) -> usize {
	let mut i = 0;
	while i + N <= data.len() {
		let v = Simd::<U, N>::from_slice(&data[i..i + N]);
		if let Some(p) = U::non_plain_index(v) {
			return i + p;
		}
		i += N;
	}
	while i < data.len() && (0x20..=0x7e).contains(&data[i].to_u32()) {
		i += 1;
	}
	i
}
