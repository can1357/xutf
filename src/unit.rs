//! Code-unit abstraction shared by the three encodings.

use core::simd::{Select, Simd, SimdCast, SimdElement, cmp::SimdPartialOrd, num::SimdUint};

mod sealed {
	pub trait Sealed {}
	impl Sealed for u8 {}
	impl Sealed for u16 {}
	impl Sealed for u32 {}
}

/// A UTF code unit: `u8` (UTF-8), `u16` (UTF-16) or `u32` (UTF-32).
///
/// Sealed; exists so the transcoding/comparison kernels can be generic over
/// unit width while keeping all SIMD operations on concrete element types.
pub trait Unit: sealed::Sealed + Copy + Eq + Default + SimdCast + SimdElement + 'static {
	/// Truncating conversion from a codepoint value.
	fn from_u32(v: u32) -> Self;
	/// Zero-extending conversion to a codepoint value.
	fn to_u32(self) -> u32;
	/// Reverses the byte order (identity for `u8`).
	fn swap_bytes(self) -> Self;
	/// Lane-wise cast directly to another code-unit type.
	fn cast<V: Unit, const N: usize>(v: Simd<Self, N>) -> Simd<V, N>;
	/// Lane-wise zero-extension to the `u32` codepoint domain.
	fn widen<const N: usize>(v: Simd<Self, N>) -> Simd<u32, N>;
	/// Lane-wise truncation from the `u32` codepoint domain.
	fn narrow<const N: usize>(v: Simd<u32, N>) -> Simd<Self, N>;
	/// Index of the first lane above `0x7F`, if any.
	fn non_ascii_index<const N: usize>(v: Simd<Self, N>) -> Option<usize>;
	/// Index of the first lane outside `0x20..=0x7E` (printable one-cell
	/// ASCII), if any.
	fn non_plain_index<const N: usize>(v: Simd<Self, N>) -> Option<usize>;
	/// ASCII case fold: XORs `0x20` into every lane within `base..=base + 25`
	/// (`base` is `'A'` to lowercase, `'a'` to uppercase).
	fn fold_case<const N: usize>(v: Simd<Self, N>, base: u32) -> Simd<Self, N>;
}

macro_rules! impl_unit {
	($t:ty) => {
		impl Unit for $t {
			#[inline(always)]
			fn from_u32(v: u32) -> Self {
				v as $t
			}

			#[inline(always)]
			fn to_u32(self) -> u32 {
				self as u32
			}

			#[inline(always)]
			fn swap_bytes(self) -> Self {
				<$t>::swap_bytes(self)
			}

			#[inline(always)]
			fn cast<V: Unit, const N: usize>(v: Simd<Self, N>) -> Simd<V, N> {
				v.cast::<V>()
			}

			#[inline(always)]
			fn widen<const N: usize>(v: Simd<Self, N>) -> Simd<u32, N> {
				v.cast::<u32>()
			}

			#[inline(always)]
			fn narrow<const N: usize>(v: Simd<u32, N>) -> Simd<Self, N> {
				v.cast::<$t>()
			}

			#[inline(always)]
			fn non_ascii_index<const N: usize>(v: Simd<Self, N>) -> Option<usize> {
				v.simd_gt(Simd::splat(0x7f)).first_set()
			}

			#[inline(always)]
			fn non_plain_index<const N: usize>(v: Simd<Self, N>) -> Option<usize> {
				// (v - 0x20) > 0x5E catches both bounds (wrapping lane sub).
				(v - Simd::splat(0x20))
					.simd_gt(Simd::splat(0x5e))
					.first_set()
			}

			#[inline(always)]
			fn fold_case<const N: usize>(v: Simd<Self, N>, base: u32) -> Simd<Self, N> {
				// v ^= ((v - base) <= 'z' - 'a') & 0x20  (wrapping lane sub).
				let alpha = (v - Simd::splat(base as $t)).simd_le(Simd::splat(25));
				alpha.select(v ^ Simd::splat(0x20), v)
			}
		}
	};
}

impl_unit!(u8);
impl_unit!(u16);
impl_unit!(u32);

/// Scalar ASCII case fold, mirroring [`Unit::fold_case`] on a single value.
#[inline(always)]
pub const fn fold_case_scalar(v: u32, base: u32) -> u32 {
	if v.wrapping_sub(base) <= 25 {
		v ^ 0x20
	} else {
		v
	}
}
