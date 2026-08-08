//! UTF-aware comparison (`utf_compare` family). Mixed-encoding operands are
//! compared by decoded codepoint; case-insensitive variants fold ASCII only.

use core::cmp::Ordering;

use crate::{
	encoding::{Encoding, same_encoding},
	simd::{CMP_WIDE, CmpStep, NARROW, ascii_cmp, ascii_eq_u8_u16, raw_eq_ignore_ascii_case},
	unit::{Unit, fold_case_scalar},
};

/// Folds ASCII uppercase to lowercase when `caseless`.
#[inline(always)]
const fn fold(cp: u32, caseless: bool) -> u32 {
	if caseless {
		fold_case_scalar(cp, 'A' as u32)
	} else {
		cp
	}
}

/// Transcodes a native UTF-8 prefix to UTF-16 and accepts it only when its
/// raw output matches the other operand. A miss is inconclusive because
/// permissive inputs can encode equal codepoints with different raw units.
///
/// Raw equality proves codepoint equality only when re-encoding preserved
/// decode boundaries. UTF-8 sources can carry surrogate *codepoints*
/// (CESU-style `ED A0 80 ED B0 80`), whose re-encoded units are raw-identical
/// to a genuine pair yet decode differently, so any chunk containing a
/// potential surrogate encoding (an `0xED` lead) is rejected wholesale.
/// `transcode` never splits codepoints, so every other surrogate in the
/// output is a proper pair produced by a single supplementary codepoint.
#[inline(always)]
fn transcode_equal_stream<F: Encoding, T: Encoding>(
	mut src: &[F::Unit],
	mut target: &[T::Unit],
	caseless: bool,
) -> Option<(usize, usize)> {
	if !matches!(F::KIND, crate::Kind::Utf8) || !matches!(T::KIND, crate::Kind::Utf16) {
		return None;
	}
	let mut scratch = [T::Unit::default(); 512];
	let mut total_read = 0;
	let mut total_written = 0;

	while !src.is_empty() && !target.is_empty() {
		let capacity = scratch.len().min(target.len());
		if capacity == 0 {
			break;
		}
		let (read, written) = crate::native::transcode::<F, T>(src, &mut scratch[..capacity])?;
		if read == 0 {
			break;
		}
		let chunk = &scratch[..written];
		let target_chunk = &target[..written];

		let has_surrogate = chunk.iter().any(|u| u.to_u32() & 0xf800 == 0xd800);
		if has_surrogate {
			let mut cur = &src[..read];
			while !cur.is_empty() {
				let cp = F::decode(&mut cur);
				if cp.wrapping_sub(0xd800) <= 0x7ff || cp > 0x10ffff {
					return None;
				}
			}
			let mut k = 0;
			while k < written {
				let u = chunk[k].to_u32();
				if u & 0xf800 == 0xd800 {
					if u >= 0xdc00 || k + 1 >= written || chunk[k + 1].to_u32() & 0xfc00 != 0xdc00 {
						return None;
					}
					k += 2;
				} else {
					k += 1;
				}
			}
		}

		let equal = if caseless {
			raw_eq_ignore_ascii_case::<T::Unit, CMP_WIDE>(chunk, target_chunk)
		} else {
			chunk == target_chunk
		};

		if !equal {
			return None;
		}

		total_read += read;
		total_written += written;
		src = &src[read..];
		target = &target[written..];
	}

	(total_read > 0).then_some((total_read, total_written))
}

/// Codepoint-order comparison core. Returns any value with the sign of the
/// result; `for_eq` allows early-out with an arbitrary non-zero value.
#[inline]
fn compare_impl<A: Encoding, B: Encoding>(
	mut a: &[A::Unit],
	mut b: &[B::Unit],
	caseless: bool,
	for_eq: bool,
) -> i64 {
	let same = const { same_encoding::<A, B>() };
	let simd_ok = const { !A::FOREIGN && !B::FOREIGN };

	// An equal raw sequence is always an equal codepoint sequence. Native
	// same-encoding caseless equality can additionally fold whole SIMD
	// vectors without stopping at non-ASCII units. Shortcut misses still
	// decode: permissive encodings allow different raw sequences (including
	// overlong UTF-8), and foreign-endian ASCII cannot be folded as raw units.
	if same && for_eq {
		// SAFETY: `same` implies `A::KIND == B::KIND`, and each kind pins a
		// single sealed unit type, so `A::Unit` and `B::Unit` are the same
		// primitive with identical layout.
		let b_units = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<A::Unit>(), b.len()) };
		let raw_equal = if caseless && !A::FOREIGN {
			raw_eq_ignore_ascii_case::<A::Unit, CMP_WIDE>(a, b_units)
		} else {
			a == b_units
		};
		if raw_equal {
			return 0;
		}
	}

	loop {
		let limit = a.len().min(b.len());
		if limit == 0 {
			break;
		}

		// ASCII prefix as SIMD.
		if simd_ok && limit >= NARROW {
			let mut adv = 0;
			let mixed_8_16 = for_eq
				&& ((A::KIND == crate::Kind::Utf8 && B::KIND == crate::Kind::Utf16)
					|| (A::KIND == crate::Kind::Utf16 && B::KIND == crate::Kind::Utf8));
			let step = if mixed_8_16 && A::KIND == crate::Kind::Utf8 {
				// SAFETY: the sealed encoding kinds pin UTF-8 units to `u8`
				// and UTF-16 units to `u16`; `simd_ok` excludes byte swapping.
				let a8 = unsafe { core::slice::from_raw_parts(a.as_ptr().cast::<u8>(), a.len()) };
				// SAFETY: same kind-to-unit invariant as above.
				let b16 = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<u16>(), b.len()) };
				if limit >= CMP_WIDE {
					ascii_eq_u8_u16::<CMP_WIDE>(a8, b16, limit, caseless, &mut adv)
				} else {
					ascii_eq_u8_u16::<NARROW>(a8, b16, limit, caseless, &mut adv)
				}
			} else if mixed_8_16 {
				// SAFETY: the sealed encoding kinds pin UTF-8 units to `u8`
				// and UTF-16 units to `u16`; `simd_ok` excludes byte swapping.
				let b8 = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<u8>(), b.len()) };
				// SAFETY: same kind-to-unit invariant as above.
				let a16 = unsafe { core::slice::from_raw_parts(a.as_ptr().cast::<u16>(), a.len()) };
				if limit >= CMP_WIDE {
					ascii_eq_u8_u16::<CMP_WIDE>(b8, a16, limit, caseless, &mut adv)
				} else {
					ascii_eq_u8_u16::<NARROW>(b8, a16, limit, caseless, &mut adv)
				}
			} else if limit >= CMP_WIDE {
				ascii_cmp::<A::Unit, B::Unit, CMP_WIDE>(a, b, limit, caseless, for_eq, &mut adv)
			} else {
				ascii_cmp::<A::Unit, B::Unit, NARROW>(a, b, limit, caseless, for_eq, &mut adv)
			};
			a = &a[adv..];
			b = &b[adv..];
			match step {
				CmpStep::Diff(d) => return d,
				CmpStep::Equal => continue,
				CmpStep::NonAscii => {},
			}
		}
		// For a native UTF-8/UTF-16 equality check, transcode a whole prefix
		// into a small stack buffer and compare its encoded form directly.
		// Equal canonical strings stay in the vector transcoders; a raw miss
		// deliberately falls through to permissive codepoint decoding.
		if for_eq && simd_ok && A::KIND == crate::Kind::Utf8 && B::KIND == crate::Kind::Utf16 {
			if let Some((read, written)) = transcode_equal_stream::<A, B>(a, b, caseless) {
				a = &a[read..];
				b = &b[written..];
				continue;
			}
		} else if for_eq && simd_ok && A::KIND == crate::Kind::Utf16 && B::KIND == crate::Kind::Utf8
			&& let Some((read, written)) = transcode_equal_stream::<B, A>(b, a, caseless) {
				a = &a[written..];
				b = &b[read..];
				continue;
			}

		// Scalar ASCII unit. Raw units only equal codepoints when both
		// encodings are native byte order, so foreign operands always take
		// the decoding paths below.
		let c1 = a[0].to_u32();
		let c2 = b[0].to_u32();
		if simd_ok && c1 <= 0x7f && c2 <= 0x7f {
			a = &a[1..];
			b = &b[1..];
			let (f1, f2) = if caseless {
				(fold_case_scalar(c1, 'A' as u32), fold_case_scalar(c2, 'A' as u32))
			} else {
				(c1, c2)
			};
			if f1 != f2 {
				return f1 as i64 - f2 as i64;
			}
			continue;
		}

		if same {
			// Same encoding: compare the encoded run unit-by-unit; only
			// decode when it mismatches.
			let len = A::run_length(a[0]);
			let truncated = limit < len;
			let mut mismatch = truncated || c1 != c2;
			if !truncated {
				for k in 1..len {
					mismatch |= a[k].to_u32() != b[k].to_u32();
				}
			}
			if mismatch {
				// Unlike the C++ (which returned the raw lead-unit delta and
				// thus UTF-16 code-unit order), order by decoded, folded
				// codepoint; mismatched units can still fold equal (e.g.
				// byte-swapped ASCII case pairs).
				let cp1 = fold(A::decode(&mut a), caseless);
				let cp2 = fold(B::decode(&mut b), caseless);
				if cp1 != cp2 {
					return if for_eq { 1 } else { cp1 as i64 - cp2 as i64 };
				}
				continue;
			}
			a = &a[len..];
			b = &b[len..];
		} else {
			let cp1 = fold(A::decode(&mut a), caseless);
			let cp2 = fold(B::decode(&mut b), caseless);
			if cp1 != cp2 {
				return if for_eq { 1 } else { cp1 as i64 - cp2 as i64 };
			}
		}
	}

	// A remaining side is a decodable suffix: longer sorts greater. (The C++
	// returned the next codepoint's value here, which misordered a trailing
	// NUL as equality.)
	if !a.is_empty() {
		1
	} else if !b.is_empty() {
		-1
	} else {
		0
	}
}

/// Compares two encoded strings in codepoint order.
#[inline]
pub fn compare<A: Encoding, B: Encoding>(a: &[A::Unit], b: &[B::Unit]) -> Ordering {
	compare_impl::<A, B>(a, b, false, false).cmp(&0)
}

/// [`compare`] with ASCII-only case folding (`utf_icompare`).
#[inline]
pub fn compare_ignore_ascii_case<A: Encoding, B: Encoding>(
	a: &[A::Unit],
	b: &[B::Unit],
) -> Ordering {
	compare_impl::<A, B>(a, b, true, false).cmp(&0)
}

/// Codepoint equality across encodings (`utf_cmpeq`).
#[inline]
pub fn equals<A: Encoding, B: Encoding>(a: &[A::Unit], b: &[B::Unit]) -> bool {
	if const { same_encoding::<A, B>() && !A::FOREIGN && !B::FOREIGN } {
		// SAFETY: same encoding kinds pin identical Unit types.
		let b_units = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<A::Unit>(), b.len()) };
		if crate::simd::raw_eq(a, b_units) {
			return true;
		}
	}
	compare_impl::<A, B>(a, b, false, true) == 0
}

/// [`equals`] with ASCII-only case folding (`utf_icmpeq`).
#[inline]
pub fn equals_ignore_ascii_case<A: Encoding, B: Encoding>(a: &[A::Unit], b: &[B::Unit]) -> bool {
	if const { same_encoding::<A, B>() && !A::FOREIGN && !B::FOREIGN } {
		// SAFETY: same encoding kinds pin identical Unit types.
		let b_units = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<A::Unit>(), b.len()) };
		if raw_eq_ignore_ascii_case::<A::Unit, CMP_WIDE>(a, b_units) {
			return true;
		}
	}
	compare_impl::<A, B>(a, b, true, true) == 0
}
