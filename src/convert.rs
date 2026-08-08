//! Transcoding between encodings (`utf_convert` / `utf_length`).

use alloc::{
	string::{FromUtf8Error, String},
	vec,
	vec::Vec,
};

use crate::{
	encoding::{Encoding, Kind, same_encoding},
	simd::{NARROW, TRANS_WIDE, ascii_run},
	unit::{Unit, fold_case_scalar},
	utf8::Utf8,
};

/// Owned destination of [`Text::transcode`](crate::Text::transcode); pins the
/// target encoding through its code-unit type.
///
/// Implemented for [`String`] and `Vec<u8>` (UTF-8), `Vec<u16>` (UTF-16) and
/// `Vec<u32>` (UTF-32), all native-endian.
pub trait TextBuf: Sized {
	/// Code-unit type of this buffer; selects [`Unit::Native`] as the encoding
	/// transcoded into.
	type Unit: Unit;

	/// Wraps freshly transcoded units.
	///
	/// Units come from a permissive transcode, so [`String`] substitutes
	/// U+FFFD for anything that is not valid UTF-8.
	fn from_units(units: Vec<Self::Unit>) -> Self;
}

impl TextBuf for String {
	type Unit = u8;

	#[inline]
	fn from_units(units: Vec<u8>) -> Self {
		match Self::from_utf8(units) {
			Ok(s) => s,
			Err(e) => Self::from_utf8_lossy(&e.into_bytes()).into_owned(),
		}
	}
}

impl TextBuf for Vec<u8> {
	type Unit = u8;

	#[inline]
	fn from_units(units: Self) -> Self {
		units
	}
}

impl TextBuf for Vec<u16> {
	type Unit = u16;

	#[inline]
	fn from_units(units: Self) -> Self {
		units
	}
}

impl TextBuf for Vec<u32> {
	type Unit = u32;

	#[inline]
	fn from_units(units: Self) -> Self {
		units
	}
}

/// ASCII case transform applied while transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsciiCase {
	/// Copy as-is.
	Preserve,
	/// Fold `A-Z` to `a-z`.
	Lower,
	/// Fold `a-z` to `A-Z`.
	Upper,
}

impl AsciiCase {
	/// Fold base: subtracting it maps the source range onto `0..=25`.
	#[inline(always)]
	const fn base(self) -> u32 {
		match self {
			// Never matches when folding is disabled (0x00..=0x19 are not
			// reachable after the ASCII-alpha guard below), but Preserve is
			// filtered out before use anyway.
			Self::Preserve => u32::MAX,
			Self::Lower => 'A' as u32,
			Self::Upper => 'a' as u32,
		}
	}
}

/// Cross-width unit bridge; a no-op at machine level for equal widths.
#[inline(always)]
fn bridge<F: Unit, T: Unit>(u: F) -> T {
	T::from_u32(u.to_u32())
}

/// Transcodes as much of `src` into `dst` as fits, returning
/// `(units_read, units_written)`. Never splits a codepoint: encoding stops
/// at the last codepoint whose output fits.
///
/// `dst` past the written count may be clobbered with scratch by the SIMD
/// fast path.
///
/// When `F` and `T` are the same encoding and `case` is `Preserve`, this
/// degrades to a bulk copy of `min(src.len(), dst.len())` units and *can*
/// split a trailing codepoint, mirroring the original's memcpy shortcut.
#[inline]
pub fn transcode_into<F: Encoding, T: Encoding>(
	src: &[F::Unit],
	dst: &mut [T::Unit],
	case: AsciiCase,
) -> (usize, usize) {
	let same = const { same_encoding::<F, T>() };
	if same && case == AsciiCase::Preserve {
		let n = src.len().min(dst.len());
		for (d, s) in dst[..n].iter_mut().zip(&src[..n]) {
			*d = bridge(*s);
		}
		return (n, n);
	}

	if case == AsciiCase::Preserve
		&& let Some(result) = crate::native::transcode::<F, T>(src, dst)
	{
		return result;
	}

	// The SIMD path stores raw lane values; with byte-swapped units ASCII is
	// not lane-ASCII, so foreign encodings on either side go scalar.
	let simd_ok = const { !F::FOREIGN && !T::FOREIGN };
	let fold = case != AsciiCase::Preserve;
	let base = case.base();

	let mut i = 0;
	let mut o = 0;
	while i < src.len() {
		if simd_ok {
			let limit = (src.len() - i).min(dst.len() - o);
			let n = if limit >= TRANS_WIDE {
				ascii_run::<F::Unit, TRANS_WIDE>(&src[i..], limit, |v, at| {
					let mut w = F::Unit::cast::<T::Unit, TRANS_WIDE>(v);
					if fold {
						w = T::Unit::fold_case(w, base);
					}
					w.copy_to_slice(&mut dst[o + at..o + at + TRANS_WIDE]);
				})
			} else if limit >= NARROW {
				ascii_run::<F::Unit, NARROW>(&src[i..], limit, |v, at| {
					let mut w = F::Unit::cast::<T::Unit, NARROW>(v);
					if fold {
						w = T::Unit::fold_case(w, base);
					}
					w.copy_to_slice(&mut dst[o + at..o + at + NARROW]);
				})
			} else {
				0
			};
			i += n;
			o += n;
			if i == src.len() {
				break;
			}
		}

		if same {
			// Same encoding: copy the encoded run verbatim, folding only the
			// lead unit (ASCII alphas are always single-unit runs).
			let front = src[i];
			// A truncated tail is still one permissive codepoint; copy the
			// available run instead of leaving it unread.
			let len = F::run_length(front).min(src.len() - i);
			if dst.len() - o < len {
				break;
			}
			let mut lead = front.to_u32();
			if fold {
				let native = if F::FOREIGN {
					front.swap_bytes().to_u32()
				} else {
					lead
				};
				if native.wrapping_sub(base) <= 25 {
					let folded = F::Unit::from_u32(native ^ 0x20);
					lead = if F::FOREIGN {
						folded.swap_bytes().to_u32()
					} else {
						folded.to_u32()
					};
				}
			}
			dst[o] = T::Unit::from_u32(lead);
			for k in 1..len {
				dst[o + k] = bridge(src[i + k]);
			}
			i += len;
			o += len;
		} else {
			let mut cur = &src[i..];
			let mut cp = F::decode(&mut cur);
			let consumed = src.len() - i - cur.len();
			if fold {
				cp = fold_case_scalar(cp, base);
			}
			if dst.len() - o < T::encoded_length(cp) {
				break; // Unlike the C++, the unencoded codepoint stays unread.
			}
			o += T::encode(cp, &mut dst[o..]);
			i += consumed;
		}
	}
	(i, o)
}

/// Transcodes all of `src` into a freshly allocated unit vector.
#[inline]
pub fn transcode<F: Encoding, T: Encoding>(src: &[F::Unit]) -> Vec<T::Unit> {
	transcode_with_case::<F, T>(src, AsciiCase::Preserve)
}

/// [`transcode`] with ASCII case folding.
pub fn transcode_with_case<F: Encoding, T: Encoding>(
	src: &[F::Unit],
	case: AsciiCase,
) -> Vec<T::Unit> {
	let mut out = vec![T::Unit::default(); src.len() * T::MAX_UNITS];
	let (read, written) = transcode_into::<F, T>(src, &mut out, case);
	debug_assert_eq!(read, src.len());
	out.truncate(written);
	out
}

/// Unit count `src` would occupy once transcoded to `T` (`utf_length`).
pub fn transcoded_len<F: Encoding, T: Encoding>(src: &[F::Unit]) -> usize {
	// One unit in, one unit out — except UTF-16 across byte orders, where a
	// high surrogate followed by a non-surrogate decodes from two units and
	// re-encodes as one (byte-identical encodings take the bulk copy in
	// `transcode_into` and keep every unit).
	let unit_preserving = const {
		size_of::<F::Unit>() == size_of::<T::Unit>()
			&& (same_encoding::<F, T>() || F::KIND as u8 != Kind::Utf16 as u8)
	};
	if unit_preserving {
		return src.len();
	}
	let simd_ok = const { !F::FOREIGN };
	let mut n = 0;
	let mut i = 0;
	while i < src.len() {
		if simd_ok {
			let limit = src.len() - i;
			let c = if limit >= TRANS_WIDE {
				ascii_run::<F::Unit, TRANS_WIDE>(&src[i..], limit, |_, _| {})
			} else if limit >= NARROW {
				ascii_run::<F::Unit, NARROW>(&src[i..], limit, |_, _| {})
			} else {
				0
			};
			i += c;
			n += c;
			if i == src.len() {
				break;
			}
		}
		let mut cur = &src[i..];
		let cp = F::decode(&mut cur);
		i = src.len() - cur.len();
		n += T::encoded_length(cp);
	}
	n
}

/// Transcodes to an owned `String`.
///
/// # Errors
/// When the source decodes to invalid scalar values (e.g. lone surrogates),
/// which would make the resulting UTF-8 invalid.
pub fn to_string<F: Encoding>(src: &[F::Unit]) -> Result<String, FromUtf8Error> {
	String::from_utf8(transcode::<F, Utf8>(src))
}
