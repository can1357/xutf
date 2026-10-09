//! The [`Encoding`] trait: per-encoding codepoint codec (`codepoint_cvt`).

use core::marker::PhantomData;

use crate::{convert::TextBuf, unit::Unit};

pub mod sealed {
	pub trait Sealed {}

	impl Sealed for crate::utf8::Utf8 {}
	impl<const FOREIGN: bool> Sealed for crate::utf16::Utf16<FOREIGN> {}
	impl<const FOREIGN: bool> Sealed for crate::utf32::Utf32<FOREIGN> {}
}

/// Encoding family, used to detect "same encoding" fast paths at
/// monomorphization time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
	Utf8,
	Utf16,
	Utf32,
}

/// A UTF encoding: a stateless codec between raw `u32` codepoints and code
/// units.
///
/// All operations are *permissive*: no validation is performed, decoding
/// never fails, and invalid input produces unspecified-but-safe codepoints
/// (truncated UTF-8 tails decode to `0`, lone UTF-16 surrogates pass
/// through). This trait is sealed because optimized same-encoding paths rely
/// on each [`Kind`] having exactly one code-unit type.
pub trait Encoding: sealed::Sealed + 'static {
	/// Code-unit type.
	type Unit: Unit;
	/// Canonical owned container type for this encoding.
	type Container: TextBuf<Unit = Self::Unit>;
	/// Encoding family.
	const KIND: Kind;
	/// `true` when code units are byte-swapped relative to native byte order.
	const FOREIGN: bool;
	/// Maximum code units per codepoint (`max_out`).
	const MAX_UNITS: usize;

	/// Length in units of the sequence starting with `front` (`rlength`).
	/// Always in `1..=MAX_UNITS`, even for garbage input.
	fn run_length(front: Self::Unit) -> usize;
	/// Length in units of `cp` once encoded (`length`). Assumes a valid
	/// codepoint outside the surrogate range.
	fn encoded_length(cp: u32) -> usize;
	/// Encodes `cp` into `out[..encoded_length(cp)]`, returning the unit
	/// count.
	///
	/// # Panics
	/// If `out.len() < encoded_length(cp)`.
	fn encode(cp: u32, out: &mut [Self::Unit]) -> usize;
	/// Decodes one codepoint from the front of `input`, advancing it by the
	/// consumed unit count.
	///
	/// # Panics
	/// If `input` is empty.
	fn decode(input: &mut &[Self::Unit]) -> u32;
	/// Decodes one codepoint from the back of `input`, shrinking it by the
	/// consumed unit count.
	///
	/// Permissive like [`decode`](Encoding::decode); a malformed tail consumes
	/// a single unit, so backward boundaries may disagree with forward ones on
	/// invalid input.
	///
	/// # Panics
	/// If `input` is empty.
	fn decode_back(input: &mut &[Self::Unit]) -> u32;

	/// Decodes a raw byte stream into [`Self::Container`](Encoding::Container),
	/// picking the source encoding from its BOM (defaulting to UTF-8).
	///
	/// # Example
	/// ```
	/// use xutf::{Encoding, Utf8, Utf16};
	///
	/// let bytes = b"\xef\xbb\xbfcaf\xc3\xa9";
	/// assert_eq!(Utf8::from_bytes(bytes), "café");
	/// // `<Utf16>` (not `Utf16`) so the endianness parameter takes its default.
	/// assert_eq!(<Utf16>::from_bytes(bytes), [b'c' as u16, b'a' as u16, b'f' as u16, 0xe9]);
	/// ```
	fn from_bytes(data: &[u8]) -> Self::Container
	where
		Self: Sized,
	{
		let units = crate::bytes::from_bytes::<Self>(data);
		TextBuf::from_units(units)
	}
}

/// `true` when `A` and `B` are byte-for-byte the same encoding.
#[inline(always)]
pub const fn same_encoding<A: Encoding, B: Encoding>() -> bool {
	A::KIND as u8 == B::KIND as u8 && A::FOREIGN == B::FOREIGN
}

/// `true` when transcoding `A` to `B` maps every unit to exactly one unit:
/// byte-identical encodings (a bulk copy) and UTF-32 in either byte order.
/// UTF-16 across byte orders is not, since a high surrogate followed by a
/// non-surrogate decodes from two units and re-encodes as one.
#[inline(always)]
pub const fn unit_preserving<A: Encoding, B: Encoding>() -> bool {
	size_of::<A::Unit>() == size_of::<B::Unit>()
		&& (same_encoding::<A, B>() || A::KIND as u8 != Kind::Utf16 as u8)
}

/// Iterator over the raw codepoints of an encoded slice.
///
/// Created by [`codepoints`].
pub struct Codepoints<'a, E: Encoding> {
	rest:      &'a [E::Unit],
	_encoding: PhantomData<E>,
}

impl<E: Encoding> Clone for Codepoints<'_, E> {
	#[inline(always)]
	fn clone(&self) -> Self {
		Self { rest: self.rest, _encoding: PhantomData }
	}
}

impl<E: Encoding> Iterator for Codepoints<'_, E> {
	type Item = u32;

	#[inline]
	fn next(&mut self) -> Option<u32> {
		if self.rest.is_empty() {
			None
		} else {
			Some(E::decode(&mut self.rest))
		}
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		// Each codepoint consumes 1..=MAX_UNITS units.
		(self.rest.len().div_ceil(E::MAX_UNITS), Some(self.rest.len()))
	}
}

impl<E: Encoding> DoubleEndedIterator for Codepoints<'_, E> {
	#[inline]
	fn next_back(&mut self) -> Option<u32> {
		if self.rest.is_empty() {
			None
		} else {
			Some(E::decode_back(&mut self.rest))
		}
	}
}

impl<E: Encoding> core::iter::FusedIterator for Codepoints<'_, E> {}

/// Iterates the raw (unvalidated) codepoints of `input`.
#[inline]
pub const fn codepoints<E: Encoding>(input: &[E::Unit]) -> Codepoints<'_, E> {
	Codepoints { rest: input, _encoding: PhantomData }
}

/// Iterates `input` as `char`s, substituting U+FFFD for unit sequences that
/// decode to invalid scalar values.
#[inline]
pub fn chars<E: Encoding>(input: &[E::Unit]) -> impl Iterator<Item = char> + '_ {
	codepoints::<E>(input).map(|cp| char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER))
}
