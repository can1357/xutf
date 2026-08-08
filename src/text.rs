//! The [`Text`] extension trait: the crate's terminal text primitives as
//! methods on `str` and native code-unit slices.

use crate::{
	compare,
	convert::{self, AsciiCase, TextBuf},
	encoding::{Codepoints, Encoding, codepoints},
	grapheme::{
		Grapheme, GraphemeIndices as IndexedGraphemes, Graphemes,
		StrGraphemeIndices as StrIndexedGraphemes, StrGraphemes, grapheme_indices,
		grapheme_indices_str, graphemes, graphemes_str,
	},
	truncate::{
		skip_columns, skip_columns_str, truncate, truncate_measured, truncate_measured_str,
		truncate_str,
	},
	unit::Unit,
	utf8::Utf8,
	utf16::Utf16,
	utf32::Utf32,
	width::{width, width_str, width_within, width_within_str},
	wrap::{
		StrWrapped, Wrapped, WrappedLine, WrappedMeasured as MeasuredWrapped, wrap, wrap_measured,
		wrap_measured_str, wrap_str,
	},
};

/// Terminal text primitives as methods on `str` and native code-unit slices.
///
/// Implemented for `str` (yielding `&str` fragments) and for native-endian
/// `[u8]` / `[u16]` / `[u32]` slices (UTF-8/16/32, yielding unit-slice
/// fragments). Foreign-endian data goes through the free functions with an
/// explicit encoding instead ([`graphemes`], [`width`], ...).
///
/// # Example
/// ```
/// use xutf::Text;
///
/// let s = "cafe\u{301} \u{1f44d}\u{1f3fd}";
/// assert_eq!(s.visible_width(), 7);
/// assert_eq!(s.graphemes().count(), 6);
/// assert_eq!(s.graphemes().next_back(), Some("\u{1f44d}\u{1f3fd}"));
/// assert_eq!(s.truncate_width(5), "cafe\u{301} ");
/// ```
pub trait Text {
	/// Native encoding of this text.
	type Encoding: Encoding;
	/// Cluster item yielded by [`Text::graphemes`] and
	/// [`Text::grapheme_indices`]: [`Grapheme`] for unit slices, `&str` for `str`.
	type Cluster<'a>: Copy
	where
		Self: 'a;
	/// Iterator returned by [`Text::graphemes`].
	type Graphemes<'a>: DoubleEndedIterator<Item = Self::Cluster<'a>> + ExactSizeIterator + Clone
	where
		Self: 'a;
	/// Iterator returned by [`Text::grapheme_indices`].
	type GraphemeIndices<'a>: DoubleEndedIterator<Item = (usize, Self::Cluster<'a>)>
		+ ExactSizeIterator
		+ Clone
	where
		Self: 'a;
	/// Iterator returned by [`Text::wrap`].
	type Wrapped<'a>: Iterator<Item = &'a Self> + Clone
	where
		Self: 'a;
	/// Iterator returned by [`Text::wrap_measured`].
	type WrappedMeasured<'a>: Iterator<Item = WrappedLine<'a, Self::Encoding>> + Clone
	where
		Self: 'a;

	/// Iterates extended grapheme clusters from either end (see
	/// [`graphemes`]).
	fn graphemes(&self) -> Self::Graphemes<'_>;
	/// Iterates grapheme clusters with byte offsets for `str` and code-unit
	/// offsets for native slices (see [`grapheme_indices`]).
	fn grapheme_indices(&self) -> Self::GraphemeIndices<'_>;
	/// Iterates raw, unvalidated codepoints (see [`codepoints`]).
	fn codepoints(&self) -> Codepoints<'_, Self::Encoding>;
	/// Visible terminal width in cells (see [`width`]).
	fn visible_width(&self) -> usize;
	/// Returns the width when it is at most `max_width`, or `None` otherwise
	/// (see [`width_within`]).
	fn width_within(&self, max_width: usize) -> Option<usize>;
	/// Longest prefix no wider than `max_width` cells, cut on a cluster
	/// boundary (see [`truncate`]).
	fn truncate_width(&self, max_width: usize) -> &Self;
	/// Longest fitting cluster-boundary prefix and its exact cell width (see
	/// [`truncate_measured`]).
	fn truncate_measured(&self, max_width: usize) -> (&Self, usize);
	/// Tail after skipping whole clusters covering `columns`, and the exact
	/// skipped cell width (see [`skip_columns`]).
	fn skip_columns(&self, columns: usize) -> (&Self, usize);
	/// Greedily wraps into borrowed sublines at most `max_width` cells wide
	/// (see [`wrap`]).
	fn wrap(&self, max_width: usize) -> Self::Wrapped<'_>;
	/// Greedily wraps while reporting each subline's code-unit offset and
	/// visible width (see [`wrap_measured`]).
	fn wrap_measured(&self, max_width: usize) -> Self::WrappedMeasured<'_>;

	/// Raw code units backing this text.
	fn as_units(&self) -> &[<Self::Encoding as Encoding>::Unit];

	/// Transcodes into an owned container, inferring the target encoding from
	/// the container's unit type ([`String`] and `Vec<u8>` → UTF-8, `Vec<u16>`
	/// → UTF-16, `Vec<u32>` → UTF-32, all native-endian).
	///
	/// Transcoding is permissive, so a [`String`] target substitutes U+FFFD for
	/// sequences that would make the result invalid UTF-8; [`to_string`]
	/// rejects them instead.
	#[inline]
	fn transcode<C: TextBuf>(&self) -> C {
		C::from_units(convert::transcode::<Self::Encoding, <C::Unit as Unit>::Native>(
			self.as_units(),
		))
	}

	/// Transcodes into `dst` as far as it fits, returning
	/// `(units_read, units_written)` (see [`transcode_into`]).
	#[inline]
	fn transcode_into<U: Unit>(&self, dst: &mut [U]) -> (usize, usize) {
		convert::transcode_into::<Self::Encoding, U::Native>(
			self.as_units(),
			dst,
			AsciiCase::Preserve,
		)
	}

	/// Unit count this text would occupy once transcoded to unit type `U` (see
	/// [`transcoded_len`]).
	#[inline]
	fn transcoded_len<U: Unit>(&self) -> usize {
		convert::transcoded_len::<Self::Encoding, U::Native>(self.as_units())
	}

	/// Codepoint equality against any native-endian text — `str`, `String`,
	/// unit slices or their `Vec`s (see [`equals`]).
	#[inline]
	fn eq_text<U: Unit>(&self, other: impl AsRef<[U]>) -> bool {
		compare::equals::<Self::Encoding, U::Native>(self.as_units(), other.as_ref())
	}

	/// [`eq_text`](Text::eq_text) with ASCII-only case folding (see
	/// [`equals_ignore_ascii_case`]).
	#[inline]
	fn eq_text_ignore_ascii_case<U: Unit>(&self, other: impl AsRef<[U]>) -> bool {
		compare::equals_ignore_ascii_case::<Self::Encoding, U::Native>(
			self.as_units(),
			other.as_ref(),
		)
	}
}

impl Text for str {
	type Cluster<'a> = &'a Self;
	type Encoding = Utf8;
	type GraphemeIndices<'a> = StrIndexedGraphemes<'a>;
	type Graphemes<'a> = StrGraphemes<'a>;
	type Wrapped<'a> = StrWrapped<'a>;
	type WrappedMeasured<'a> = MeasuredWrapped<'a, Utf8>;

	#[inline]
	fn graphemes(&self) -> StrGraphemes<'_> {
		graphemes_str(self)
	}

	#[inline]
	fn grapheme_indices(&self) -> Self::GraphemeIndices<'_> {
		grapheme_indices_str(self)
	}

	#[inline]
	fn codepoints(&self) -> Codepoints<'_, Utf8> {
		codepoints::<Utf8>(self.as_bytes())
	}

	#[inline]
	fn visible_width(&self) -> usize {
		width_str(self)
	}

	#[inline]
	fn width_within(&self, max_width: usize) -> Option<usize> {
		width_within_str(self, max_width)
	}

	#[inline]
	fn truncate_width(&self, max_width: usize) -> &Self {
		truncate_str(self, max_width)
	}

	#[inline]
	fn truncate_measured(&self, max_width: usize) -> (&Self, usize) {
		truncate_measured_str(self, max_width)
	}

	#[inline]
	fn skip_columns(&self, columns: usize) -> (&Self, usize) {
		skip_columns_str(self, columns)
	}

	#[inline]
	fn wrap(&self, max_width: usize) -> StrWrapped<'_> {
		wrap_str(self, max_width)
	}

	#[inline]
	fn wrap_measured(&self, max_width: usize) -> Self::WrappedMeasured<'_> {
		wrap_measured_str(self, max_width)
	}

	#[inline]
	fn as_units(&self) -> &[u8] {
		self.as_bytes()
	}
}

macro_rules! impl_text_for_units {
	($unit:ty, $encoding:ty) => {
		impl Text for [$unit] {
			type Cluster<'a> = Grapheme<'a, $encoding>;
			type Encoding = $encoding;
			type GraphemeIndices<'a> = IndexedGraphemes<'a, $encoding>;
			type Graphemes<'a> = Graphemes<'a, $encoding>;
			type Wrapped<'a> = Wrapped<'a, $encoding>;
			type WrappedMeasured<'a> = MeasuredWrapped<'a, $encoding>;

			#[inline]
			fn graphemes(&self) -> Graphemes<'_, $encoding> {
				graphemes::<$encoding>(self)
			}

			#[inline]
			fn grapheme_indices(&self) -> Self::GraphemeIndices<'_> {
				grapheme_indices::<$encoding>(self)
			}

			#[inline]
			fn codepoints(&self) -> Codepoints<'_, $encoding> {
				codepoints::<$encoding>(self)
			}

			#[inline]
			fn visible_width(&self) -> usize {
				width::<$encoding>(self)
			}

			#[inline]
			fn width_within(&self, max_width: usize) -> Option<usize> {
				width_within::<$encoding>(self, max_width)
			}

			#[inline]
			fn truncate_width(&self, max_width: usize) -> &[$unit] {
				truncate::<$encoding>(self, max_width)
			}

			#[inline]
			fn truncate_measured(&self, max_width: usize) -> (&[$unit], usize) {
				truncate_measured::<$encoding>(self, max_width)
			}

			#[inline]
			fn skip_columns(&self, columns: usize) -> (&[$unit], usize) {
				skip_columns::<$encoding>(self, columns)
			}

			#[inline]
			fn wrap(&self, max_width: usize) -> Wrapped<'_, $encoding> {
				wrap::<$encoding>(self, max_width)
			}

			#[inline]
			fn wrap_measured(&self, max_width: usize) -> Self::WrappedMeasured<'_> {
				wrap_measured::<$encoding>(self, max_width)
			}

			#[inline]
			fn as_units(&self) -> &[$unit] {
				self
			}
		}
	};
}

impl_text_for_units!(u8, Utf8);
impl_text_for_units!(u16, Utf16<false>);
impl_text_for_units!(u32, Utf32<false>);
