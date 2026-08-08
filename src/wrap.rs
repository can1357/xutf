//! Greedy terminal-cell word wrapping over borrowed encoded text.
//!
//! Break opportunities occur after runs of U+0020 spaces; soft breaks consume
//! the whole breaking run. LF, CRLF, and lone CR are hard breaks. Words wider
//! than the target break only at grapheme-cluster boundaries. Every yielded
//! line has trailing spaces trimmed, while leading spaces are preserved on the
//! first line and after hard breaks. Control units, including tabs, occupy zero
//! cells; callers should pre-expand tabs when tab stops matter.
//!
//! An overflowing space run becomes the break point before it is measured and
//! is swallowed completely. A cluster wider than the target is yielded alone,
//! including when the target is zero, so iteration always progresses. A final
//! hard newline produces a final empty line.
//!
//! [`wrap_measured`] additionally reports each line's original code-unit
//! offset and visible width. Offsets support cursor-to-line mapping in editors,
//! while widths support padding and alignment without measuring lines again.

use core::marker::PhantomData;

use crate::{encoding::Encoding, grapheme::next_cluster, unit::Unit, utf8::Utf8};

/// Iterator of wrapped lines, created by [`wrap`].
pub struct Wrapped<'a, E: Encoding> {
	inner: WrappedMeasured<'a, E>,
}

impl<E: Encoding> Clone for Wrapped<'_, E> {
	#[inline]
	fn clone(&self) -> Self {
		Self { inner: self.inner.clone() }
	}
}
/// A borrowed wrapped subline with its position and terminal cell width.
///
/// `at` is measured in code units from the original input. `width` excludes
/// trailing U+0020 spaces removed while wrapping.
pub struct WrappedLine<'a, E: Encoding> {
	/// Code units belonging to this subline.
	pub units: &'a [E::Unit],
	/// Code-unit offset of this subline in the original input.
	pub at:    usize,
	/// Terminal cells occupied by this subline.
	pub width: usize,
}

impl<E: Encoding> Clone for WrappedLine<'_, E> {
	#[inline(always)]
	fn clone(&self) -> Self {
		*self
	}
}

impl<E: Encoding> Copy for WrappedLine<'_, E> {}

impl WrappedLine<'_, Utf8> {
	/// Returns this UTF-8 subline as a string.
	#[inline(always)]
	pub const fn as_str(&self) -> &str {
		// SAFETY: line boundaries fall on cluster boundaries (char boundaries) in valid
		// UTF-8.
		unsafe { core::str::from_utf8_unchecked(self.units) }
	}
}

/// Iterator of wrapped lines with source offsets and visible widths, created
/// by [`wrap_measured`].
pub struct WrappedMeasured<'a, E: Encoding> {
	rest:      &'a [E::Unit],
	at:        usize,
	max_width: usize,
	done:      bool,
	marker:    PhantomData<E>,
}

impl<E: Encoding> Clone for WrappedMeasured<'_, E> {
	#[inline]
	fn clone(&self) -> Self {
		Self {
			rest:      self.rest,
			at:        self.at,
			max_width: self.max_width,
			done:      self.done,
			marker:    PhantomData,
		}
	}
}

#[inline(always)]
fn unit_of<E: Encoding>(ascii: u8) -> E::Unit {
	let unit = E::Unit::from_u32(ascii as u32);
	if E::FOREIGN { unit.swap_bytes() } else { unit }
}

// An ASCII unit can bypass general grapheme segmentation when the following
// unit is ASCII too (or there is no following unit). Keep the final ASCII unit
// before a non-ASCII unit on the general path: an extending codepoint may join
// it.
#[inline(always)]
fn plain_ascii<E: Encoding>(input: &[E::Unit], pos: usize) -> Option<u32> {
	let native = |unit: E::Unit| {
		let unit = if E::FOREIGN { unit.swap_bytes() } else { unit };
		unit.to_u32()
	};
	let unit = native(input[pos]);
	(unit < 0x80 && (pos + 1 == input.len() || native(input[pos + 1]) < 0x80)).then_some(unit)
}

#[inline(always)]
fn skip_breaking_spaces<E: Encoding>(input: &[E::Unit], mut pos: usize, space: E::Unit) -> usize {
	while pos < input.len() {
		if plain_ascii::<E>(input, pos) == Some(b' ' as u32) {
			pos += 1;
			continue;
		}
		let scan = next_cluster::<E>(&input[pos..]);
		if scan.units != 1 || input[pos] != space {
			break;
		}
		pos += 1;
	}
	pos
}

impl<'a, E: Encoding> Iterator for WrappedMeasured<'a, E> {
	type Item = WrappedLine<'a, E>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.done {
			return None;
		}

		let s = self.rest;
		let at = self.at;
		let space = unit_of::<E>(b' ');
		let (lf, cr) = (unit_of::<E>(b'\n'), unit_of::<E>(b'\r'));
		let mut pos = 0;
		let mut width = 0usize;
		// Start and measured width of the most recent breaking space run, and
		// of the run that currently reaches `pos` (`None` once a non-space
		// cluster follows). Trailing spaces are dropped by cutting at `trail`
		// rather than by scanning units backwards: a space unit can also be
		// the tail of a malformed multi-unit sequence, which must not be split.
		let mut candidate = None;
		let mut trail = None;

		loop {
			if pos == s.len() {
				self.done = true;
				let (end, line_width) = trail.unwrap_or((pos, width));
				return Some(WrappedLine { units: &s[..end], at, width: line_width });
			}

			if let Some(ascii) = plain_ascii::<E>(s, pos) {
				if ascii == b'\n' as u32 || ascii == b'\r' as u32 {
					let consumed = pos
						+ if ascii == b'\r' as u32 && pos + 1 < s.len() && s[pos + 1] == lf {
							2
						} else {
							1
						};
					self.rest = &s[consumed..];
					self.at += consumed;
					let (end, line_width) = trail.unwrap_or((pos, width));
					return Some(WrappedLine { units: &s[..end], at, width: line_width });
				}

				if ascii == b' ' as u32 {
					let start = pos;
					while pos < s.len() && plain_ascii::<E>(s, pos) == Some(b' ' as u32) {
						pos += 1;
					}
					let cells = pos - start;
					if trail.is_none() {
						trail = Some((start, width));
						candidate = Some((start, width));
					}
					if width.saturating_add(cells) > self.max_width && (start > 0 || cells > 1) {
						let (cut, line_width) = candidate.unwrap();
						let resume = skip_breaking_spaces::<E>(s, cut, space);
						self.rest = &s[resume..];
						self.at += resume;
						return Some(WrappedLine { units: &s[..cut], at, width: line_width });
					}
					width = width.saturating_add(cells);
					continue;
				}

				trail = None;
				if (0x21..=0x7e).contains(&ascii) {
					let start = pos;
					while pos < s.len()
						&& plain_ascii::<E>(s, pos).is_some_and(|unit| (0x21..=0x7e).contains(&unit))
					{
						pos += 1;
					}
					let cells = pos - start;
					if width.saturating_add(cells) > self.max_width {
						if start > 0 {
							if let Some((cut, line_width)) = candidate {
								let resume = skip_breaking_spaces::<E>(s, cut, space);
								self.rest = &s[resume..];
								self.at += resume;
								return Some(WrappedLine { units: &s[..cut], at, width: line_width });
							}
							let take = self.max_width.saturating_sub(width);
							let cut = start + take;
							self.rest = &s[cut..];
							self.at += cut;
							return Some(WrappedLine { units: &s[..cut], at, width: width + take });
						}

						let take = self.max_width.saturating_sub(width).max(1);
						if take < cells {
							let cut = start + take;
							self.rest = &s[cut..];
							self.at += cut;
							return Some(WrappedLine {
								units: &s[..cut],
								at,
								width: width.saturating_add(take),
							});
						}
					}
					width = width.saturating_add(cells);
					continue;
				}

				// Other ASCII controls and DEL are zero-width clusters. Consume
				// a whole run without entering grapheme segmentation.
				while pos < s.len()
					&& plain_ascii::<E>(s, pos).is_some_and(|unit| {
						unit != b' ' as u32
							&& unit != b'\n' as u32
							&& unit != b'\r' as u32
							&& !(0x21..=0x7e).contains(&unit)
					}) {
					pos += 1;
				}
				continue;
			}

			let scan = next_cluster::<E>(&s[pos..]);
			let first = s[pos];
			if first == lf || first == cr {
				let consumed = pos + scan.units;
				self.rest = &s[consumed..];
				self.at += consumed;
				let (end, line_width) = trail.unwrap_or((pos, width));
				return Some(WrappedLine { units: &s[..end], at, width: line_width });
			}

			if scan.units == 1 && first == space {
				if trail.is_none() {
					trail = Some((pos, width));
					candidate = Some((pos, width));
				}
			} else {
				trail = None;
			}

			if width.saturating_add(scan.width) > self.max_width && pos > 0 {
				if let Some((cut, line_width)) = candidate {
					let resume = skip_breaking_spaces::<E>(s, cut, space);
					self.rest = &s[resume..];
					self.at += resume;
					return Some(WrappedLine { units: &s[..cut], at, width: line_width });
				}

				self.rest = &s[pos..];
				self.at += pos;
				return Some(WrappedLine { units: &s[..pos], at, width });
			}

			pos += scan.units;
			width = width.saturating_add(scan.width);
		}
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		(usize::from(!self.done), Some(self.rest.len() + 1))
	}
}

impl<'a, E: Encoding> Iterator for Wrapped<'a, E> {
	type Item = &'a [E::Unit];

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.inner.next().map(|line| line.units)
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}
}

/// Greedily wraps `input` into at least one borrowed terminal-width subline.
#[inline]
pub const fn wrap<E: Encoding>(input: &[E::Unit], max_width: usize) -> Wrapped<'_, E> {
	Wrapped {
		inner: WrappedMeasured { rest: input, at: 0, max_width, done: false, marker: PhantomData },
	}
}

/// Greedily wraps `input`, reporting each subline's offset and visible width.
///
/// The offset is measured in code units from `input`; the width excludes
/// trailing spaces removed by wrapping.
#[inline]
pub const fn wrap_measured<E: Encoding>(
	input: &[E::Unit],
	max_width: usize,
) -> WrappedMeasured<'_, E> {
	WrappedMeasured { rest: input, at: 0, max_width, done: false, marker: PhantomData }
}

/// Iterator of wrapped `&str` lines, created by [`wrap_str`].
#[derive(Clone)]
pub struct StrWrapped<'a> {
	inner: Wrapped<'a, Utf8>,
}

impl<'a> Iterator for StrWrapped<'a> {
	type Item = &'a str;

	#[inline]
	fn next(&mut self) -> Option<&'a str> {
		// SAFETY: line boundaries fall on cluster boundaries (char boundaries)
		// in valid UTF-8.
		self
			.inner
			.next()
			.map(|line| unsafe { core::str::from_utf8_unchecked(line) })
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}
}

/// Applies [`wrap`] to a UTF-8 string.
#[inline]
pub const fn wrap_str(input: &str, max_width: usize) -> StrWrapped<'_> {
	StrWrapped { inner: wrap::<Utf8>(input.as_bytes(), max_width) }
}

/// Applies [`wrap_measured`] to a UTF-8 string.
///
/// Offsets are UTF-8 byte offsets, and each yielded width excludes trailing
/// spaces removed by wrapping.
#[inline]
pub const fn wrap_measured_str(input: &str, max_width: usize) -> WrappedMeasured<'_, Utf8> {
	wrap_measured::<Utf8>(input.as_bytes(), max_width)
}
