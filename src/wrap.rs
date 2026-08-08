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

use core::marker::PhantomData;

use crate::{encoding::Encoding, grapheme::next_cluster, unit::Unit, utf8::Utf8};

/// Iterator of wrapped lines, created by [`wrap`].
pub struct Wrapped<'a, E: Encoding> {
	rest:      &'a [E::Unit],
	max_width: usize,
	done:      bool,
	marker:    PhantomData<E>,
}

impl<E: Encoding> Clone for Wrapped<'_, E> {
	#[inline]
	fn clone(&self) -> Self {
		Self {
			rest:      self.rest,
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

impl<'a, E: Encoding> Iterator for Wrapped<'a, E> {
	type Item = &'a [E::Unit];

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.done {
			return None;
		}

		let s = self.rest;
		let space = unit_of::<E>(b' ');
		let (lf, cr) = (unit_of::<E>(b'\n'), unit_of::<E>(b'\r'));
		let mut pos = 0;
		let mut width = 0usize;
		// Start of the most recent breaking space run, and of the run that
		// currently reaches `pos` (`None` once a non-space cluster follows).
		// Trailing spaces are dropped by cutting at `trail` rather than by
		// scanning units backwards: a space unit can also be the tail of a
		// malformed multi-unit sequence, which must not be split.
		let mut candidate = None;
		let mut trail = None;

		loop {
			if pos == s.len() {
				self.done = true;
				return Some(&s[..trail.unwrap_or(pos)]);
			}

			let scan = next_cluster::<E>(&s[pos..]);
			let first = s[pos];
			if first == lf || first == cr {
				self.rest = &s[pos + scan.units..];
				return Some(&s[..trail.unwrap_or(pos)]);
			}

			if scan.units == 1 && first == space {
				if trail.is_none() {
					trail = Some(pos);
					candidate = Some(pos);
				}
			} else {
				trail = None;
			}

			if width.saturating_add(scan.width) > self.max_width && pos > 0 {
				if let Some(cut) = candidate {
					let mut resume = cut;
					while resume < s.len() {
						let next = next_cluster::<E>(&s[resume..]);
						if next.units == 1 && s[resume] == space {
							resume += 1;
						} else {
							break;
						}
					}
					self.rest = &s[resume..];
					return Some(&s[..cut]);
				}

				self.rest = &s[pos..];
				return Some(&s[..pos]);
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

/// Greedily wraps `input` into at least one borrowed terminal-width subline.
#[inline]
pub const fn wrap<E: Encoding>(input: &[E::Unit], max_width: usize) -> Wrapped<'_, E> {
	Wrapped { rest: input, max_width, done: false, marker: PhantomData }
}

/// Applies [`wrap`] to a UTF-8 string.
#[inline]
pub fn wrap_str(input: &str, max_width: usize) -> impl Iterator<Item = &str> + Clone {
	wrap::<Utf8>(input.as_bytes(), max_width).map(|line| {
		// SAFETY: line boundaries fall on cluster boundaries (char boundaries) in valid
		// UTF-8.
		unsafe { core::str::from_utf8_unchecked(line) }
	})
}
