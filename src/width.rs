//! Visible terminal width without ANSI interpretation or allocation.
//!
//! Widths follow kitty's algorithm for splitting text into cells (the "text
//! sizing protocol" specification): text splits into extended grapheme
//! clusters (UAX #29), and a cluster takes the cells of its first codepoint
//! ([`width_char`]). Later codepoints add none, except that U+FE0F widens a
//! one-cell emoji presentation base ([`is_emoji_presentation_base`]), U+FE0E
//! narrows a two-cell one, and Thai and Lao AM (spacing marks with a width of
//! their own) widen a one-cell base. A SIMD path counts printable ASCII in
//! bulk. Controls, including `\t`, have width zero; callers should expand tabs
//! before measuring when tab stops matter.
//! Bounded measurement can stop as soon as a limit is exceeded, while
//! per-character measurement reports the standalone width of one scalar.

use core::simd::{
	Mask, Simd,
	cmp::{SimdPartialEq, SimdPartialOrd},
};

use crate::{
	encoding::Encoding,
	grapheme::{ClusterState, width_value},
	kernel::{decode_utf8_3x16, load_utf8_3x16},
	props::{CB_LV, CB_LVT, CB_MASK, CB_OTHER, CB_OTHER_INCB_CONSONANT, props},
	simd::plain_prefix,
	unit::Unit,
	utf8::Utf8,
};

/// `true` for codepoints that take cells and provably form single-codepoint
/// clusters next to each other: break class Other (`InCB` consonants and
/// `Extended_Pictographic` included) or a precomposed Hangul LV/LVT syllable.
/// Two adjacent such codepoints always have a boundary between them (`GB9c`
/// needs a linker between consonants, `GB11` a ZWJ between pictographs), so
/// runs are summed without join-state churn, and what follows a run joins
/// as it would join its last codepoint alone. See [`simple_width`].
#[inline(always)]
const fn simple(p: u8) -> bool {
	matches!(p & CB_MASK, CB_OTHER | CB_OTHER_INCB_CONSONANT | CB_LV | CB_LVT) && width_value(p) != 0
}

/// Visible width of `input` in terminal cells, kitty's: per extended grapheme
/// cluster, with an ASCII bulk path.
pub fn width<E: Encoding>(input: &[E::Unit]) -> usize {
	measure_width::<E, false>(input, 0).expect("unbounded width scan cannot fail")
}

#[inline(always)]
fn commit_width<const BOUNDED: bool>(measured: &mut usize, width: usize, max_width: usize) -> bool {
	if BOUNDED && width > max_width - *measured {
		false
	} else {
		*measured += width;
		true
	}
}

/// Shared flat scanner for full and bounded measurement. `BOUNDED` lets the
/// optimizer erase all limit bookkeeping from [`width`].
fn measure_width<E: Encoding, const BOUNDED: bool>(
	input: &[E::Unit],
	max_width: usize,
) -> Option<usize> {
	let mut measured = 0usize;
	let mut rest = input;

	'bulk: loop {
		if !E::FOREIGN {
			// A bounded scan inspects at most budget + 1 plain units, retaining
			// early exit on long ASCII. An unbounded scan takes the whole run.
			let window = if BOUNDED {
				let budget = max_width - measured;
				rest.len().min(budget.saturating_add(1))
			} else {
				rest.len()
			};
			let run = plain_prefix(&rest[..window]);
			if run > 0 {
				// A non-ASCII successor may join the run's last unit, whose
				// cluster then needs the join state: VS16 widens `#`, `*` and
				// the digits, Thai and Lao AM widen any of them.
				let take = if run < rest.len() && rest[run].to_u32() >= 0x80 {
					run - 1
				} else {
					run
				};
				if !commit_width::<BOUNDED>(&mut measured, take, max_width) {
					return None;
				}
				rest = &rest[take..];
			}
		}
		if rest.is_empty() {
			return Some(measured);
		}

		let mut cp = E::decode(&mut rest);
		let mut p = props(cp);
		'cluster: loop {
			let mut state = 'simple: {
				// Simple runs need one decode and one property load per scalar.
				// A successor confirms the boundary of the held codepoint.
				while simple(p) {
					if rest.is_empty() {
						return commit_width::<BOUNDED>(&mut measured, width_value(p), max_width)
							.then_some(measured);
					}
					let mut peek = rest;
					let next = E::decode(&mut peek);
					if !E::FOREIGN && next.wrapping_sub(0x20) <= 0x5e {
						if !commit_width::<BOUNDED>(&mut measured, width_value(p), max_width) {
							return None;
						}
						continue 'bulk;
					}
					let np = props(next);
					if !simple(np) {
						let mut state = ClusterState::start(cp, p);
						rest = peek;
						if state.try_join(next, np) {
							break 'simple state;
						}
						if !commit_width::<BOUNDED>(&mut measured, state.finish(), max_width) {
							return None;
						}
						break 'simple ClusterState::start(next, np);
					}
					if !commit_width::<BOUNDED>(&mut measured, width_value(p), max_width) {
						return None;
					}
					cp = next;
					p = np;
					rest = peek;
				}
				break 'simple ClusterState::start(cp, p);
			};

			// The rejected boundary scalar is retained as the next base, so
			// cluster transitions never decode or load its properties twice.
			loop {
				if rest.is_empty() {
					return commit_width::<BOUNDED>(&mut measured, state.finish(), max_width)
						.then_some(measured);
				}
				let mut peek = rest;
				let next = E::decode(&mut peek);
				if !E::FOREIGN && next.wrapping_sub(0x20) <= 0x5e && !state.joins_plain() {
					if !commit_width::<BOUNDED>(&mut measured, state.finish(), max_width) {
						return None;
					}
					continue 'bulk;
				}
				let np = props(next);
				if state.try_join(next, np) {
					rest = peek;
					continue;
				}
				if !commit_width::<BOUNDED>(&mut measured, state.finish(), max_width) {
					return None;
				}
				rest = peek;
				if simple(np) {
					cp = next;
					p = np;
					continue 'cluster;
				}
				state = ClusterState::start(next, np);
			}
		}
	}
}

/// [`width`] over UTF-8 `str`.
#[inline]
pub fn width_str(input: &str) -> usize {
	width::<Utf8>(input.as_bytes())
}

/// Visible width of `input` when it is at most `max_width`, or `None` once the
/// running width exceeds the bound.
///
/// Unlike `width(input) <= max_width`, this stops at the first cluster that
/// proves the bound is exceeded, avoiding a full scan of long inputs when the
/// budget is small. Following zero-width clusters are still scanned when the
/// running width is exactly at the bound.
pub fn width_within<E: Encoding>(input: &[E::Unit], max_width: usize) -> Option<usize> {
	measure_width::<E, true>(input, max_width)
}

/// [`width_within`] over UTF-8 `str`.
#[inline]
pub fn width_within_str(input: &str, max_width: usize) -> Option<usize> {
	width_within::<Utf8>(input.as_bytes(), max_width)
}

/// Standalone terminal-cell width of `c`, kitty's.
///
/// Two cells for regional indicators, East Asian Wide and Fullwidth
/// characters and emoji with emoji presentation; zero for marks, format
/// characters, other default-ignorables, controls, surrogates and
/// noncharacters; one for everything else, ambiguous, private use and
/// unassigned code points included. This is also the width of a cluster `c`
/// starts, before the variation selector and Thai/Lao AM adjustments the
/// string APIs apply.
///
/// ```
/// assert_eq!(xutf::width_char('中'), 2);
/// assert_eq!(xutf::width_char('\u{1f1fa}'), 2); // a lone regional indicator
/// assert_eq!(xutf::width_char('\u{93e}'), 0); // a spacing combining mark
/// ```
#[inline]
pub fn width_char(c: char) -> usize {
	width_value(props(c as u32))
}

/// The cells `c` takes when it is *simple*, 0 when it is not.
///
/// A simple character takes one or two cells and is a grapheme cluster of its
/// own between any two simple characters: letters, digits, punctuation,
/// symbols, CJK ideographs, precomposed Hangul syllables, Indic consonants and
/// emoji, but not marks, joiners, variation selectors, controls, regional
/// indicators, prepended characters or conjoining jamo. A run of simple
/// characters is therefore as wide as the sum of their [`width_char`]s, and
/// segmentation after the run depends only on its last character: the fast
/// path of [`width`], and of a terminal printing text.
///
/// ```
/// assert_eq!(xutf::simple_width('a'), 1);
/// assert_eq!(xutf::simple_width('中'), 2);
/// assert_eq!(xutf::simple_width('\u{301}'), 0); // joins the character before
/// assert_eq!(xutf::simple_width('\u{1f1fa}'), 0); // pairs with the next one
/// ```
#[inline]
pub fn simple_width(c: char) -> usize {
	let p = props(c as u32);
	if simple(p) { width_value(p) } else { 0 }
}

/// How many characters [`wide_run`] takes at once.
pub const WIDE_RUN: usize = 16;

/// Ranges of code points that are all simple and two cells wide
/// ([`simple_width`]): CJK punctuation up to the tone marks, ideographs
/// (extension A and the unified block), Hangul syllables and the fullwidth
/// forms. [`wide_run`] takes a character inside them without the property
/// table.
const WIDE_RANGES: [(u16, u16); 5] =
	[(0x3000, 0x3029), (0x3400, 0x4dbf), (0x4e00, 0x9fff), (0xac00, 0xd7a3), (0xff01, 0xff60)];

/// Decodes the simple two-cell characters leading `bytes`, up to
/// [`WIDE_RUN`] of them, together.
///
/// The characters are the three-byte ones [`simple_width`] gives two
/// cells: CJK ideographs and punctuation, Hangul syllables, fullwidth
/// forms. Their code points go to `points` and their count is returned.
/// One interleaved load splits the bytes into lanes and the characters are
/// classified with no branch per character, for a terminal to write a cell
/// and its spacer each. Malformed sequences end the run like any other
/// character.
///
/// ```
/// let mut points = [0; xutf::WIDE_RUN];
/// assert_eq!(xutf::wide_run("中文字a".as_bytes(), &mut points), 3);
/// assert_eq!(points[..3], [0x4e2d, 0x6587, 0x5b57]);
/// assert_eq!(xutf::wide_run("a中".as_bytes(), &mut points), 0);
/// ```
pub fn wide_run(bytes: &[u8], points: &mut [u16; WIDE_RUN]) -> usize {
	match bytes.first_chunk() {
		Some(bytes) => wide_run_48(bytes, points),
		None => wide_run_short(bytes, points),
	}
}

/// [`wide_run`] over fewer than 48 bytes, followed by zeros, which end the
/// run.
#[inline(never)]
fn wide_run_short(bytes: &[u8], points: &mut [u16; WIDE_RUN]) -> usize {
	let mut padded = [0; 3 * WIDE_RUN];
	padded[..bytes.len()].copy_from_slice(bytes);
	wide_run_48(&padded, points)
}

/// [`wide_run`] over exactly the 48 bytes 16 three-byte characters take.
#[inline(always)]
fn wide_run_48(bytes: &[u8; 3 * WIDE_RUN], points: &mut [u16; WIDE_RUN]) -> usize {
	// SAFETY: `bytes` spans 48 bytes.
	let (leads, middles, tails) = unsafe { load_utf8_3x16(bytes.as_ptr()) };
	let continuation = |v: Simd<u8, WIDE_RUN>| (v & Simd::splat(0xc0)).simd_eq(Simd::splat(0x80));
	// Up to the first lane that is not a three-byte lead and two
	// continuations, the lanes hold whole characters back to back.
	let shaped = (leads & Simd::splat(0xf0)).simd_eq(Simd::splat(0xe0))
		& continuation(middles)
		& continuation(tails);
	let decoded = decode_utf8_3x16(leads, middles, tails);
	*points = decoded.to_array();
	// `point - first <= last - first`, wrapping.
	let known = WIDE_RANGES
		.iter()
		.fold(Mask::splat(false), |known, &(first, last)| {
			known | (decoded - Simd::splat(first)).simd_le(Simd::splat(last - first))
		});
	let run = (!shaped).first_set().unwrap_or(WIDE_RUN);
	// The lanes outside the known ranges go to the property table; an
	// overlong form or a surrogate decodes to no wide character.
	let mut unknown = (!known).to_bitmask() & ((1 << run) - 1);
	while unknown != 0 {
		let lane = unknown.trailing_zeros() as usize;
		if char::from_u32(points[lane].into()).map_or(0, simple_width) != 2 {
			return lane;
		}
		unknown &= unknown - 1;
	}
	run
}

/// Whether a variation selector switches `c` between text and emoji
/// presentation: U+FE0F widens it from one cell to two, U+FE0E narrows it
/// from two to one.
///
/// The bases are the code points of the `Basic_Emoji`, keycap, flag, tag and
/// modifier sequences of `emoji-sequences.txt`; a terminal printing a
/// variation selector into a cell whose last character is one resizes that
/// cell as [`width`] measures it.
///
/// ```
/// assert!(xutf::is_emoji_presentation_base('\u{2764}')); // ❤, one cell
/// assert_eq!(xutf::width_str("\u{2764}\u{fe0f}"), 2);
/// assert!(!xutf::is_emoji_presentation_base('中'));
/// ```
#[inline]
pub fn is_emoji_presentation_base(c: char) -> bool {
	crate::props::is_emoji_presentation_base(c as u32)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn wide_ranges_hold_only_simple_wide_characters() {
		for (first, last) in WIDE_RANGES {
			for point in first..=last {
				let c = char::from_u32(point.into()).unwrap();
				assert_eq!(simple_width(c), 2, "U+{point:04X}");
			}
		}
	}
}
