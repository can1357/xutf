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

use crate::{
	encoding::Encoding,
	grapheme::{ClusterState, width_value},
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
