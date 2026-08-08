//! Visible terminal width without ANSI interpretation or allocation.
//!
//! Width is measured per extended grapheme cluster using UAX #11 table widths
//! and emoji-presentation rules. Combining marks, variation selectors, skin
//! tones, and content following a ZWJ contribute no extra cells; other joined
//! characters retain their table width. VS16 and keycap sequences promote a
//! promotable text base to two cells. A SIMD path counts printable ASCII in
//! bulk. Controls, including `\t`, have width zero; callers should expand tabs
//! before measuring when tab stops matter.
//! Bounded measurement can stop as soon as a limit is exceeded, while
//! per-character measurement reports the standalone width of one scalar.
//!
//! Compared with `unicode-width` string width, deliberate differences include
//! CR/LF, Arabic Lam-Alef and Hebrew Lamed ligatures, Khmer coeng sequences,
//! Buginese ya-ZWJ, Tifinagh consonant joins, Old Turkic Orkhon I, Kirat Rai
//! vowel ligatures, VS15 and VS1-3 presentation, bare keycaps, and U+17D8.

use crate::{
	encoding::Encoding,
	grapheme::{ClusterState, width_value},
	props::{CB_LV, CB_LVT, CB_MASK, CB_OTHER, EPIC_BIT, WIDTH_EMOJI_TEXT, WIDTH_SHIFT, props},
	simd::plain_prefix,
	unit::Unit,
	utf8::Utf8,
};

/// `true` for codepoints that provably form single-codepoint clusters next to
/// each other: break class Other/LV/LVT, not `Extended_Pictographic`, and no
/// pending VS16/keycap promotion. Two adjacent such codepoints always have a
/// boundary between them, so runs (CJK prose, precomposed Hangul) are summed
/// without join-state churn.
#[inline(always)]
const fn simple(p: u8) -> bool {
	matches!(p & CB_MASK, CB_OTHER | CB_LV | CB_LVT)
		&& p & EPIC_BIT == 0
		&& (p >> WIDTH_SHIFT) & 3 != WIDTH_EMOJI_TEXT
}

#[inline(always)]
const fn promotable_ascii(cp: u32) -> bool {
	matches!(cp, 0x23 | 0x2a | 0x30..=0x39)
}

/// Visible width of `input` in terminal cells (extended grapheme clusters,
/// UAX #11 plus emoji presentation rules, with an ASCII bulk path).
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
				// Only `#`, `*`, and digits need the last unit held for
				// possible VS16/keycap promotion. Other printable ASCII is
				// additive even when an extending codepoint follows.
				let promote = promotable_ascii(rest[run - 1].to_u32());
				let take = if run < rest.len() && promote {
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

/// Standalone terminal-cell width of `c`.
///
/// Controls, combining marks, and joiners have width zero. Trailing-context
/// promotion by variation selectors, keycaps, or skin tones requires the full
/// string APIs.
#[inline]
pub fn width_char(c: char) -> usize {
	width_value(props(c as u32))
}
