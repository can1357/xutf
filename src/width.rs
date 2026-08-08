//! Visible terminal width without ANSI interpretation or allocation.
//!
//! Width is measured per extended grapheme cluster using UAX #11 table widths
//! and emoji-presentation rules. Combining marks, variation selectors, skin
//! tones, and content following a ZWJ contribute no extra cells; other joined
//! characters retain their table width. VS16 and keycap sequences promote a
//! promotable text base to two cells. A SIMD path counts printable ASCII in
//! bulk. Controls, including `\t`, have width zero; callers should expand tabs
//! before measuring when tab stops matter.
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
	let mut w = 0usize;
	let mut rest = input;

	'bulk: loop {
		if !E::FOREIGN {
			let run = plain_prefix(rest);
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
				w += take;
				rest = &rest[take..];
			}
		}
		if rest.is_empty() {
			return w;
		}

		let mut cp = E::decode(&mut rest);
		let mut p = props(cp);
		'cluster: loop {
			let mut state = 'simple: {
				// Simple runs: one decode, one table load, and one byte test per
				// codepoint. The held codepoint only counts once its successor
				// confirms the boundary.
				while simple(p) {
					if rest.is_empty() {
						return w + width_value(p);
					}
					let mut peek = rest;
					let next = E::decode(&mut peek);
					if !E::FOREIGN && next.wrapping_sub(0x20) <= 0x5e {
						w += width_value(p);
						continue 'bulk;
					}
					let np = props(next);
					if !simple(np) {
						let mut state = ClusterState::start(cp, p);
						rest = peek;
						if state.try_join(next, np) {
							break 'simple state;
						}
						w += state.finish();
						break 'simple ClusterState::start(next, np);
					}
					w += width_value(p);
					cp = next;
					p = np;
					rest = peek;
				}
				break 'simple ClusterState::start(cp, p);
			};

			// Full cluster machine: reuses the rejected boundary codepoint as
			// the next base instead of re-scanning it.
			loop {
				if rest.is_empty() {
					return w + state.finish();
				}
				let mut peek = rest;
				let next = E::decode(&mut peek);
				if !E::FOREIGN && next.wrapping_sub(0x20) <= 0x5e && !state.joins_plain() {
					// Boundary onto printable ASCII: resume the bulk path.
					w += state.finish();
					continue 'bulk;
				}
				let np = props(next);
				if state.try_join(next, np) {
					rest = peek;
					continue;
				}
				w += state.finish();
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
