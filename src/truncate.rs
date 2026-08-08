//! Width-truncation returning a borrowed prefix with no delimiter or
//! allocation.
//!
//! The cut always lands on an extended grapheme cluster boundary. To add an
//! ellipsis, reserve its width before truncating and append it at the call
//! site.
//!
//! # Example
//! ```
//! let input = "abcdef";
//! let max_width = 4;
//! let mut rendered = String::from(xutf::truncate_str(input, max_width - 1));
//! rendered.push('…');
//! assert_eq!(rendered, "abc…");
//! ```

use crate::{encoding::Encoding, grapheme::next_cluster, simd::plain_prefix, utf8::Utf8};

/// Longest prefix of `input` no wider than `max_width` terminal cells, cut on
/// an extended grapheme cluster boundary.
///
/// After the width budget is filled, following zero-width clusters are retained
/// so combining marks, ZWJ tails, and zero-width controls are not orphaned.
#[inline]
pub fn truncate<E: Encoding>(input: &[E::Unit], max_width: usize) -> &[E::Unit] {
	let mut width = 0usize;
	let mut pos = 0usize;

	loop {
		if pos == input.len() {
			return input;
		}
		if !E::FOREIGN {
			// Scan at most budget + 1 plain units: enough to fill the budget
			// and detect that another plain cell follows, without sweeping a
			// megabyte run for an 80-cell cut.
			let remaining = &input[pos..];
			let budget = max_width - width;
			let window = remaining.len().min(budget.saturating_add(1));
			let run = plain_prefix(&remaining[..window]);
			if run > 0 {
				// Hold one unit back when data follows: the run's last char may
				// open a promotable or extending cluster, such as a keycap.
				let safe = if pos + run < input.len() {
					run - 1
				} else {
					run
				};
				let take = safe.min(budget);
				pos += take;
				width += take;
				if take < safe {
					return &input[..pos];
				}
				if pos == input.len() {
					return input;
				}
			}
		}

		let scan = next_cluster::<E>(&input[pos..]);
		if width + scan.width > max_width {
			return &input[..pos];
		}
		pos += scan.units;
		width += scan.width;
	}
}

/// [`truncate`] over UTF-8 `str`.
#[inline]
pub fn truncate_str(input: &str, max_width: usize) -> &str {
	// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
	unsafe { core::str::from_utf8_unchecked(truncate::<Utf8>(input.as_bytes(), max_width)) }
}
