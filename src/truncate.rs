//! Width-based truncation and front-skipping on extended grapheme cluster
//! boundaries, returning borrowed slices with no delimiter or allocation.
//!
//! Measured variants also return the exact cell width, while front-skipping
//! drops enough whole clusters to satisfy a terminal scroll offset.
//!
//! To add an ellipsis, reserve its width before truncating and append it at the
//! call site.
//!
//! # Example
//! ```
//! use xutf::Text;
//!
//! let input = "abcdef";
//! let max_width = 4;
//! let mut rendered = String::from(input.truncate_width(max_width - 1));
//! rendered.push('…');
//! assert_eq!(rendered, "abc…");
//! ```

use crate::{encoding::Encoding, grapheme::next_cluster, simd::plain_prefix, utf8::Utf8};

/// Longest prefix of `input` no wider than `max_width` terminal cells, cut on
/// an extended grapheme cluster boundary, and its exact terminal-cell width.
///
/// After the width budget is filled, following zero-width clusters are retained
/// so combining marks, ZWJ tails, and zero-width controls are not orphaned.
#[inline]
pub fn truncate_measured<E: Encoding>(input: &[E::Unit], max_width: usize) -> (&[E::Unit], usize) {
	let mut width = 0usize;
	let mut pos = 0usize;

	loop {
		if pos == input.len() {
			return (input, width);
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
				if run == remaining.len() {
					// A wholly plain tail needs no cluster scan. The bounded
					// window can exceed the budget by at most one unit.
					let take = run.min(budget);
					pos += take;
					width += take;
					return if take == run {
						(input, width)
					} else {
						(&input[..pos], width)
					};
				}
				// Data follows, so hold one unit back: the run's last char may
				// open a cluster the next codepoint joins and widens (U+FE0F
				// after a digit).
				// Since the scan is bounded to budget + 1, this is affordable
				// without a second min or cutoff branch.
				let safe = run - 1;
				pos += safe;
				width += safe;
			}
		}

		let scan = next_cluster::<E>(&input[pos..]);
		if width + scan.width > max_width {
			return (&input[..pos], width);
		}
		pos += scan.units;
		width += scan.width;
	}
}

/// Longest prefix of `input` no wider than `max_width` terminal cells, cut on
/// an extended grapheme cluster boundary.
///
/// After the width budget is filled, following zero-width clusters are retained
/// so combining marks, ZWJ tails, and zero-width controls are not orphaned.
#[inline]
pub fn truncate<E: Encoding>(input: &[E::Unit], max_width: usize) -> &[E::Unit] {
	truncate_measured::<E>(input, max_width).0
}

/// [`truncate_measured`] over UTF-8 `str`.
#[inline]
pub fn truncate_measured_str(input: &str, max_width: usize) -> (&str, usize) {
	let (prefix, width) = truncate_measured::<Utf8>(input.as_bytes(), max_width);
	// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
	(unsafe { core::str::from_utf8_unchecked(prefix) }, width)
}

/// [`truncate`] over UTF-8 `str`.
#[inline]
pub fn truncate_str(input: &str, max_width: usize) -> &str {
	// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
	unsafe { core::str::from_utf8_unchecked(truncate::<Utf8>(input.as_bytes(), max_width)) }
}

/// Tail of `input` after dropping enough whole clusters to cover `columns`
/// terminal cells, together with the exact dropped width.
///
/// `columns == 0` leaves the input unchanged. Once the target is reached,
/// immediately following zero-width clusters are also dropped. A cluster that
/// straddles the target is dropped whole, so the returned width may exceed
/// `columns`; unlike [`truncate`], which never exceeds its budget, this
/// guarantees the remaining width is at most the input width minus `columns`.
/// Input narrower than `columns` produces an empty tail and its total width.
#[inline]
pub fn skip_columns<E: Encoding>(input: &[E::Unit], columns: usize) -> (&[E::Unit], usize) {
	if columns == 0 {
		return (input, 0);
	}

	let mut width = 0usize;
	let mut pos = 0usize;

	loop {
		if pos == input.len() {
			return (&input[pos..], width);
		}
		if width >= columns {
			let scan = next_cluster::<E>(&input[pos..]);
			if scan.width != 0 {
				return (&input[pos..], width);
			}
			pos += scan.units;
			continue;
		}
		if !E::FOREIGN {
			// Scan at most budget + 1 plain units: enough to fill the budget
			// and detect that another plain cell follows, without sweeping a
			// megabyte run for an 80-cell cut.
			let remaining = &input[pos..];
			let budget = columns - width;
			let window = remaining.len().min(budget.saturating_add(1));
			let run = plain_prefix(&remaining[..window]);
			if run > 0 {
				// Hold one unit back when data follows: the run's last char may
				// open a cluster the next codepoint joins and widens (U+FE0F
				// after a digit).
				let safe = if pos + run < input.len() {
					run - 1
				} else {
					run
				};
				let take = safe.min(budget);
				pos += take;
				width += take;
				if take < safe {
					return (&input[pos..], width);
				}
				if pos == input.len() {
					return (&input[pos..], width);
				}
			}
		}

		let scan = next_cluster::<E>(&input[pos..]);
		pos += scan.units;
		width += scan.width;
	}
}

/// [`skip_columns`] over UTF-8 `str`.
#[inline]
pub fn skip_columns_str(input: &str, columns: usize) -> (&str, usize) {
	let (tail, width) = skip_columns::<Utf8>(input.as_bytes(), columns);
	// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
	(unsafe { core::str::from_utf8_unchecked(tail) }, width)
}
