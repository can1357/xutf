//! Extended grapheme cluster segmentation (UAX #29, Unicode 17) with code-unit
//! offsets and terminal cell width computed in the same pass. The permissive
//! iterators allocate no memory; [`Graphemes`] also iterates from both ends and
//! reports exact lengths.

use core::{iter::FusedIterator, marker::PhantomData};

use crate::{
	encoding::Encoding,
	props::{
		CB_CONTROL, CB_CR, CB_EXTEND, CB_EXTEND_INCB_LINKER, CB_L, CB_LF, CB_LV, CB_LVT, CB_MASK,
		CB_OTHER_INCB_CONSONANT, CB_PREPEND, CB_RI, CB_SPACING_MARK, CB_T, CB_V, CB_ZWJ, EPIC_BIT,
		INCB_EXTEND_BIT, WIDTH_SHIFT, is_emoji_presentation_base, props,
	},
	simd::plain_prefix,
	unit::Unit,
	utf8::Utf8,
};

/// A borrowed extended grapheme cluster and its terminal cell width.
pub struct Grapheme<'a, E: Encoding> {
	/// Code units belonging to this cluster.
	pub units: &'a [E::Unit],
	/// Terminal cells occupied by this cluster.
	pub width: usize,
}

impl<E: Encoding> Clone for Grapheme<'_, E> {
	#[inline(always)]
	fn clone(&self) -> Self {
		*self
	}
}

impl<E: Encoding> Copy for Grapheme<'_, E> {}

impl<E: Encoding> Grapheme<'_, E> {
	/// Returns whether the cluster's base codepoint is a C0, DEL, or C1
	/// control (`Cc`).
	///
	/// CRLF is one cluster and reports `true`. Controls always have width zero,
	/// but the converse does not hold: zero-width formats such as ZWSP and ZWJ,
	/// and combining marks, report `false`. Terminal renderers can use this
	/// distinction to decide whether to skip a cluster or draw it. An empty
	/// manually constructed cluster reports `false`.
	#[inline]
	pub fn is_control(&self) -> bool {
		let mut units = self.units;
		if units.is_empty() {
			return false;
		}
		let cp = E::decode(&mut units);
		matches!(cp, 0x00..=0x1f | 0x7f | 0x80..=0x9f)
	}
}

/// Allocation-free double-ended iterator over extended grapheme clusters.
///
/// The length is exact: [`size_hint`](Iterator::size_hint) and
/// [`len`](ExactSizeIterator::len) count the remaining clusters in one O(n)
/// scan instead of returning cheap bounds.
pub struct Graphemes<'a, E: Encoding> {
	rest:      &'a [E::Unit],
	_encoding: PhantomData<E>,
}

impl<E: Encoding> Clone for Graphemes<'_, E> {
	#[inline(always)]
	fn clone(&self) -> Self {
		Self { rest: self.rest, _encoding: PhantomData }
	}
}

impl<'a, E: Encoding> Iterator for Graphemes<'a, E> {
	type Item = Grapheme<'a, E>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if self.rest.is_empty() {
			return None;
		}
		let scan = next_cluster::<E>(self.rest);
		let (units, rest) = self.rest.split_at(scan.units);
		self.rest = rest;
		Some(Grapheme { units, width: scan.width })
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		let len = cluster_count::<E>(self.rest);
		(len, Some(len))
	}

	#[inline]
	fn count(self) -> usize {
		cluster_count::<E>(self.rest)
	}

	#[inline]
	fn last(mut self) -> Option<Grapheme<'a, E>> {
		self.next_back()
	}
}

impl<'a, E: Encoding> DoubleEndedIterator for Graphemes<'a, E> {
	#[inline]
	fn next_back(&mut self) -> Option<Grapheme<'a, E>> {
		if self.rest.is_empty() {
			return None;
		}
		let scan = prev_cluster::<E>(self.rest);
		let (rest, units) = self.rest.split_at(self.rest.len() - scan.units);
		self.rest = rest;
		Some(Grapheme { units, width: scan.width })
	}
}

impl<E: Encoding> ExactSizeIterator for Graphemes<'_, E> {
	#[inline]
	fn len(&self) -> usize {
		cluster_count::<E>(self.rest)
	}
}

impl<E: Encoding> FusedIterator for Graphemes<'_, E> {}

/// Allocation-free double-ended iterator over extended grapheme clusters and
/// their code-unit offsets.
///
/// Offsets are measured from the start of the original input: bytes for UTF-8,
/// `u16` units for UTF-16, and `u32` units for UTF-32. Empty input yields no
/// items. Like [`Graphemes`], the length is exact at the cost of a counting
/// scan.
pub struct GraphemeIndices<'a, E: Encoding> {
	inner:  Graphemes<'a, E>,
	offset: usize,
}

impl<E: Encoding> Clone for GraphemeIndices<'_, E> {
	#[inline(always)]
	fn clone(&self) -> Self {
		Self { inner: self.inner.clone(), offset: self.offset }
	}
}

impl<'a, E: Encoding> Iterator for GraphemeIndices<'a, E> {
	type Item = (usize, Grapheme<'a, E>);

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let grapheme = self.inner.next()?;
		let offset = self.offset;
		self.offset += grapheme.units.len();
		Some((offset, grapheme))
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	#[inline]
	fn count(self) -> usize {
		self.inner.count()
	}

	#[inline]
	fn last(mut self) -> Option<Self::Item> {
		self.next_back()
	}
}

impl<E: Encoding> DoubleEndedIterator for GraphemeIndices<'_, E> {
	#[inline]
	fn next_back(&mut self) -> Option<Self::Item> {
		let grapheme = self.inner.next_back()?;
		Some((self.offset + self.inner.rest.len(), grapheme))
	}
}

impl<E: Encoding> ExactSizeIterator for GraphemeIndices<'_, E> {
	#[inline]
	fn len(&self) -> usize {
		self.inner.len()
	}
}

impl<E: Encoding> FusedIterator for GraphemeIndices<'_, E> {}

/// Iterates the extended grapheme clusters of an encoded slice with their
/// code-unit offsets.
///
/// Offsets are bytes for UTF-8 and element counts for UTF-16 and UTF-32. Empty
/// input yields no items.
#[inline(always)]
pub const fn grapheme_indices<E: Encoding>(input: &[E::Unit]) -> GraphemeIndices<'_, E> {
	GraphemeIndices { inner: graphemes(input), offset: 0 }
}

/// Iterates the extended grapheme clusters of an encoded slice without
/// allocating.
#[inline(always)]
pub const fn graphemes<E: Encoding>(input: &[E::Unit]) -> Graphemes<'_, E> {
	Graphemes { rest: input, _encoding: PhantomData }
}

/// Double-ended, exact-length iterator over the extended grapheme clusters of
/// a UTF-8 string, yielding borrowed sub-strings. Created by
/// [`graphemes_str`].
#[derive(Clone)]
pub struct StrGraphemes<'a> {
	inner: Graphemes<'a, Utf8>,
}

impl<'a> Iterator for StrGraphemes<'a> {
	type Item = &'a str;

	#[inline]
	fn next(&mut self) -> Option<&'a str> {
		// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
		self
			.inner
			.next()
			.map(|g| unsafe { core::str::from_utf8_unchecked(g.units) })
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	#[inline]
	fn count(self) -> usize {
		self.inner.count()
	}

	#[inline]
	fn last(mut self) -> Option<&'a str> {
		self.next_back()
	}
}

impl<'a> DoubleEndedIterator for StrGraphemes<'a> {
	#[inline]
	fn next_back(&mut self) -> Option<&'a str> {
		// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
		self
			.inner
			.next_back()
			.map(|g| unsafe { core::str::from_utf8_unchecked(g.units) })
	}
}

impl ExactSizeIterator for StrGraphemes<'_> {
	#[inline]
	fn len(&self) -> usize {
		self.inner.len()
	}
}

impl FusedIterator for StrGraphemes<'_> {}

/// Double-ended, exact-length iterator over a UTF-8 string's grapheme
/// clusters and byte offsets.
#[derive(Clone)]
pub struct StrGraphemeIndices<'a> {
	inner: GraphemeIndices<'a, Utf8>,
}

#[inline(always)]
const fn indexed_str(item: (usize, Grapheme<'_, Utf8>)) -> (usize, &str) {
	let (offset, grapheme) = item;
	// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
	(offset, unsafe { core::str::from_utf8_unchecked(grapheme.units) })
}

impl<'a> Iterator for StrGraphemeIndices<'a> {
	type Item = (usize, &'a str);

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.inner.next().map(indexed_str)
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.inner.size_hint()
	}

	#[inline]
	fn count(self) -> usize {
		self.inner.count()
	}

	#[inline]
	fn last(mut self) -> Option<Self::Item> {
		self.next_back()
	}
}

impl DoubleEndedIterator for StrGraphemeIndices<'_> {
	#[inline]
	fn next_back(&mut self) -> Option<Self::Item> {
		self.inner.next_back().map(indexed_str)
	}
}

impl ExactSizeIterator for StrGraphemeIndices<'_> {
	#[inline]
	fn len(&self) -> usize {
		self.inner.len()
	}
}

impl FusedIterator for StrGraphemeIndices<'_> {}

/// Iterates the extended grapheme clusters of a UTF-8 string as borrowed
/// strings.
#[inline]
pub const fn graphemes_str(input: &str) -> StrGraphemes<'_> {
	StrGraphemes { inner: graphemes::<Utf8>(input.as_bytes()) }
}

/// Iterates a UTF-8 string's extended grapheme clusters with their byte
/// offsets.
///
/// Each offset is a valid character boundary measured from the start of the
/// input. Empty input yields no items.
#[inline]
pub const fn grapheme_indices_str(input: &str) -> StrGraphemeIndices<'_> {
	StrGraphemeIndices { inner: grapheme_indices::<Utf8>(input.as_bytes()) }
}

/// One scanned cluster: code units consumed and terminal cell width.
pub struct ClusterScan {
	pub units: usize,
	pub width: usize,
}

/// Standalone cell width encoded in a props byte.
#[inline(always)]
pub const fn width_value(p: u8) -> usize {
	((p >> WIDTH_SHIFT) & 3) as usize
}

/// Incremental join state for one cluster, driving [`next_cluster`] and the
/// flat scan inside [`crate::width`]: one decode and one table load per
/// codepoint, no re-scanning at boundaries.
///
/// The width follows kitty: a cluster takes the cells of its first codepoint;
/// a joining codepoint adds none, except that U+FE0F widens a one-cell emoji
/// presentation base, U+FE0E narrows a two-cell one, and a spacing mark with
/// a width of its own (Thai and Lao AM) widens a one-cell base.
#[derive(Clone, Copy, Debug)]
pub struct ClusterState {
	width:      usize,
	/// Cells the last codepoint left the cluster at, kitty's `prev_width`:
	/// the base's width, as changed by the variation selectors and spacing
	/// marks after it; zero after a variation selector that changed nothing.
	last_width: usize,
	prev:       u8,
	prev_cp:    u32,
	epic:       u8,
	incb:       u8,
	ri_odd:     bool,
}

impl ClusterState {
	/// Starts a cluster whose base is `cp0` with packed props `p0`.
	#[inline(always)]
	pub fn start(cp0: u32, p0: u8) -> Self {
		let c0 = p0 & CB_MASK;
		Self {
			width:      width_value(p0),
			last_width: width_value(p0),
			prev:       c0,
			prev_cp:    cp0,
			epic:       u8::from(p0 & EPIC_BIT != 0),
			incb:       u8::from(c0 == CB_OTHER_INCB_CONSONANT),
			ri_odd:     c0 == CB_RI,
		}
	}

	/// `true` when the cluster would absorb a following printable ASCII unit;
	/// only a Prepend base does (`GB9b`).
	#[inline(always)]
	pub const fn joins_plain(&self) -> bool {
		self.prev == CB_PREPEND
	}

	/// Tries to join `cp` (packed props `p`) onto the cluster, updating break
	/// and width state when it joins.
	#[inline(always)]
	pub fn try_join(&mut self, cp: u32, p: u8) -> bool {
		let c = p & CB_MASK;

		if matches!(self.prev, CB_CR | CB_LF | CB_CONTROL) {
			// GB3/GB4: controls break from everything except CR before LF.
			if self.prev != CB_CR || c != CB_LF {
				return false;
			}
		} else {
			let join = match c {
				CB_CR | CB_LF | CB_CONTROL => false,
				CB_EXTEND | CB_EXTEND_INCB_LINKER | CB_ZWJ => true,
				CB_SPACING_MARK => true,
				_ if self.prev == CB_PREPEND => true,
				CB_L => self.prev == CB_L,
				CB_V => matches!(self.prev, CB_L | CB_LV | CB_V),
				CB_T => matches!(self.prev, CB_LV | CB_V | CB_LVT | CB_T),
				CB_LV | CB_LVT => self.prev == CB_L,
				CB_RI => self.prev == CB_RI && self.ri_odd,
				CB_OTHER_INCB_CONSONANT => self.incb == 2,
				_ => self.prev == CB_ZWJ && self.epic == 2 && p & EPIC_BIT != 0,
			};
			if !join {
				return false;
			}
		}

		if c == CB_EXTEND || c == CB_EXTEND_INCB_LINKER {
			if self.epic != 1 {
				self.epic = 0;
			}
		} else if c == CB_ZWJ {
			self.epic = if self.epic == 1 { 2 } else { 0 };
		} else if p & EPIC_BIT != 0 {
			self.epic = 1;
		} else {
			self.epic = 0;
		}

		if c == CB_OTHER_INCB_CONSONANT {
			self.incb = 1;
		} else if c == CB_EXTEND_INCB_LINKER {
			self.incb = if self.incb != 0 { 2 } else { 0 };
		} else if p & INCB_EXTEND_BIT == 0 {
			self.incb = 0;
		}

		self.ri_odd = c == CB_RI && !self.ri_odd;

		// `last_width` is either `width` or zero, so narrowing never
		// underflows and widening never takes a cluster past two cells.
		match cp {
			0xfe0f => {
				if self.last_width == 1 && is_emoji_presentation_base(self.prev_cp) {
					self.width += 1;
					self.last_width = 2;
				} else {
					self.last_width = 0;
				}
			},
			0xfe0e => {
				if self.last_width == 2 && is_emoji_presentation_base(self.prev_cp) {
					self.width -= 1;
					self.last_width = 1;
				} else {
					self.last_width = 0;
				}
			},
			_ => {
				if c == CB_SPACING_MARK && width_value(p) != 0 && self.last_width == 1 {
					self.width += 1;
					self.last_width = 2;
				}
			},
		}
		self.prev_cp = cp;
		self.prev = c;
		true
	}

	/// Cells the cluster takes.
	#[inline(always)]
	pub const fn finish(&self) -> usize {
		self.width
	}
}

/// A grapheme cluster built one codepoint at a time, for text that arrives
/// incrementally (a terminal's output stream) and cannot be re-scanned.
///
/// [`Cluster::push`] answers whether a codepoint joins the cluster, with the
/// same UAX #29 decisions [`graphemes`] makes walking the text forward: one
/// table load per codepoint, no buffering.
///
/// ```
/// let mut cluster = xutf::Cluster::new('e');
/// assert!(cluster.push('\u{301}'));
/// assert!(!cluster.push('x'));
/// assert_eq!(cluster.width(), 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Cluster(ClusterState);

impl Cluster {
	/// A cluster starting with `c`.
	#[inline]
	#[must_use]
	pub fn new(c: char) -> Self {
		Self(ClusterState::start(c as u32, props(c as u32)))
	}

	/// Adds `c` when it continues the cluster and reports whether it did; a
	/// `false` leaves the cluster unchanged, `c` starting the next one.
	#[inline]
	pub fn push(&mut self, c: char) -> bool {
		self.0.try_join(c as u32, props(c as u32))
	}

	/// Terminal-cell width of the cluster so far, as [`graphemes`] reports
	/// it for the same codepoints.
	#[inline]
	#[must_use]
	pub const fn width(&self) -> usize {
		self.0.finish()
	}
}

/// Scans the cluster at the head of a non-empty encoded slice in one pass.
#[inline]
pub fn next_cluster<E: Encoding>(input: &[E::Unit]) -> ClusterScan {
	if !E::FOREIGN {
		let u0 = input[0].to_u32();
		if u0 < 0x80 {
			if u0 == 0x0d {
				if input.len() > 1 && input[1].to_u32() == 0x0a {
					return ClusterScan { units: 2, width: 0 };
				}
				return ClusterScan { units: 1, width: 0 };
			}
			if input.len() == 1 || input[1].to_u32() < 0x80 {
				let width = usize::from((0x20..=0x7e).contains(&u0));
				return ClusterScan { units: 1, width };
			}
		}
	}

	let mut rest = input;
	let cp0 = E::decode(&mut rest);
	let mut state = ClusterState::start(cp0, props(cp0));

	while !rest.is_empty() {
		let mut peek = rest;
		let cp = E::decode(&mut peek);
		if !state.try_join(cp, props(cp)) {
			break;
		}
		rest = peek;
	}

	ClusterScan { units: input.len() - rest.len(), width: state.finish() }
}

/// Conservative pairwise join test for the backward scan: `true` when a
/// codepoint with packed props `b` could join a cluster ending in props `a`
/// under *some* preceding context. `false` therefore proves a break
/// regardless of history; the history-dependent rules (regional-indicator
/// parity, emoji ZWJ chains, `InCB` linkers) conservatively stay `true`.
#[inline(always)]
const fn may_join(a: u8, b: u8) -> bool {
	let ca = a & CB_MASK;
	let cb = b & CB_MASK;
	if matches!(ca, CB_CR | CB_LF | CB_CONTROL) {
		// GB3/GB4: controls break from everything except CR before LF.
		return ca == CB_CR && cb == CB_LF;
	}
	match cb {
		CB_CR | CB_LF | CB_CONTROL => false,
		CB_EXTEND | CB_EXTEND_INCB_LINKER | CB_ZWJ | CB_SPACING_MARK => true,
		_ if ca == CB_PREPEND => true,
		CB_L => ca == CB_L,
		CB_V => matches!(ca, CB_L | CB_LV | CB_V),
		CB_T => matches!(ca, CB_LV | CB_V | CB_LVT | CB_T),
		CB_LV | CB_LVT => ca == CB_L,
		CB_RI => ca == CB_RI,
		// GB9c: a consonant joins only when a linker chain is still open,
		// which requires the previous codepoint to keep it open.
		CB_OTHER_INCB_CONSONANT => {
			ca == CB_EXTEND_INCB_LINKER || (a & INCB_EXTEND_BIT != 0 && ca != CB_OTHER_INCB_CONSONANT)
		},
		_ => ca == CB_ZWJ && b & EPIC_BIT != 0,
	}
}

/// Scans the cluster at the end of a non-empty encoded slice.
///
/// Decodes backwards to the nearest boundary provable from a codepoint pair
/// alone ([`may_join`]), then re-runs the forward scanner from there, so the
/// forward state machine stays the single segmentation authority. Malformed
/// input may cut differently than the forward direction, but progress and
/// in-bounds cuts still hold.
#[inline]
pub fn prev_cluster<E: Encoding>(input: &[E::Unit]) -> ClusterScan {
	if !E::FOREIGN {
		let last = input[input.len() - 1].to_u32();
		if last < 0x80 {
			let prev = if input.len() > 1 {
				input[input.len() - 2].to_u32()
			} else {
				0x80
			};
			if last == 0x0a && prev == 0x0d {
				return ClusterScan { units: 2, width: 0 };
			}
			// Two adjacent ASCII units always break (GB9b Prepend and CR are
			// the only absorbers before ASCII, and both are handled above).
			if input.len() == 1 || prev < 0x80 {
				let width = usize::from((0x20..=0x7e).contains(&last));
				return ClusterScan { units: 1, width };
			}
		}
	}

	let mut back = input;
	let mut after = props(E::decode_back(&mut back));
	while !back.is_empty() {
		let mut peek = back;
		let p = props(E::decode_back(&mut peek));
		if !may_join(p, after) {
			break;
		}
		back = peek;
		after = p;
	}

	// Forward re-scan from the guaranteed boundary at `back.len()`.
	let mut at = back.len();
	loop {
		let scan = next_cluster::<E>(&input[at..]);
		if at + scan.units == input.len() {
			return scan;
		}
		at += scan.units;
	}
}

/// Counts the extended grapheme clusters of an encoded slice in one pass,
/// with a SIMD bulk path over printable ASCII runs.
pub fn cluster_count<E: Encoding>(input: &[E::Unit]) -> usize {
	let mut rest = input;
	let mut count = 0;
	while !rest.is_empty() {
		if !E::FOREIGN {
			let run = plain_prefix(rest);
			if run == rest.len() {
				return count + run;
			}
			// All but the run's last unit are whole clusters; the last may
			// open a cluster that the next codepoint joins, such as a keycap.
			if run > 1 {
				count += run - 1;
				rest = &rest[run - 1..];
			}
		}
		let scan = next_cluster::<E>(rest);
		count += 1;
		rest = &rest[scan.units..];
	}
	count
}
