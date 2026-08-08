//! Extended grapheme cluster segmentation (UAX #29, Unicode 17) with terminal
//! cell width computed in the same pass. The permissive iterators allocate no
//! memory.

use core::marker::PhantomData;

use crate::{
	encoding::Encoding,
	props::{
		CB_CONTROL, CB_CR, CB_EXTEND, CB_EXTEND_INCB_LINKER, CB_L, CB_LF, CB_LV, CB_LVT, CB_MASK,
		CB_OTHER_INCB_CONSONANT, CB_PREPEND, CB_RI, CB_SPACING_MARK, CB_T, CB_V, CB_ZWJ, EPIC_BIT,
		INCB_EXTEND_BIT, WIDTH_EMOJI_TEXT, WIDTH_SHIFT, is_emoji_modifier_base, props,
	},
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

/// Allocation-free iterator over extended grapheme clusters.
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

	#[inline(always)]
	fn size_hint(&self) -> (usize, Option<usize>) {
		(usize::from(!self.rest.is_empty()), Some(self.rest.len()))
	}
}

/// Iterates the extended grapheme clusters of an encoded slice without
/// allocating.
#[inline(always)]
pub const fn graphemes<E: Encoding>(input: &[E::Unit]) -> Graphemes<'_, E> {
	Graphemes { rest: input, _encoding: PhantomData }
}

/// Iterates the extended grapheme clusters of a UTF-8 string as borrowed
/// strings.
#[inline]
pub fn graphemes_str(input: &str) -> impl Iterator<Item = &str> + Clone {
	graphemes::<Utf8>(input.as_bytes()).map(|g| {
		// SAFETY: cluster boundaries fall on char boundaries in valid UTF-8.
		unsafe { core::str::from_utf8_unchecked(g.units) }
	})
}

/// One scanned cluster: code units consumed and terminal cell width.
pub struct ClusterScan {
	pub units: usize,
	pub width: usize,
}

/// Standalone cell width encoded in a props byte (class 3 reads as 1).
#[inline(always)]
pub const fn width_value(p: u8) -> usize {
	let width = (p >> WIDTH_SHIFT) & 3;
	if width == WIDTH_EMOJI_TEXT {
		1
	} else {
		width as usize
	}
}

/// Incremental join state for one cluster, driving [`next_cluster`] and the
/// flat scan inside [`crate::width`]: one decode and one table load per
/// codepoint, no re-scanning at boundaries.
pub struct ClusterState {
	width:      usize,
	prev:       u8,
	prev_cp:    u32,
	epic:       u8,
	incb:       u8,
	ri_odd:     bool,
	after_zwj:  bool,
	promotable: bool,
	promote:    bool,
}

impl ClusterState {
	/// Starts a cluster whose base is `cp0` with packed props `p0`.
	#[inline(always)]
	pub fn start(cp0: u32, p0: u8) -> Self {
		let c0 = p0 & CB_MASK;
		Self {
			width:      width_value(p0),
			prev:       c0,
			prev_cp:    cp0,
			epic:       u8::from(p0 & EPIC_BIT != 0),
			incb:       u8::from(c0 == CB_OTHER_INCB_CONSONANT),
			ri_odd:     c0 == CB_RI,
			after_zwj:  false,
			promotable: (p0 >> WIDTH_SHIFT) & 3 == WIDTH_EMOJI_TEXT,
			promote:    false,
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

		if cp == 0xfe0f || cp == 0x20e3 {
			self.promote = true;
		}
		if c == CB_ZWJ {
			self.after_zwj = true;
		} else if !self.after_zwj {
			// Extend-class marks are zero-width, except a few that carry
			// intrinsic width (emoji skin-tone modifiers, Kirat Rai vowels).
			// A modifier renders into an immediately preceding
			// Emoji_Modifier_Base ("\u{1F44D}\u{1F3FD}" is 2 cells) and
			// stands alone otherwise ("0\u{1F3FD}" is 3).
			let modifier = (c == CB_EXTEND || c == CB_EXTEND_INCB_LINKER)
				&& width_value(p) == 2
				&& is_emoji_modifier_base(self.prev_cp);
			if !modifier {
				self.width += width_value(p);
			}
		}
		self.prev_cp = cp;
		self.prev = c;
		true
	}

	/// Cluster width with any pending VS16/keycap promotion applied.
	#[inline(always)]
	pub fn finish(&self) -> usize {
		if self.promote && self.promotable {
			self.width.max(2)
		} else {
			self.width
		}
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
