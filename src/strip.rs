//! Allocation-free ANSI/VT escape stripping and visible-width measurement for
//! UTF-8, UTF-16, and UTF-32.
//!
//! Escape discovery scans 256-byte batches with portable SIMD. Once the first
//! sequence is removed, retained spans are compacted forward in register-sized
//! chunks; inputs without escapes are never written. The scalar state machine
//! follows ECMA-48 control strings plus the `XTerm` escape forms used by Bun.

use alloc::{string::String, vec::Vec};
use core::{
	mem::size_of,
	simd::{Simd, SimdElement, cmp::SimdPartialEq, num::SimdUint},
};

use crate::{
	encoding::{Encoding, Kind},
	unit::Unit,
	utf8::Utf8,
	utf16::Utf16,
	utf32::Utf32,
};

const ESC: u32 = 0x1b;
const BEL: u32 = 0x07;
const DCS: u32 = 0x90;
const SOS: u32 = 0x98;
const CSI: u32 = 0x9b;
const ST: u32 = 0x9c;
const OSC: u32 = 0x9d;
const PM: u32 = 0x9e;
const APC: u32 = 0x9f;

#[inline(always)]
const fn is_introducer(c: u32) -> bool {
	matches!(c, ESC | DCS | SOS | CSI | OSC | PM | APC)
}

#[inline(always)]
fn find_vectorized<T: SimdElement + Copy, const N: usize>(
	input: &[T],
	start: usize,
	locate: impl Fn(Simd<T, N>) -> Option<usize>,
	is_candidate: impl Fn(T) -> bool,
) -> usize {
	let mut at = start;
	while input.len() - at >= N * 4 {
		if let Some(found) = locate(Simd::from_slice(&input[at..])) {
			return at + found;
		}
		if let Some(found) = locate(Simd::from_slice(&input[at + N..])) {
			return at + N + found;
		}
		if let Some(found) = locate(Simd::from_slice(&input[at + N * 2..])) {
			return at + N * 2 + found;
		}
		if let Some(found) = locate(Simd::from_slice(&input[at + N * 3..])) {
			return at + N * 3 + found;
		}
		at += N * 4;
	}
	while input.len() - at >= N {
		if let Some(found) = locate(Simd::from_slice(&input[at..])) {
			return at + found;
		}
		at += N;
	}
	while at < input.len() {
		if is_candidate(input[at]) {
			return at;
		}
		at += 1;
	}
	input.len()
}

#[inline(always)]
fn escape_mask_u8<const N: usize>(chunk: Simd<u8, N>) -> Option<usize> {
	// C1 controls U+0080..=U+009F start with C2 in UTF-8. The scalar
	// state machine refines C2 false positives after the vector sweep.
	let matches = chunk.simd_eq(Simd::splat(ESC as u8)) | chunk.simd_eq(Simd::splat(0xc2));
	if matches.any() {
		matches.first_set()
	} else {
		None
	}
}

macro_rules! escape_mask {
	($name:ident, $unit:ty) => {
		#[inline(always)]
		fn $name<const FOREIGN: bool, const N: usize>(mut chunk: Simd<$unit, N>) -> Option<usize> {
			if FOREIGN {
				chunk = chunk.swap_bytes();
			}
			// One cheap broad comparison rejects almost every block. It matches
			// 0x10..=0x1F and 0x90..=0x9F; exact comparisons only run on hits.
			let broad = (chunk & Simd::splat(!0x8f)).simd_eq(Simd::splat(0x10));
			if !broad.any() {
				return None;
			}
			let exact = chunk.simd_eq(Simd::splat(ESC as $unit))
				| chunk.simd_eq(Simd::splat(DCS as $unit))
				| chunk.simd_eq(Simd::splat(SOS as $unit))
				| chunk.simd_eq(Simd::splat(CSI as $unit))
				| chunk.simd_eq(Simd::splat(OSC as $unit))
				| chunk.simd_eq(Simd::splat(PM as $unit))
				| chunk.simd_eq(Simd::splat(APC as $unit));
			exact.first_set()
		}
	};
}

escape_mask!(escape_mask_u16, u16);
escape_mask!(escape_mask_u32, u32);

#[inline(always)]
fn find_u8(input: &[u8], start: usize) -> usize {
	if input.len() - start >= 16 {
		if let Some(found) = escape_mask_u8(Simd::<u8, 16>::from_slice(&input[start..])) {
			return start + found;
		}
		return find_vectorized::<u8, 64>(input, start + 16, escape_mask_u8, |c| {
			c == ESC as u8 || c == 0xc2
		});
	}
	find_vectorized::<u8, 64>(input, start, escape_mask_u8, |c| c == ESC as u8 || c == 0xc2)
}

#[inline(always)]
fn find_u16<const FOREIGN: bool>(input: &[u16], start: usize) -> usize {
	if input.len() - start >= 8 {
		if let Some(found) =
			escape_mask_u16::<FOREIGN, 8>(Simd::<u16, 8>::from_slice(&input[start..]))
		{
			return start + found;
		}
		return find_vectorized::<u16, 32>(
			input,
			start + 8,
			escape_mask_u16::<FOREIGN, 32>,
			|mut c| {
				if FOREIGN {
					c = c.swap_bytes();
				}
				is_introducer(u32::from(c))
			},
		);
	}
	find_vectorized::<u16, 32>(input, start, escape_mask_u16::<FOREIGN, 32>, |mut c| {
		if FOREIGN {
			c = c.swap_bytes();
		}
		is_introducer(u32::from(c))
	})
}

#[inline(always)]
fn find_u32<const FOREIGN: bool>(input: &[u32], start: usize) -> usize {
	if input.len() - start >= 4 {
		if let Some(found) =
			escape_mask_u32::<FOREIGN, 4>(Simd::<u32, 4>::from_slice(&input[start..]))
		{
			return start + found;
		}
		return find_vectorized::<u32, 16>(
			input,
			start + 4,
			escape_mask_u32::<FOREIGN, 16>,
			|mut c| {
				if FOREIGN {
					c = c.swap_bytes();
				}
				is_introducer(c)
			},
		);
	}
	find_vectorized::<u32, 16>(input, start, escape_mask_u32::<FOREIGN, 16>, |mut c| {
		if FOREIGN {
			c = c.swap_bytes();
		}
		is_introducer(c)
	})
}

#[inline(always)]
unsafe fn as_units<T, U>(input: &[T]) -> &[U] {
	debug_assert_eq!(size_of::<T>(), size_of::<U>());
	// SAFETY: callers dispatch from the sealed Encoding::KIND invariant, which
	// maps each encoding family to exactly one same-sized unsigned unit type.
	unsafe { core::slice::from_raw_parts(input.as_ptr().cast(), input.len()) }
}

#[inline(always)]
fn find_escape<E: Encoding>(input: &[E::Unit], start: usize) -> usize {
	match E::KIND {
		Kind::Utf8 => {
			// SAFETY: Utf8 is the only sealed encoding with Kind::Utf8.
			find_u8(unsafe { as_units(input) }, start)
		},
		Kind::Utf16 => {
			// SAFETY: Utf16 is the only sealed encoding with Kind::Utf16.
			let input = unsafe { as_units(input) };
			if E::FOREIGN {
				find_u16::<true>(input, start)
			} else {
				find_u16::<false>(input, start)
			}
		},
		Kind::Utf32 => {
			// SAFETY: Utf32 is the only sealed encoding with Kind::Utf32.
			let input = unsafe { as_units(input) };
			if E::FOREIGN {
				find_u32::<true>(input, start)
			} else {
				find_u32::<false>(input, start)
			}
		},
	}
}

#[derive(Clone, Copy)]
enum State {
	Start,
	GotEsc,
	IgnoreNext,
	Csi,
	Osc,
	OscGotEsc,
	NeedSt,
	NeedStGotEsc,
}

#[inline(always)]
fn decode_at<E: Encoding>(input: &[E::Unit], at: usize) -> (u32, usize) {
	let mut rest = &input[at..];
	let before = rest.len();
	let codepoint = E::decode(&mut rest);
	(codepoint, at + before - rest.len())
}

#[inline(always)]
fn unit_value<E: Encoding>(unit: E::Unit) -> u32 {
	if E::FOREIGN {
		unit.swap_bytes().to_u32()
	} else {
		unit.to_u32()
	}
}

#[inline(always)]
fn consume_csi<E: Encoding>(input: &[E::Unit], start: usize) -> Option<usize> {
	if unit_value::<E>(input[start]) != ESC {
		return None;
	}
	if start + 1 == input.len() {
		return Some(input.len());
	}
	if unit_value::<E>(input[start + 1]) != 0x5b {
		return None;
	}
	let mut at = start + 2;
	while at < input.len() {
		if (0x40..=0x7e).contains(&unit_value::<E>(input[at])) {
			return Some(at + 1);
		}
		at += 1;
	}
	Some(input.len())
}

/// Consumes one or more adjacent ANSI sequences, returning the first retained
/// unit. A recognized but unterminated sequence consumes the remaining input.
#[inline]
fn consume_ansi<E: Encoding>(input: &[E::Unit], start: usize) -> usize {
	if let Some(end) = consume_csi::<E>(input, start) {
		return end;
	}
	let mut state = State::Start;
	let mut at = start;
	while at < input.len() {
		let current = at;
		let (c, next) = decode_at::<E>(input, at);
		state = match state {
			State::Start => match c {
				ESC => State::GotEsc,
				CSI => State::Csi,
				OSC => State::Osc,
				DCS | SOS | PM | APC => State::NeedSt,
				_ => return current,
			},
			State::GotEsc => match c {
				0x5b => State::Csi,
				0x20 | 0x23 | 0x25 | 0x28 | 0x29 | 0x2a | 0x2b | 0x2e | 0x2f => State::IgnoreNext,
				0x5d => State::Osc,
				0x50 | 0x58 | 0x5e | 0x5f => State::NeedSt,
				_ => State::Start,
			},
			State::IgnoreNext => State::Start,
			State::Csi => {
				if (0x40..=0x7e).contains(&c) {
					State::Start
				} else {
					State::Csi
				}
			},
			State::Osc => match c {
				ESC => State::OscGotEsc,
				ST | BEL => State::Start,
				_ => State::Osc,
			},
			State::OscGotEsc => {
				if c == 0x5c {
					State::Start
				} else {
					State::Osc
				}
			},
			State::NeedSt => match c {
				ESC => State::NeedStGotEsc,
				ST => State::Start,
				_ => State::NeedSt,
			},
			State::NeedStGotEsc => {
				if c == 0x5c {
					State::Start
				} else {
					State::NeedSt
				}
			},
		};
		at = next;
	}
	input.len()
}

#[inline(always)]
fn move_forward_n<U: Unit, const N: usize>(
	input: &mut [U],
	mut source: usize,
	mut destination: usize,
	end: usize,
) {
	debug_assert!(destination <= source && source <= end && end <= input.len());
	if destination == source {
		return;
	}
	while end - source >= N {
		let chunk = Simd::<U, N>::from_slice(&input[source..]);
		chunk.copy_to_slice(&mut input[destination..]);
		source += N;
		destination += N;
	}
	while source < end {
		input[destination] = input[source];
		source += 1;
		destination += 1;
	}
}

#[inline(always)]
fn move_forward<U: Unit>(input: &mut [U], source: usize, destination: usize, end: usize) {
	match size_of::<U>() {
		1 => move_forward_n::<U, 64>(input, source, destination, end),
		2 => move_forward_n::<U, 32>(input, source, destination, end),
		_ => move_forward_n::<U, 16>(input, source, destination, end),
	}
}

#[inline(always)]
fn strip_len_from<E: Encoding>(input: &mut [E::Unit], mut candidate: usize) -> usize {
	let (mut source, mut write) = loop {
		if candidate == input.len() {
			return input.len();
		}
		let after = consume_ansi::<E>(input, candidate);
		if after != candidate {
			break (after, candidate);
		}
		candidate = find_escape::<E>(input, candidate + 1);
	};

	loop {
		candidate = find_escape::<E>(input, source);
		move_forward(input, source, write, candidate);
		write += candidate - source;
		if candidate == input.len() {
			return write;
		}

		let after = consume_ansi::<E>(input, candidate);
		if after == candidate {
			move_forward(input, candidate, write, candidate + 1);
			write += 1;
			source = candidate + 1;
		} else {
			source = after;
		}
	}
}

#[inline]
fn strip_len<E: Encoding>(input: &mut [E::Unit]) -> usize {
	let candidate = find_escape::<E>(input, 0);
	strip_len_from::<E>(input, candidate)
}

#[inline]
unsafe fn copy_find_u8(input: &[u8], output: *mut u8) -> usize {
	let mut at = 0;
	while input.len() - at >= 256 {
		let a = Simd::<u8, 64>::from_slice(&input[at..]);
		let b = Simd::<u8, 64>::from_slice(&input[at + 64..]);
		let c = Simd::<u8, 64>::from_slice(&input[at + 128..]);
		let d = Simd::<u8, 64>::from_slice(&input[at + 192..]);
		// SAFETY: the caller provides writable space for `input.len()` bytes;
		// these four disjoint stores initialize the current 256-byte block.
		unsafe {
			core::ptr::write_unaligned(output.add(at).cast(), a);
			core::ptr::write_unaligned(output.add(at + 64).cast(), b);
			core::ptr::write_unaligned(output.add(at + 128).cast(), c);
			core::ptr::write_unaligned(output.add(at + 192).cast(), d);
		}

		let found = escape_mask_u8(a)
			.map(|index| at + index)
			.or_else(|| escape_mask_u8(b).map(|index| at + 64 + index))
			.or_else(|| escape_mask_u8(c).map(|index| at + 128 + index))
			.or_else(|| escape_mask_u8(d).map(|index| at + 192 + index));
		if let Some(found) = found {
			let rest = input.len() - at - 256;
			// SAFETY: the allocations are disjoint and both ranges contain
			// `rest` bytes after the initialized block.
			unsafe {
				core::ptr::copy_nonoverlapping(
					input.as_ptr().add(at + 256),
					output.add(at + 256),
					rest,
				);
			}
			return found;
		}
		at += 256;
	}
	while input.len() - at >= 64 {
		let chunk = Simd::<u8, 64>::from_slice(&input[at..]);
		// SAFETY: the caller provides writable space for this 64-byte block.
		unsafe { core::ptr::write_unaligned(output.add(at).cast(), chunk) };
		if let Some(found) = escape_mask_u8(chunk) {
			let rest = input.len() - at - 64;
			// SAFETY: the allocations are disjoint and both ranges contain
			// `rest` bytes after the initialized block.
			unsafe {
				core::ptr::copy_nonoverlapping(input.as_ptr().add(at + 64), output.add(at + 64), rest);
			}
			return at + found;
		}
		at += 64;
	}
	let rest = input.len() - at;
	// SAFETY: the allocations are disjoint and both ranges contain `rest`
	// bytes. This initializes the final partial block.
	unsafe { core::ptr::copy_nonoverlapping(input.as_ptr().add(at), output.add(at), rest) };
	while at < input.len() {
		if input[at] == ESC as u8 || input[at] == 0xc2 {
			return at;
		}
		at += 1;
	}
	input.len()
}

#[inline]
fn clone_utf8(input: &str) -> (Vec<u8>, usize) {
	let mut output = Vec::with_capacity(input.len());
	// SAFETY: the allocation has capacity for every input byte. The copy
	// initializes all of them before `set_len` exposes the elements.
	let candidate = unsafe { copy_find_u8(input.as_bytes(), output.as_mut_ptr()) };
	// SAFETY: `copy_find_u8` initialized exactly `input.len()` bytes.
	unsafe { output.set_len(input.len()) };
	(output, candidate)
}

/// Visible terminal width of `input`, ignoring ANSI/VT escape sequences.
///
/// Escape sequences form grapheme-cluster boundaries, matching terminal
/// execution: a combining mark separated from its base is measured as its own
/// cluster. C1 introducers are recognized, and an unterminated sequence makes
/// the remainder contribute zero width, matching stripping semantics.
pub fn width_ansi<E: Encoding>(input: &[E::Unit]) -> usize {
	let mut measured = 0;
	let mut text_start = 0;
	loop {
		let mut search_start = text_start;
		let (escape, after) = loop {
			let escape = find_escape::<E>(input, search_start);
			if escape == input.len() {
				return measured + crate::width::width::<E>(&input[text_start..]);
			}
			let after = consume_ansi::<E>(input, escape);
			if after != escape {
				break (escape, after);
			}
			// UTF-8 scans C2 as a possible encoded C1 introducer. Advance only
			// the search cursor so false positives remain in the measured span.
			search_start = escape + 1;
		};
		measured += crate::width::width::<E>(&input[text_start..escape]);
		text_start = after;
	}
}

/// Visible terminal width of a UTF-8 string, ignoring ANSI/VT escape sequences.
///
/// Escape sequences form grapheme-cluster boundaries. C1 introducers are
/// recognized, and an unterminated sequence makes the remainder contribute
/// zero width.
#[inline]
pub fn width_ansi_str(input: &str) -> usize {
	width_ansi::<Utf8>(input.as_bytes())
}

/// Compacts ANSI/VT-free text into a mutable code-unit slice view.
///
/// Implemented for native UTF-8 (`u8`), UTF-16 (`u16`), and UTF-32 (`u32`)
/// slices. The receiver's slice length is shortened; the backing allocation is
/// unchanged.
pub trait MakeAnsiStripped {
	/// Removes ANSI/VT sequences in place and shortens this slice view.
	fn make_ansi_stripped(&mut self);
}

macro_rules! impl_make_ansi_stripped {
	($unit:ty, $encoding:ty) => {
		impl MakeAnsiStripped for &mut [$unit] {
			#[inline]
			fn make_ansi_stripped(&mut self) {
				let len = strip_len::<$encoding>(&mut **self);
				let input = core::mem::take(self);
				*self = &mut input[..len];
			}
		}
	};
}

impl_make_ansi_stripped!(u8, Utf8);
impl_make_ansi_stripped!(u16, Utf16<false>);
impl_make_ansi_stripped!(u32, Utf32<false>);

/// Creates an ANSI/VT-free owned string from a borrowed string.
pub trait ToAnsiStripped {
	/// Copies this string once, strips the copy in place, and returns it.
	fn to_ansi_stripped(&self) -> String;
}

impl ToAnsiStripped for str {
	#[inline]
	fn to_ansi_stripped(&self) -> String {
		let (mut output, candidate) = clone_utf8(self);
		let len = strip_len_from::<Utf8>(&mut output, candidate);
		output.truncate(len);
		// SAFETY: `self` is valid UTF-8 and stripping removes whole codepoints.
		unsafe { String::from_utf8_unchecked(output) }
	}
}

/// Converts an owned string into ANSI/VT-free text without reallocating.
pub trait IntoAnsiStripped {
	/// Strips this string in place and returns the same allocation.
	fn into_ansi_stripped(self) -> String;
}

impl IntoAnsiStripped for String {
	#[inline]
	fn into_ansi_stripped(self) -> String {
		strip_string(self)
	}
}

#[inline]
fn strip_string(mut input: String) -> String {
	// SAFETY: the source is valid UTF-8 and stripping only removes whole decoded
	// codepoints, so compaction preserves UTF-8 validity. Truncation occurs
	// before the mutable byte-vector borrow ends.
	let bytes = unsafe { input.as_mut_vec() };
	let len = strip_len::<Utf8>(bytes);
	bytes.truncate(len);
	input
}
