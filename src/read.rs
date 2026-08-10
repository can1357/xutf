//! Strict, error-reporting char decoding from [`BufRead`] streams.
//!
//! The rest of the crate is permissive; this module is the opposite: decoding
//! from an I/O stream validates every sequence and reports the offending
//! bytes. [`BufReadCharsExt`] is a drop-in replacement for the `utf8-chars`
//! crate's trait of the same name, extended to every encoding in this crate
//! via [`BufReadCharsExt::decode_chars`].
//!
//! ```
//! use std::io::BufReader;
//!
//! use xutf::BufReadCharsExt;
//!
//! let mut input = BufReader::new("caf\u{e9}".as_bytes());
//! let text: String = input.chars().collect::<Result<_, _>>().unwrap();
//! assert_eq!(text, "café");
//! ```
//!
//! Unlike a byte-at-a-time reader, iteration decodes directly from the
//! reader's buffer in batches, so throughput tracks the buffer size rather
//! than the per-call overhead of [`BufRead::fill_buf`].
//!
//! # Consumption guarantees
//!
//! A char's bytes are consumed from the reader only when that char (or its
//! error) is yielded, so dropping an iterator mid-stream leaves the reader
//! positioned after the last yielded item. Two deviations, both inherent to
//! lookahead over `BufRead`:
//!
//! - UTF-8: an invalid continuation byte is *not* consumed with its failed
//!   sequence (it may start the next char), matching `utf8-chars`.
//! - UTF-16/UTF-32: code units span multiple bytes, so validating a surrogate
//!   pair may buffer the following unit internally. Iterating to the end always
//!   accounts for every byte, but dropping a [`Chars`] right after a
//!   lone-surrogate error can lose that one unit of lookahead.

use core::{fmt, marker::PhantomData};
use std::{
	error::Error,
	io::{self, BufRead, ErrorKind},
};

use crate::{encoding::Encoding, utf8::Utf8, utf16::Utf16, utf32::Utf32};

/// Decoded-char lookahead per [`BufRead::fill_buf`] call.
const QUEUE: usize = 64;

/// Bytes read from a stream that failed to decode, plus the [`io::Error`]
/// classifying the failure.
///
/// The error kind is [`InvalidData`](ErrorKind::InvalidData) for a malformed
/// sequence, [`UnexpectedEof`](ErrorKind::UnexpectedEof) for a sequence cut
/// short by end of stream, or a real I/O error propagated from the reader.
#[derive(Debug)]
pub struct ReadCharError {
	bytes:    [u8; 4],
	len:      u8,
	io_error: io::Error,
}

impl ReadCharError {
	fn new(bytes: &[u8], io_error: io::Error) -> Self {
		let mut store = [0u8; 4];
		store[..bytes.len()].copy_from_slice(bytes);
		Self { bytes: store, len: bytes.len() as u8, io_error }
	}

	fn invalid(bytes: &[u8]) -> Self {
		Self::new(bytes, ErrorKind::InvalidData.into())
	}

	fn eof(bytes: &[u8]) -> Self {
		Self::new(bytes, ErrorKind::UnexpectedEof.into())
	}

	/// The raw bytes, in stream order, of the invalid or incomplete sequence.
	pub fn as_bytes(&self) -> &[u8] {
		&self.bytes[..self.len as usize]
	}

	/// The I/O error classifying this failure.
	pub const fn as_io_error(&self) -> &io::Error {
		&self.io_error
	}

	/// Converts into an [`io::Error`], keeping the byte context as the error
	/// payload (and `self` reachable through [`Error::source`]).
	pub fn into_io_error(self) -> io::Error {
		if self.len == 0 {
			return self.io_error;
		}
		io::Error::new(self.io_error.kind(), self)
	}
}

impl Error for ReadCharError {
	fn source(&self) -> Option<&(dyn Error + 'static)> {
		Some(&self.io_error)
	}
}

impl From<ReadCharError> for io::Error {
	fn from(e: ReadCharError) -> Self {
		e.into_io_error()
	}
}

impl fmt::Display for ReadCharError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "invalid byte sequence")?;
		for b in self.as_bytes() {
			write!(f, " {b:02X}")?;
		}
		write!(f, " read")?;
		match self.io_error.kind() {
			ErrorKind::InvalidData => Ok(()),
			ErrorKind::UnexpectedEof => write!(f, " (unexpected EOF)"),
			_ => write!(f, " ({})", self.io_error),
		}
	}
}

/// Decoder state shared by [`Chars`] / [`CharsRaw`]: decoded lookahead whose
/// bytes are consumed on yield, plus already-consumed bytes of a code unit
/// that straddled the reader's buffer.
#[doc(hidden)]
#[derive(Debug)]
pub struct State {
	queue:       [(char, u8); QUEUE],
	head:        u8,
	tail:        u8,
	pending:     [u8; 4],
	pending_len: u8,
}

impl State {
	const fn new() -> Self {
		Self {
			queue:       [('\0', 0); QUEUE],
			head:        0,
			tail:        0,
			pending:     [0; 4],
			pending_len: 0,
		}
	}

	#[inline(always)]
	fn pop(&mut self) -> Option<(char, u8)> {
		if self.head == self.tail {
			self.head = 0;
			self.tail = 0;
			return None;
		}
		// SAFETY: `push` caps `tail` at QUEUE and `head < tail` here.
		let entry = unsafe { *self.queue.get_unchecked(self.head as usize) };
		self.head += 1;
		Some(entry)
	}

	#[inline(always)]
	fn push(&mut self, c: char, debt: u8) {
		debug_assert!((self.tail as usize) < QUEUE);
		// SAFETY: every `push` call site checks `tail < QUEUE` first.
		unsafe { *self.queue.get_unchecked_mut(self.tail as usize) = (c, debt) };
		self.tail += 1;
	}

	#[inline(always)]
	const fn queued(&self) -> usize {
		(self.tail - self.head) as usize
	}

	/// Discards the first `n` pending bytes, keeping any lookahead.
	fn drop_pending(&mut self, n: usize) {
		self.pending.copy_within(n..self.pending_len as usize, 0);
		self.pending_len -= n as u8;
	}

	/// Takes all pending bytes for error reporting.
	const fn take_pending(&mut self) -> ([u8; 4], usize) {
		let bytes = self.pending;
		let len = self.pending_len as usize;
		self.pending_len = 0;
		(bytes, len)
	}
}

/// Batch-decode verdict for the front of the reader's buffer.
#[doc(hidden)]
#[derive(Debug)]
pub enum Decide {
	/// At least one char was queued.
	Queued,
	/// The front sequence is invalid; consume the given byte count.
	Error(ReadCharError, u8),
	/// The front sequence straddles the buffer or hits EOF; the exact
	/// per-byte path must resolve it.
	Slow,
}

/// Copies the next unread byte without consuming it, retrying on
/// [`ErrorKind::Interrupted`]. `None` means end of stream.
fn peek_byte<R: BufRead + ?Sized>(reader: &mut R) -> io::Result<Option<u8>> {
	loop {
		match reader.fill_buf() {
			Ok(buf) => return Ok(buf.first().copied()),
			Err(e) if e.kind() == ErrorKind::Interrupted => {},
			Err(e) => return Err(e),
		}
	}
}

/// Consumes bytes into `state.pending` until it holds `target` bytes.
///
/// `Ok(false)` reports a clean end of stream (nothing pending); a partial
/// unit at EOF or an I/O error takes the pending bytes into the error.
fn fill_pending<R: BufRead + ?Sized>(
	reader: &mut R,
	state: &mut State,
	target: usize,
) -> Result<bool, ReadCharError> {
	while (state.pending_len as usize) < target {
		match peek_byte(reader) {
			Err(e) => {
				let (bytes, len) = state.take_pending();
				return Err(ReadCharError::new(&bytes[..len], e));
			},
			Ok(None) => {
				if state.pending_len == 0 {
					return Ok(false);
				}
				let (bytes, len) = state.take_pending();
				return Err(ReadCharError::eof(&bytes[..len]));
			},
			Ok(Some(b)) => {
				state.pending[state.pending_len as usize] = b;
				state.pending_len += 1;
				reader.consume(1);
			},
		}
	}
	Ok(true)
}

/// Minimum codepoint for a UTF-8 sequence of the indexed length (rejects
/// overlong encodings).
const UTF8_MIN: [u32; 5] = [0, 0, 0x80, 0x800, 0x10000];

/// Exact single-char UTF-8 decode with `utf8-chars` semantics: the lead byte
/// and valid continuation bytes are consumed, an invalid continuation byte is
/// left in the stream.
fn utf8_step<R: BufRead + ?Sized>(reader: &mut R) -> Result<Option<char>, ReadCharError> {
	// Snapshot up to one sequence from the current buffer.
	let (chunk, avail) = loop {
		match reader.fill_buf() {
			Ok(buf) => {
				if buf.is_empty() {
					return Ok(None);
				}
				let n = buf.len().min(4);
				let mut chunk = [0u8; 4];
				chunk[..n].copy_from_slice(&buf[..n]);
				break (chunk, n);
			},
			Err(e) if e.kind() == ErrorKind::Interrupted => {},
			Err(e) => return Err(ReadCharError::new(&[], e)),
		}
	};
	let lead = chunk[0];
	if lead < 0x80 {
		reader.consume(1);
		return Ok(Some(lead as char));
	}
	let need = lead.leading_ones() as usize;
	if !(2..=4).contains(&need) {
		reader.consume(1);
		return Err(ReadCharError::invalid(&chunk[..1]));
	}
	if avail >= need {
		// Sequence wholly buffered: validate in place, one consume.
		for (i, &tail) in chunk[1..need].iter().enumerate() {
			if tail & 0xc0 != 0x80 {
				reader.consume(i + 1);
				return Err(ReadCharError::invalid(&chunk[..=i]));
			}
		}
		reader.consume(need);
		return utf8_finish(&chunk[..need]);
	}
	// Sequence straddles the buffer: continue byte-wise.
	let mut bytes = chunk;
	let mut len = 1;
	reader.consume(1);
	while len < need {
		let tail = match peek_byte(reader) {
			Err(e) => return Err(ReadCharError::new(&bytes[..len], e)),
			Ok(None) => return Err(ReadCharError::eof(&bytes[..len])),
			Ok(Some(b)) => b,
		};
		if tail & 0xc0 != 0x80 {
			return Err(ReadCharError::invalid(&bytes[..len]));
		}
		bytes[len] = tail;
		len += 1;
		reader.consume(1);
	}
	utf8_finish(&bytes[..need])
}

/// Range checks for a fully-read multi-byte sequence (all bytes consumed).
fn utf8_finish(bytes: &[u8]) -> Result<Option<char>, ReadCharError> {
	let need = bytes.len();
	let mut cp = (bytes[0] as u32) & (0x7f >> need);
	for &tail in &bytes[1..] {
		cp = (cp << 6) | (tail & 0x3f) as u32;
	}
	if cp < UTF8_MIN[need] {
		return Err(ReadCharError::invalid(bytes));
	}
	// `from_u32` rejects surrogates and values above U+10FFFF.
	char::from_u32(cp)
		.map(Some)
		.ok_or_else(|| ReadCharError::invalid(bytes))
}

/// Reads one native-order code unit out of raw stream bytes.
#[inline(always)]
fn unit16<const FOREIGN: bool>(bytes: [u8; 2]) -> u32 {
	let unit = u16::from_ne_bytes(bytes);
	(if FOREIGN { unit.swap_bytes() } else { unit }) as u32
}

/// An encoding whose chars can be strictly decoded from a byte stream.
///
/// Implemented by [`Utf8`], [`Utf16`] and [`Utf32`] in both byte orders;
/// sealed via [`Encoding`].
pub trait StreamDecode: Encoding {
	/// Batch-decodes valid chars from the front of `buf` into the queue,
	/// stopping at the first malformed or straddling sequence.
	#[doc(hidden)]
	fn decide(buf: &[u8], state: &mut State) -> Decide;

	/// Exactly decodes one char (or its error) from the front of the stream.
	#[doc(hidden)]
	fn step<R: BufRead + ?Sized>(
		reader: &mut R,
		state: &mut State,
	) -> Result<Option<char>, ReadCharError>;
}

impl StreamDecode for Utf8 {
	fn decide(buf: &[u8], state: &mut State) -> Decide {
		/// One-sequence outcome inside the buffered slice.
		enum Seq {
			Char(char, u8),
			/// `len` bytes are both the error report and the consume count
			/// (an invalid continuation byte itself stays unconsumed).
			Bad(u8),
			Incomplete,
		}

		#[inline(always)]
		fn seq(buf: &[u8], i: usize) -> Seq {
			let lead = buf[i];
			let need = lead.leading_ones() as usize;
			if !(2..=4).contains(&need) {
				return Seq::Bad(1);
			}
			let mut cp = (lead as u32) & (0x7f >> need);
			let avail = need.min(buf.len() - i);
			for k in 1..avail {
				let tail = buf[i + k];
				if tail & 0xc0 != 0x80 {
					return Seq::Bad(k as u8);
				}
				cp = (cp << 6) | (tail & 0x3f) as u32;
			}
			if avail < need {
				return Seq::Incomplete;
			}
			if cp < UTF8_MIN[need] {
				return Seq::Bad(need as u8);
			}
			// `from_u32` rejects surrogates and values above U+10FFFF.
			match char::from_u32(cp) {
				Some(c) => Seq::Char(c, need as u8),
				None => Seq::Bad(need as u8),
			}
		}

		let n = buf.len();
		let mut i = 0;
		while i < n && (state.tail as usize) < QUEUE {
			let lead = buf[i];
			if lead < 0x80 {
				// Gulp 8 ASCII bytes at a time.
				if i + 8 <= n && state.tail as usize + 8 <= QUEUE {
					let word = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
					if word & 0x8080_8080_8080_8080 == 0 {
						for k in 0..8 {
							state.push(buf[i + k] as char, 1);
						}
						i += 8;
						continue;
					}
				}
				state.push(lead as char, 1);
				i += 1;
				continue;
			}
			match seq(buf, i) {
				Seq::Char(c, len) => {
					state.push(c, len);
					i += len as usize;
				},
				Seq::Bad(len) => {
					// An error surfaces immediately only at the front of the
					// stream; behind queued chars it is re-derived once their
					// bytes have been consumed.
					if state.queued() == 0 {
						let len = len as usize;
						return Decide::Error(ReadCharError::invalid(&buf[i..i + len]), len as u8);
					}
					return Decide::Queued;
				},
				Seq::Incomplete => break,
			}
		}
		if state.queued() > 0 {
			Decide::Queued
		} else {
			Decide::Slow
		}
	}

	fn step<R: BufRead + ?Sized>(
		reader: &mut R,
		_state: &mut State,
	) -> Result<Option<char>, ReadCharError> {
		utf8_step(reader)
	}
}

impl<const FOREIGN: bool> StreamDecode for Utf16<FOREIGN> {
	fn decide(buf: &[u8], state: &mut State) -> Decide {
		debug_assert_eq!(state.pending_len, 0);
		let n = buf.len();
		let mut i = 0;
		while i + 2 <= n && (state.tail as usize) < QUEUE {
			let unit = unit16::<FOREIGN>([buf[i], buf[i + 1]]);
			if !(0xd800..0xe000).contains(&unit) {
				// SAFETY: not a surrogate, so a valid BMP scalar.
				state.push(unsafe { char::from_u32_unchecked(unit) }, 2);
				i += 2;
			} else if unit < 0xdc00 && i + 4 <= n {
				let low = unit16::<FOREIGN>([buf[i + 2], buf[i + 3]]);
				if !(0xdc00..0xe000).contains(&low) {
					break;
				}
				let cp = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
				// SAFETY: surrogate pairs decode to U+10000..=U+10FFFF.
				state.push(unsafe { char::from_u32_unchecked(cp) }, 4);
				i += 4;
			} else {
				break;
			}
		}
		if state.queued() > 0 {
			return Decide::Queued;
		}
		// Queue empty with at least 2 buffered bytes: the front unit is a
		// lone surrogate (anything else would have queued or straddled).
		if n < 2 {
			return Decide::Slow;
		}
		let unit = unit16::<FOREIGN>([buf[0], buf[1]]);
		debug_assert!((0xd800..0xe000).contains(&unit));
		if unit < 0xdc00 && n < 4 {
			// High surrogate whose pair straddles the buffer.
			return Decide::Slow;
		}
		// Lone low surrogate, or high surrogate whose follower is not a low
		// surrogate; the follower stays buffered as the next char.
		Decide::Error(ReadCharError::invalid(&buf[..2]), 2)
	}

	fn step<R: BufRead + ?Sized>(
		reader: &mut R,
		state: &mut State,
	) -> Result<Option<char>, ReadCharError> {
		if !fill_pending(reader, state, 2)? {
			return Ok(None);
		}
		let unit = unit16::<FOREIGN>([state.pending[0], state.pending[1]]);
		if !(0xd800..0xe000).contains(&unit) {
			state.drop_pending(2);
			// SAFETY: not a surrogate, so a valid BMP scalar.
			return Ok(Some(unsafe { char::from_u32_unchecked(unit) }));
		}
		if unit >= 0xdc00 {
			let e = ReadCharError::invalid(&state.pending[..2]);
			state.drop_pending(2);
			return Err(e);
		}
		// High surrogate: the pair is invalid without a low unit, and a
		// non-low follower stays pending as lookahead for the next char.
		fill_pending(reader, state, 4)?;
		let low = unit16::<FOREIGN>([state.pending[2], state.pending[3]]);
		if (0xdc00..0xe000).contains(&low) {
			let cp = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
			state.drop_pending(4);
			// SAFETY: surrogate pairs decode to U+10000..=U+10FFFF.
			return Ok(Some(unsafe { char::from_u32_unchecked(cp) }));
		}
		let e = ReadCharError::invalid(&state.pending[..2]);
		state.drop_pending(2);
		Err(e)
	}
}

impl<const FOREIGN: bool> StreamDecode for Utf32<FOREIGN> {
	fn decide(buf: &[u8], state: &mut State) -> Decide {
		debug_assert_eq!(state.pending_len, 0);
		let n = buf.len();
		let mut i = 0;
		while i + 4 <= n && (state.tail as usize) < QUEUE {
			let unit = u32::from_ne_bytes(buf[i..i + 4].try_into().unwrap());
			let unit = if FOREIGN { unit.swap_bytes() } else { unit };
			match char::from_u32(unit) {
				Some(c) => {
					state.push(c, 4);
					i += 4;
				},
				None => break,
			}
		}
		if state.queued() > 0 {
			return Decide::Queued;
		}
		if n < 4 {
			return Decide::Slow;
		}
		// Complete unit that is not a valid scalar.
		Decide::Error(ReadCharError::invalid(&buf[..4]), 4)
	}

	fn step<R: BufRead + ?Sized>(
		reader: &mut R,
		state: &mut State,
	) -> Result<Option<char>, ReadCharError> {
		if !fill_pending(reader, state, 4)? {
			return Ok(None);
		}
		let unit = u32::from_ne_bytes(state.pending);
		let unit = if FOREIGN { unit.swap_bytes() } else { unit };
		let (bytes, len) = state.take_pending();
		char::from_u32(unit)
			.map(Some)
			.ok_or_else(|| ReadCharError::invalid(&bytes[..len]))
	}
}

/// Produces the next char: batch-decodes from the reader's buffer when the
/// front of the stream is clean, otherwise defers to the exact path.
fn pump<E: StreamDecode, R: BufRead + ?Sized>(
	reader: &mut R,
	state: &mut State,
) -> Option<Result<char, ReadCharError>> {
	// Callers only reach here with the queue drained, but `head`/`tail` may
	// both sit at capacity after an exactly-drained batch; rewind so `decide`
	// has room to queue.
	debug_assert_eq!(state.head, state.tail);
	state.head = 0;
	state.tail = 0;
	loop {
		if state.pending_len != 0 {
			return E::step(reader, state).transpose();
		}
		let decide = match reader.fill_buf() {
			Err(e) if e.kind() == ErrorKind::Interrupted => continue,
			Err(e) => return Some(Err(ReadCharError::new(&[], e))),
			Ok([]) => return None,
			Ok(buf) => E::decide(buf, state),
		};
		match decide {
			Decide::Queued => {
				let (c, debt) = state.pop().expect("batch queued at least one char");
				reader.consume(debt as usize);
				return Some(Ok(c));
			},
			Decide::Error(e, taken) => {
				reader.consume(taken as usize);
				return Some(Err(e));
			},
			Decide::Slow => return E::step(reader, state).transpose(),
		}
	}
}

/// Iterator over the chars of a [`BufRead`], yielding
/// <code>[Result]&lt;char, [ReadCharError]&gt;</code>.
///
/// Created by [`chars_raw`](BufReadCharsExt::chars_raw) (UTF-8) or
/// [`decode_chars_raw`](BufReadCharsExt::decode_chars_raw) (any encoding).
#[derive(Debug)]
pub struct CharsRaw<'a, R: BufRead + ?Sized, E: StreamDecode = Utf8> {
	reader:    &'a mut R,
	state:     State,
	_encoding: PhantomData<E>,
}

/// Outstanding [`BufRead::consume`] debt for chars already handed to a fold
/// callback, settled on scope exit including unwind.
struct Debt<'r, R: BufRead + ?Sized> {
	reader: &'r mut R,
	owed:   usize,
}

impl<R: BufRead + ?Sized> Drop for Debt<'_, R> {
	fn drop(&mut self) {
		self.reader.consume(self.owed);
	}
}

impl<R: BufRead + ?Sized, E: StreamDecode> Iterator for CharsRaw<'_, R, E> {
	type Item = Result<char, ReadCharError>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		if let Some((c, debt)) = self.state.pop() {
			self.reader.consume(debt as usize);
			return Some(Ok(c));
		}
		pump::<E, R>(self.reader, &mut self.state)
	}

	/// Internal iteration (`count`, `for_each`, `collect`, `last`, ...) drains
	/// each decoded batch in a tight loop, paying its debt with a single
	/// [`BufRead::consume`] call instead of one per char.
	///
	/// The debt is settled by a guard, so an unwinding `f` still leaves the
	/// reader positioned exactly after the char it was handed: the rest of
	/// the batch is neither skipped nor re-delivered.
	fn fold<B, F>(mut self, init: B, mut f: F) -> B
	where
		F: FnMut(B, Self::Item) -> B,
	{
		let mut acc = init;
		loop {
			if self.state.head != self.state.tail {
				let mut debt = Debt { reader: &mut *self.reader, owed: 0 };
				while let Some((c, d)) = self.state.pop() {
					// Charged before the callback: an unwind out of `f`
					// still consumes this char.
					debt.owed += d as usize;
					acc = f(acc, Ok(c));
				}
				drop(debt);
			}
			match pump::<E, R>(self.reader, &mut self.state) {
				Some(item) => acc = f(acc, item),
				None => return acc,
			}
		}
	}
}

/// Iterator over the chars of a [`BufRead`], yielding
/// <code>[io::Result]&lt;char&gt;</code>.
///
/// Like [`CharsRaw`] but with [`io::Error`] items for drop-in compatibility;
/// the byte context stays reachable via the error payload. Created by
/// [`chars`](BufReadCharsExt::chars) or
/// [`decode_chars`](BufReadCharsExt::decode_chars).
#[derive(Debug)]
pub struct Chars<'a, R: BufRead + ?Sized, E: StreamDecode = Utf8>(CharsRaw<'a, R, E>);

impl<R: BufRead + ?Sized, E: StreamDecode> Iterator for Chars<'_, R, E> {
	type Item = io::Result<char>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|r| r.map_err(io::Error::from))
	}

	fn fold<B, F>(self, init: B, mut f: F) -> B
	where
		F: FnMut(B, Self::Item) -> B,
	{
		self
			.0
			.fold(init, |acc, r| f(acc, r.map_err(io::Error::from)))
	}
}

/// Extends [`BufRead`] with strict char decoding, batch-accelerated over the
/// reader's internal buffer.
pub trait BufReadCharsExt: BufRead {
	/// Iterates the UTF-8 chars of this reader as
	/// <code>[io::Result]&lt;char&gt;</code>.
	///
	/// # Example
	/// ```
	/// use std::io::BufReader;
	///
	/// use xutf::BufReadCharsExt;
	///
	/// let mut input = BufReader::new("héllo".as_bytes());
	/// let text: String = input.chars().collect::<Result<_, _>>().unwrap();
	/// assert_eq!(text, "héllo");
	/// ```
	fn chars(&mut self) -> Chars<'_, Self, Utf8> {
		self.decode_chars()
	}

	/// Iterates the UTF-8 chars of this reader, with errors carrying the
	/// offending bytes ([`ReadCharError`]).
	fn chars_raw(&mut self) -> CharsRaw<'_, Self, Utf8> {
		self.decode_chars_raw()
	}

	/// Iterates the chars of this reader in encoding `E` as
	/// <code>[io::Result]&lt;char&gt;</code>.
	///
	/// # Example
	/// ```
	/// use std::io::BufReader;
	///
	/// use xutf::{BufReadCharsExt, Utf16Be};
	///
	/// let mut input = BufReader::new(&[0x00, 0x68, 0x00, 0x69][..]);
	/// let text: String = input
	/// 	.decode_chars::<Utf16Be>()
	/// 	.collect::<Result<_, _>>()
	/// 	.unwrap();
	/// assert_eq!(text, "hi");
	/// ```
	fn decode_chars<E: StreamDecode>(&mut self) -> Chars<'_, Self, E> {
		Chars(self.decode_chars_raw())
	}

	/// Iterates the chars of this reader in encoding `E`, with errors
	/// carrying the offending bytes ([`ReadCharError`]).
	fn decode_chars_raw<E: StreamDecode>(&mut self) -> CharsRaw<'_, Self, E> {
		CharsRaw { reader: self, state: State::new(), _encoding: PhantomData }
	}

	/// Reads one UTF-8 char; `Ok(None)` at end of stream.
	///
	/// [`ErrorKind::Interrupted`] reads are retried.
	fn read_char(&mut self) -> io::Result<Option<char>> {
		self.read_char_raw().map_err(io::Error::from)
	}

	/// Reads one UTF-8 char, with errors carrying the offending bytes;
	/// `Ok(None)` at end of stream.
	///
	/// [`ErrorKind::Interrupted`] reads are retried.
	fn read_char_raw(&mut self) -> Result<Option<char>, ReadCharError> {
		utf8_step(self)
	}
}

impl<T: BufRead + ?Sized> BufReadCharsExt for T {}
