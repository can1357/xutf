//! Compile-time shuffle tables for mixed-width UTF blocks.

/// Six decoded codepoints selected from up to twelve UTF-8 bytes.
#[derive(Clone, Copy)]
pub struct Utf8OneTwo {
	/// Byte indexes for six little-endian `u16` lanes; `0xFF` zero-fills.
	pub shuffle:  [u8; 16],
	/// Input bytes represented by the six output lanes; zero marks an
	/// invalid pattern (a sequence wider than two bytes in the window).
	pub consumed: u8,
}

const EMPTY_UTF8_ONE_TWO: Utf8OneTwo = Utf8OneTwo { shuffle: [0xff; 16], consumed: 0 };

/// Mixed ASCII/two-byte decoder indexed by the low twelve *end-of-codepoint*
/// bits (bit `p` set when byte `p` terminates a codepoint). Widths follow
/// from set-bit gaps alone, so no lookahead past the index is ever needed.
pub static UTF8_ONE_TWO: [Utf8OneTwo; 4096] = {
	let mut table = [EMPTY_UTF8_ONE_TWO; 4096];
	let mut mask = 0usize;
	while mask < table.len() {
		let mut row = EMPTY_UTF8_ONE_TWO;
		let mut pos = 0usize;
		let mut lane = 0usize;
		let mut valid = true;
		while lane < 6 {
			if ((mask >> pos) & 1) != 0 {
				// This byte ends its codepoint: one-byte sequence.
				row.shuffle[lane * 2] = pos as u8;
				pos += 1;
			} else if ((mask >> (pos + 1)) & 1) != 0 {
				// Ends one byte later: two-byte sequence.
				row.shuffle[lane * 2] = (pos + 1) as u8;
				row.shuffle[lane * 2 + 1] = pos as u8;
				pos += 2;
			} else {
				// Wider than two bytes.
				valid = false;
				break;
			}
			lane += 1;
		}
		if valid {
			row.consumed = pos as u8;
		}
		table[mask] = row;
		mask += 1;
	}
	table
};

/// Up to twelve decoded codepoints from one mixed ASCII/two-byte window.
#[derive(Clone, Copy)]
pub struct Utf8OneTwoWide {
	/// Byte indexes for lanes 0-7 as little-endian `u16` pairs.
	pub lo:      [u8; 16],
	/// Byte indexes for lanes 8-11; the tail zero-fills.
	pub hi:      [u8; 16],
	/// Decoded codepoints; zero marks a window containing a wider sequence.
	pub written: u8,
}

const EMPTY_UTF8_ONE_TWO_WIDE: Utf8OneTwoWide =
	Utf8OneTwoWide { lo: [0xff; 16], hi: [0xff; 16], written: 0 };

/// Whole-window variant of [`UTF8_ONE_TWO`]: decodes every complete one- or
/// two-byte sequence in the twelve-bit window, so callers derive the byte
/// advance from the mask alone (`64 - clz`), keeping the row load off the
/// window-stride dependency chain. A row is empty when a wider sequence
/// starts inside the window.
pub static UTF8_ONE_TWO_WIDE: [Utf8OneTwoWide; 4096] = {
	let mut table = [EMPTY_UTF8_ONE_TWO_WIDE; 4096];
	let mut mask = 1usize;
	while mask < table.len() {
		let mut row = EMPTY_UTF8_ONE_TWO_WIDE;
		let mut pos = 0usize;
		let mut lane = 0usize;
		let mut valid = true;
		while pos < 12 {
			let (end, start) = if ((mask >> pos) & 1) != 0 {
				(pos, 0xff)
			} else if pos + 1 < 12 && ((mask >> (pos + 1)) & 1) != 0 {
				(pos + 1, pos)
			} else if pos + 1 >= 12 {
				// The trailing sequence completes in the next window.
				break;
			} else {
				// Wider than two bytes.
				valid = false;
				break;
			};
			if lane < 8 {
				row.lo[lane * 2] = end as u8;
				row.lo[lane * 2 + 1] = start as u8;
			} else {
				row.hi[(lane - 8) * 2] = end as u8;
				row.hi[(lane - 8) * 2 + 1] = start as u8;
			}
			pos = end + 1;
			lane += 1;
		}
		if valid && lane != 0 {
			row.written = lane as u8;
		}
		table[mask] = row;
		mask += 1;
	}
	table
};

/// Eight UTF-16 lanes compressed to UTF-8 according to an ASCII-lane mask.
#[derive(Clone, Copy)]
pub struct Utf16OneTwo {
	/// Byte indexes for the packed output; `0xFF` zero-fills.
	pub shuffle: [u8; 16],
	/// Number of live output bytes.
	pub written: u8,
}

const EMPTY_UTF16_ONE_TWO: Utf16OneTwo = Utf16OneTwo { shuffle: [0xff; 16], written: 0 };

const fn build_utf16_one_two() -> [Utf16OneTwo; 256] {
	let mut table = [EMPTY_UTF16_ONE_TWO; 256];
	let mut mask = 0usize;
	while mask < table.len() {
		let mut row = EMPTY_UTF16_ONE_TWO;
		let mut lane = 0usize;
		let mut out = 0usize;
		while lane < 8 {
			row.shuffle[out] = (lane * 2) as u8;
			out += 1;
			if ((mask >> lane) & 1) == 0 {
				row.shuffle[out] = (lane * 2 + 1) as u8;
				out += 1;
			}
			lane += 1;
		}
		row.written = out as u8;
		table[mask] = row;
		mask += 1;
	}
	table
}

/// Mixed UTF-16 one/two-byte packer indexed by ASCII lanes (`1` = ASCII).
pub static UTF16_ONE_TWO: [Utf16OneTwo; 256] = build_utf16_one_two();

/// Four decoded codepoints selected from up to twelve UTF-8 bytes of mixed
/// one/two/three-byte sequences.
#[derive(Clone, Copy)]
pub struct Utf8OneTwoThree {
	/// Byte indexes for four little-endian `u32` lanes laid out as
	/// `[last, mid, lead, zero]`; `0xFF` zero-fills missing bytes.
	pub shuffle:  [u8; 16],
	/// Input bytes represented by the four lanes; zero marks a window the
	/// vector decoder must reject (stray continuation, 4-byte sequence, or a
	/// sequence that cannot be width-classified inside the 12-bit mask).
	pub consumed: u8,
}

const EMPTY_UTF8_ONE_TWO_THREE: Utf8OneTwoThree =
	Utf8OneTwoThree { shuffle: [0xff; 16], consumed: 0 };

/// Mixed one/two/three-byte decoder indexed by the low twelve
/// end-of-codepoint bits; four codepoints per row.
pub static UTF8_ONE_TWO_THREE: [Utf8OneTwoThree; 4096] = {
	let mut table = [EMPTY_UTF8_ONE_TWO_THREE; 4096];
	let mut mask = 0usize;
	while mask < table.len() {
		let mut row = EMPTY_UTF8_ONE_TWO_THREE;
		let mut pos = 0usize;
		let mut lane = 0usize;
		let mut valid = true;
		while lane < 4 {
			let width = if ((mask >> pos) & 1) != 0 {
				1
			} else if ((mask >> (pos + 1)) & 1) != 0 {
				2
			} else if ((mask >> (pos + 2)) & 1) != 0 {
				3
			} else {
				// Wider than three bytes.
				valid = false;
				break;
			};
			let mut byte = 0usize;
			while byte < width {
				row.shuffle[lane * 4 + byte] = (pos + width - 1 - byte) as u8;
				byte += 1;
			}
			pos += width;
			lane += 1;
		}
		if valid {
			row.consumed = pos as u8;
		}
		table[mask] = row;
		mask += 1;
	}
	table
};

/// Four UTF-8 encodings (widths one to three) compacted from four-byte zip
/// groups laid out `[ascii, cont, lead3, lead2/mid]` (see `pack_utf16_1_2_3`).
#[derive(Clone, Copy)]
pub struct Utf8OneTwoThreePack {
	/// Byte indexes for the packed output; `0xFF` zero-fills.
	pub shuffle: [u8; 16],
	/// Number of live output bytes.
	pub written: u8,
}

/// Packer rows indexed by four interleaved lane flags: bit `2k` set when
/// lane `k` is one byte, bit `2k + 1` set when it is at most two.
pub static UTF8_ONE_TWO_THREE_PACK: [Utf8OneTwoThreePack; 256] = {
	let mut table = [Utf8OneTwoThreePack { shuffle: [0xff; 16], written: 0 }; 256];
	let mut flags = 0usize;
	while flags < table.len() {
		let mut row = Utf8OneTwoThreePack { shuffle: [0xff; 16], written: 0 };
		let mut lane = 0usize;
		let mut out = 0usize;
		while lane < 4 {
			let base = (lane * 4) as u8;
			let one = (flags >> (lane * 2)) & 1 != 0;
			let one_or_two = (flags >> (lane * 2 + 1)) & 1 != 0;
			if one {
				row.shuffle[out] = base;
				out += 1;
			} else if one_or_two {
				row.shuffle[out] = base + 3;
				row.shuffle[out + 1] = base + 1;
				out += 2;
			} else {
				row.shuffle[out] = base + 2;
				row.shuffle[out + 1] = base + 3;
				row.shuffle[out + 2] = base + 1;
				out += 3;
			}
			lane += 1;
		}
		row.written = out as u8;
		table[flags] = row;
		flags += 1;
	}
	table
};

/// Four UTF-8 encodings compacted from four lane-aligned four-byte slots.
#[derive(Clone, Copy)]
pub struct Utf8Four {
	/// Byte indexes for the packed output; `0xFF` zero-fills.
	pub shuffle: [u8; 16],
	/// Number of live output bytes.
	pub written: u8,
}

const fn build_utf8_four() -> [Utf8Four; 256] {
	let mut table = [Utf8Four { shuffle: [0xff; 16], written: 0 }; 256];
	let mut widths = 0usize;
	while widths < table.len() {
		let mut row = Utf8Four { shuffle: [0xff; 16], written: 0 };
		let mut lane = 0usize;
		let mut out = 0usize;
		while lane < 4 {
			let width = ((widths >> (lane * 2)) & 3) + 1;
			let mut byte = 0usize;
			while byte < width {
				row.shuffle[out] = (lane * 4 + byte) as u8;
				out += 1;
				byte += 1;
			}
			lane += 1;
		}
		row.written = out as u8;
		table[widths] = row;
		widths += 1;
	}
	table
}

/// Mixed one-through-four-byte UTF-8 packer indexed by four 2-bit widths.
pub static UTF8_FOUR: [Utf8Four; 256] = build_utf8_four();
