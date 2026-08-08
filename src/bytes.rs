//! Byte-order-mark detection and raw-byte decoding.

use alloc::vec::Vec;

use crate::{convert::transcode, encoding::Encoding, utf8::Utf8, utf16::Utf16, utf32::Utf32};

/// Byte order mark at the head of a raw byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bom {
	/// No BOM; the stream is treated as UTF-8.
	None,
	/// `EF BB BF`.
	Utf8,
	/// `FEFF` in native byte order.
	Utf16Native,
	/// `FEFF` byte-swapped.
	Utf16Swapped,
	/// `0000FEFF` in native byte order.
	Utf32Native,
	/// `0000FEFF` byte-swapped.
	Utf32Swapped,
}

/// Detects a BOM, returning it plus the byte length to skip.
///
/// UTF-32 is probed before UTF-16 since a native-order UTF-32 BOM contains a
/// valid UTF-16 BOM prefix.
pub fn detect_bom(data: &[u8]) -> (Bom, usize) {
	if data.len() >= 3 && data[..3] == [0xef, 0xbb, 0xbf] {
		return (Bom::Utf8, 3);
	}
	if data.len() >= 4 {
		let w = u32::from_ne_bytes(data[..4].try_into().unwrap());
		if w == 0xfeff {
			return (Bom::Utf32Native, 4);
		}
		if w == 0xfeff_u32.swap_bytes() {
			return (Bom::Utf32Swapped, 4);
		}
	}
	if data.len() >= 2 {
		let w = u16::from_ne_bytes(data[..2].try_into().unwrap());
		if w == 0xfeff {
			return (Bom::Utf16Native, 2);
		}
		if w == 0xfeff_u16.swap_bytes() {
			return (Bom::Utf16Swapped, 2);
		}
	}
	(Bom::None, 0)
}

/// Gathers the (possibly unaligned) byte stream into native-order units,
/// dropping trailing partial units.
fn units<const W: usize, U>(data: &[u8], read: impl Fn([u8; W]) -> U) -> Vec<U> {
	data.as_chunks::<W>().0.iter().map(|c| read(*c)).collect()
}

/// Identifies the encoding of a raw byte stream from its BOM (defaulting to
/// UTF-8) and transcodes it to `T`.
pub fn from_bytes<T: Encoding>(data: &[u8]) -> Vec<T::Unit> {
	let (bom, skip) = detect_bom(data);
	let body = &data[skip..];
	match bom {
		Bom::None | Bom::Utf8 => transcode::<Utf8, T>(body),
		Bom::Utf16Native => transcode::<Utf16<false>, T>(&units(body, u16::from_ne_bytes)),
		Bom::Utf16Swapped => transcode::<Utf16<true>, T>(&units(body, u16::from_ne_bytes)),
		Bom::Utf32Native => transcode::<Utf32<false>, T>(&units(body, u32::from_ne_bytes)),
		Bom::Utf32Swapped => transcode::<Utf32<true>, T>(&units(body, u32::from_ne_bytes)),
	}
}
