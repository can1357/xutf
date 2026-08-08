//! Demo: BOM-sniffing decode, transcoding, and UTF-aware comparison.
//!
//! Run: `cargo run --release --example transcode -- <file>`

use xutf::{Encoding, Text, Utf8, Utf16, Utf16Be, transcode};

fn main() {
	let path = std::env::args().nth(1).expect("usage: transcode <file>");
	let raw = std::fs::read(&path).expect("read file");

	// Decode whatever the BOM says (defaulting to UTF-8) into UTF-16.
	let utf16: Vec<u16> = <Utf16>::from_bytes(&raw);
	println!(
		"{}: {} UTF-16 units (visible width {}, {} graphemes)",
		path,
		utf16.len(),
		utf16.visible_width(),
		utf16.graphemes().count()
	);
	println!("as UTF-8: {} bytes", utf16.transcoded_len::<u8>());

	// Round-trip through big-endian UTF-16 and back.
	let be: Vec<u16> = transcode::<Utf16, Utf16Be>(&utf16);
	let round: Vec<u16> = transcode::<Utf16Be, Utf16>(&be);
	assert_eq!(utf16, round);

	// Case-insensitive comparison across encodings.
	let upper = xutf::transcode_with_case::<Utf16, Utf8>(&utf16, xutf::AsciiCase::Upper);
	assert!(upper.eq_text_ignore_ascii_case(&utf16));
	println!("round-trip and case-insensitive self-compare OK");
}
