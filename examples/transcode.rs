//! Demo: BOM-sniffing decode, transcoding, and UTF-aware comparison.
//!
//! Run: `cargo run --release --example transcode -- <file>`

use xutf::{
	Utf8, Utf16, Utf16Be, compare_ignore_ascii_case, from_bytes, transcode, transcoded_len,
};

fn main() {
	let path = std::env::args().nth(1).expect("usage: transcode <file>");
	let raw = std::fs::read(&path).expect("read file");

	// Decode whatever the BOM says (defaulting to UTF-8) into UTF-16.
	let utf16: Vec<u16> = from_bytes::<Utf16>(&raw);
	println!("{}: {} UTF-16 units", path, utf16.len());
	println!("as UTF-8: {} bytes", transcoded_len::<Utf16, Utf8>(&utf16));

	// Round-trip through big-endian UTF-16 and back.
	let be: Vec<u16> = transcode::<Utf16, Utf16Be>(&utf16);
	let round: Vec<u16> = transcode::<Utf16Be, Utf16>(&be);
	assert_eq!(utf16, round);

	// Case-insensitive comparison across encodings.
	let upper = xutf::transcode_with_case::<Utf16, Utf8>(&utf16, xutf::AsciiCase::Upper);
	assert_eq!(compare_ignore_ascii_case::<Utf8, Utf16>(&upper, &utf16), std::cmp::Ordering::Equal);
	println!("round-trip and case-insensitive self-compare OK");
}
