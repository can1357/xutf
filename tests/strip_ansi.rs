use xutf::{IntoAnsiStripped, MakeAnsiStripped, ToAnsiStripped};

const CONTROL_CORPUS: &str = concat!(
	"plain ",
	"\x1b[31mred\x1b[0m ",
	"\u{9b}1mbold\u{9b}0m ",
	"\x1b]8;;https://example.com\x07link\x1b]8;;\x1b\\ ",
	"\u{9d}title\u{9c}visible ",
	"\x1bPdevice-data\x1b\\dcs ",
	"\u{90}device-data\u{9c}c1-dcs ",
	"\x1bXsos-data\x1b\\sos ",
	"\x1b^private\x1b\\pm ",
	"\x1b_application\x1b\\apc ",
	"\x1b(0q\x1b(B ",
	"\x1b[1m\x1b[4madjacent\x1b[0m ",
	"\x1b7saved",
);

const CONTROL_EXPECTED: &str = "plain red bold link visible dcs c1-dcs sos pm apc q adjacent saved";

fn stripped_utf16(input: &str) -> String {
	let mut units: Vec<u16> = input.encode_utf16().collect();
	let mut output = units.as_mut_slice();
	output.make_ansi_stripped();
	String::from_utf16(output).unwrap()
}

fn stripped_utf32(input: &str) -> String {
	let mut units: Vec<u32> = input.chars().map(u32::from).collect();
	let mut output = units.as_mut_slice();
	output.make_ansi_stripped();
	output
		.iter()
		.map(|&unit| char::from_u32(unit).unwrap())
		.collect()
}

#[test]
fn strips_owned_string_in_place() {
	let input = String::from(CONTROL_CORPUS);
	let pointer = input.as_ptr();
	let capacity = input.capacity();

	let stripped = input.into_ansi_stripped();
	assert_eq!(stripped, CONTROL_EXPECTED);
	assert_eq!(stripped.as_ptr(), pointer);
	assert_eq!(stripped.capacity(), capacity);
}

#[test]
fn strips_borrowed_string_into_one_allocation() {
	assert_eq!(CONTROL_CORPUS.to_ansi_stripped(), CONTROL_EXPECTED);
}

#[test]
fn utf8_utf16_and_utf32_slices_have_identical_results() {
	let mut utf8 = CONTROL_CORPUS.as_bytes().to_vec();
	let mut output = utf8.as_mut_slice();
	output.make_ansi_stripped();

	assert_eq!(output, CONTROL_EXPECTED.as_bytes());
	assert_eq!(stripped_utf16(CONTROL_CORPUS), CONTROL_EXPECTED);
	assert_eq!(stripped_utf32(CONTROL_CORPUS), CONTROL_EXPECTED);
}

#[test]
fn preserves_non_introducer_c1_and_multibyte_text() {
	assert_eq!("\x1b[31m© café \u{85} 漢字 🦀\x1b[0m".to_ansi_stripped(), "© café \u{85} 漢字 🦀");
}

#[test]
fn strips_unknown_escape_with_one_whole_codepoint() {
	let stripped = "before\x1b🦀after".to_ansi_stripped();
	assert_eq!(stripped, "beforeafter");
	assert!(stripped.is_char_boundary(stripped.len()));
}

#[test]
fn unterminated_sequences_consume_the_remainder() {
	for input in ["kept\x1b", "kept\x1b[31", "kept\x1b]title", "kept\u{90}payload"] {
		assert_eq!(input.to_ansi_stripped(), "kept");
	}
}

#[test]
fn clean_owned_input_keeps_its_allocation() {
	let input = String::from("plain © café \u{85} 漢字 🦀 text");
	let pointer = input.as_ptr();
	let capacity = input.capacity();

	let stripped = input.into_ansi_stripped();
	assert_eq!(stripped, "plain © café \u{85} 漢字 🦀 text");
	assert_eq!(stripped.as_ptr(), pointer);
	assert_eq!(stripped.capacity(), capacity);
}

#[test]
fn compacts_long_overlapping_spans() {
	let mut input = String::new();
	let mut expected = String::new();
	for index in 0..512 {
		let text = format!("segment-{index:04}-café-漢字-🦀;");
		input.push_str("\x1b[38;5;244m");
		input.push_str(&text);
		input.push_str("\x1b[0m");
		expected.push_str(&text);
	}
	let pointer = input.as_ptr();

	let stripped = input.into_ansi_stripped();
	assert_eq!(stripped, expected);
	assert_eq!(stripped.as_ptr(), pointer);
}

#[test]
fn borrowed_copy_handles_every_vector_boundary() {
	for length in [0, 1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 511, 512, 513] {
		let plain = "x".repeat(length);
		assert_eq!(plain.to_ansi_stripped(), plain);

		let suffix = "y".repeat(73);
		let input = format!("{plain}\x1b[38;5;244mred\x1b[0m{suffix}");
		let expected = format!("{plain}red{suffix}");
		assert_eq!(input.to_ansi_stripped(), expected);
	}
}

#[test]
fn empty_inputs_are_unchanged() {
	assert_eq!(String::new().into_ansi_stripped(), "");
	assert_eq!("".to_ansi_stripped(), "");

	let mut units: [u16; 0] = [];
	let mut output = units.as_mut_slice();
	output.make_ansi_stripped();
	assert_eq!(output, []);
}
