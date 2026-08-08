use xutf::{ToAnsiStripped, Utf16, width_ansi, width_ansi_str, width_str};

const CONTROL_CORPUS: &[&str] = &[
	"\x1b[31mred\x1b[0m plain",
	"\x1b]8;;https://example.com\x07link\x1b]8;;\x1b\\",
	"left\x1b[12Cright\u{9b}2Ddone",
	"\x1bPdevice-data\x1b\\dcs \x1bXsos-data\x1b\\sos \x1b^private\x1b\\pm \
	 \x1b_application\x1b\\apc",
	"\u{90}device-data\u{9c}c1-dcs \u{98}sos\u{9c} \u{9e}pm\u{9c} \u{9f}apc\u{9c}",
	"\x1b[1m\x1b[4madjacent\x1b[0m",
	"visible\x1b]unterminated tail",
	"escape-free ASCII text",
	"状态 café 漢字 🦀 👩‍🚀",
	"\x1b[36m状态\x1b[0m café \x1b[33m漢字 🦀 👩‍🚀\x1b[0m",
	"",
];

#[test]
fn matches_stripping_then_measuring() {
	for &input in CONTROL_CORPUS {
		assert_eq!(width_ansi_str(input), width_str(&input.to_ansi_stripped()), "input: {input:?}");
	}
}

#[test]
fn escape_free_input_matches_plain_width() {
	for input in ["", "The quick brown fox", "状态 café 漢字 🦀 👩‍🚀"] {
		assert_eq!(width_ansi_str(input), width_str(input));
	}
}

#[test]
fn utf8_c1_lead_false_positives_remain_visible_text() {
	for input in ["©", "\u{85}", "©\u{85} text"] {
		assert_eq!(width_ansi_str(input), width_str(input));
	}
	assert_eq!(width_ansi_str("\x1b[31m©\u{85} text\x1b[0m"), width_str("©\u{85} text"));
}

#[test]
fn unterminated_sequence_hides_the_remainder() {
	assert_eq!(width_ansi_str("visible\x1b]unterminated"), width_str("visible"));
	assert_eq!(width_ansi_str("visible\u{90}unterminated"), width_str("visible"));
}

#[test]
fn escape_sequences_are_grapheme_boundaries() {
	let input = "👩\x1b[0m\u{200d}🚀";
	assert_eq!(width_ansi_str(input), 4);
	assert_eq!(width_str(&input.to_ansi_stripped()), 2);
}

#[test]
fn utf16_matches_utf8() {
	for &input in CONTROL_CORPUS {
		let utf16: Vec<u16> = input.encode_utf16().collect();
		assert_eq!(width_ansi::<Utf16<false>>(&utf16), width_ansi_str(input), "input: {input:?}");
	}
}
