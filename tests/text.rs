//! The [`Text`] extension trait mirrors the free functions on every receiver.

use xutf::{
	Text, Utf8, Utf16, Utf32, codepoints, grapheme_indices, grapheme_indices_str, graphemes,
	graphemes_str, skip_columns, skip_columns_str, truncate, truncate_measured,
	truncate_measured_str, truncate_str, width, width_str, width_within, width_within_str, wrap,
	wrap_measured, wrap_measured_str, wrap_str,
};

#[test]
fn str_methods_match_free_functions() {
	let s = "cafe\u{301} \u{1f44d}\u{1f3fd} wrap sample \r\n tail";
	assert_eq!(s.visible_width(), width_str(s));
	assert_eq!(s.width_within(20), width_within_str(s, 20));
	assert_eq!(s.width_within(2), width_within_str(s, 2));
	assert_eq!(s.graphemes().collect::<Vec<_>>(), graphemes_str(s).collect::<Vec<_>>());
	assert_eq!(s.graphemes().len(), graphemes_str(s).count());
	assert_eq!(s.graphemes().next_back(), graphemes_str(s).last());
	assert_eq!(
		s.codepoints().collect::<Vec<_>>(),
		codepoints::<Utf8>(s.as_bytes()).collect::<Vec<_>>()
	);
	assert_eq!(s.truncate_width(7), truncate_str(s, 7));
	let truncated = s.truncate_measured(6);
	assert_eq!(truncated, truncate_measured_str(s, 6));
	assert_eq!(truncated.1, 5);
	let skipped = s.skip_columns(6);
	assert_eq!(skipped, skip_columns_str(s, 6));
	assert_eq!(skipped.1, 7);
	assert_eq!(s.wrap(8).collect::<Vec<_>>(), wrap_str(s, 8).collect::<Vec<_>>());
}

#[test]
fn unit_slice_methods_match_free_functions() {
	let s = "가나다 \u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467} x\r\n";
	let bytes = s.as_bytes();
	assert_eq!(bytes.visible_width(), width::<Utf8>(bytes));
	assert_eq!(bytes.width_within(20), width_within::<Utf8>(bytes, 20));
	assert_eq!(bytes.width_within(2), width_within::<Utf8>(bytes, 2));
	assert_eq!(
		bytes.graphemes().map(|g| g.units).collect::<Vec<_>>(),
		graphemes::<Utf8>(bytes)
			.map(|g| g.units)
			.collect::<Vec<_>>()
	);
	assert_eq!(bytes.truncate_width(4), truncate::<Utf8>(bytes, 4));
	let truncated = bytes.truncate_measured(5);
	assert_eq!(truncated, truncate_measured::<Utf8>(bytes, 5));
	assert_eq!(truncated.1, 4);
	let skipped = bytes.skip_columns(5);
	assert_eq!(skipped, skip_columns::<Utf8>(bytes, 5));
	assert_eq!(skipped.1, 6);
	assert_eq!(bytes.wrap(5).collect::<Vec<_>>(), wrap::<Utf8>(bytes, 5).collect::<Vec<_>>());

	let units16: Vec<u16> = s.encode_utf16().collect();
	assert_eq!(units16.visible_width(), width::<Utf16<false>>(&units16));
	assert_eq!(units16.width_within(20), width_within::<Utf16<false>>(&units16, 20));
	assert_eq!(units16.width_within(2), width_within::<Utf16<false>>(&units16, 2));
	assert_eq!(units16.graphemes().len(), graphemes::<Utf16<false>>(&units16).count());
	assert_eq!(
		units16.codepoints().collect::<Vec<_>>(),
		codepoints::<Utf16<false>>(&units16).collect::<Vec<_>>()
	);
	let truncated = units16.truncate_measured(5);
	assert_eq!(truncated, truncate_measured::<Utf16<false>>(&units16, 5));
	assert_eq!(truncated.1, 4);
	let skipped = units16.skip_columns(5);
	assert_eq!(skipped, skip_columns::<Utf16<false>>(&units16, 5));
	assert_eq!(skipped.1, 6);

	let units32: Vec<u32> = s.chars().map(u32::from).collect();
	assert_eq!(units32.visible_width(), width::<Utf32<false>>(&units32));
	assert_eq!(units32.width_within(20), width_within::<Utf32<false>>(&units32, 20));
	assert_eq!(units32.width_within(2), width_within::<Utf32<false>>(&units32, 2));
	assert_eq!(units32.truncate_width(3), truncate::<Utf32<false>>(&units32, 3));
	let truncated = units32.truncate_measured(5);
	assert_eq!(truncated, truncate_measured::<Utf32<false>>(&units32, 5));
	assert_eq!(truncated.1, 4);
	let skipped = units32.skip_columns(5);
	assert_eq!(skipped, skip_columns::<Utf32<false>>(&units32, 5));
	assert_eq!(skipped.1, 6);
}

#[test]
fn methods_resolve_through_owned_receivers() {
	let owned = String::from("h\u{e9}llo");
	assert_eq!(owned.visible_width(), 5);
	assert_eq!(owned.graphemes().len(), 5);
	assert_eq!(owned.truncate_width(2), "h\u{e9}");
	assert_eq!(owned.truncate_measured(2), ("h\u{e9}", 2));
	assert_eq!(owned.skip_columns(2), ("llo", 2));

	let vec16: Vec<u16> = owned.encode_utf16().collect();
	assert_eq!(vec16.visible_width(), 5);
	assert_eq!(vec16.truncate_measured(2), (&vec16[..2], 2));
	assert_eq!(vec16.skip_columns(2), (&vec16[2..], 2));
}

#[test]
fn visible_width_has_the_expected_name_and_result() {
	assert_eq!("abc".visible_width(), 3);
}

#[test]
fn wrap_measured_methods_match_free_functions_and_clone() {
	let s = "가나다 cafe\u{301} \u{1f44d}\u{1f3fd}\r\n tail";
	let lines = s.wrap_measured(6);
	let cloned = lines.clone();
	assert_eq!(
		lines
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>(),
		wrap_measured_str(s, 6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>()
	);
	assert_eq!(
		cloned
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>(),
		wrap_measured_str(s, 6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>()
	);

	let bytes = s.as_bytes();
	assert_eq!(
		bytes
			.wrap_measured(6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>(),
		wrap_measured::<Utf8>(bytes, 6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>()
	);

	let units16 = s.encode_utf16().collect::<Vec<_>>();
	assert_eq!(
		units16
			.wrap_measured(6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>(),
		wrap_measured::<Utf16<false>>(&units16, 6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>()
	);

	let units32 = s.chars().map(u32::from).collect::<Vec<_>>();
	assert_eq!(
		units32
			.wrap_measured(6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>(),
		wrap_measured::<Utf32<false>>(&units32, 6)
			.map(|line| (line.units, line.at, line.width))
			.collect::<Vec<_>>()
	);
}
#[test]
fn grapheme_indices_report_native_offsets_and_support_mixed_iteration() {
	let s = "aé\u{301}\u{1f44d}\u{1f3fd}z";
	let mut indices = s.grapheme_indices();
	assert_eq!(indices.len(), 4);
	assert_eq!(indices.clone().collect::<Vec<_>>(), grapheme_indices_str(s).collect::<Vec<_>>());
	assert_eq!(indices.next(), Some((0, "a")));
	assert_eq!(indices.next_back(), Some((13, "z")));
	assert_eq!(indices.next(), Some((1, "é\u{301}")));
	assert_eq!(indices.next_back(), Some((5, "\u{1f44d}\u{1f3fd}")));
	assert_eq!(indices.next(), None);

	let bytes = s.as_bytes();
	assert_eq!(
		bytes
			.grapheme_indices()
			.map(|(offset, cluster)| (offset, cluster.units))
			.collect::<Vec<_>>(),
		grapheme_indices::<Utf8>(bytes)
			.map(|(offset, cluster)| (offset, cluster.units))
			.collect::<Vec<_>>()
	);

	let units16: Vec<u16> = s.encode_utf16().collect();
	let indexed16 = units16
		.grapheme_indices()
		.map(|(offset, cluster)| (offset, cluster.units))
		.collect::<Vec<_>>();
	assert_eq!(
		indexed16
			.iter()
			.map(|(offset, _)| *offset)
			.collect::<Vec<_>>(),
		[0, 1, 3, 7]
	);
	assert_eq!(indexed16[1].1, &units16[1..3]);
	assert_eq!(indexed16[2].1, &units16[3..7]);

	let units32: Vec<u32> = s.chars().map(u32::from).collect();
	let indexed32 = units32
		.grapheme_indices()
		.map(|(offset, cluster)| (offset, cluster.units))
		.collect::<Vec<_>>();
	assert_eq!(
		indexed32
			.iter()
			.map(|(offset, _)| *offset)
			.collect::<Vec<_>>(),
		[0, 1, 3, 5]
	);
	assert_eq!(indexed32[2].1, &units32[3..5]);
}
