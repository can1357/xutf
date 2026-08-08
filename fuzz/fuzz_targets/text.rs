#![no_main]

mod support;

use libfuzzer_sys::fuzz_target;
use support::{u16_units, u32_units};
use unicode_segmentation::UnicodeSegmentation;
use xutf::{
	Encoding, Unit, Utf8, Utf16, Utf32, graphemes, graphemes_str, to_string, truncate, truncate_str,
	width, width_str, wrap, wrap_str,
};

fn ascii_unit<E: Encoding>(cp: u32) -> E::Unit {
	let mut encoded = vec![E::Unit::default(); E::MAX_UNITS];
	assert_eq!(E::encode(cp, &mut encoded), 1);
	encoded[0]
}

fn check_text<E: Encoding>(input: &[E::Unit], max_width: usize) {
	let cluster_limit = input.len().saturating_add(1);
	let iter = graphemes::<E>(input);
	let cloned = iter.clone();
	let clusters: Vec<_> = iter.take(cluster_limit).collect();
	let cloned_shapes: Vec<_> = cloned
		.take(cluster_limit)
		.map(|cluster| (cluster.units.len(), cluster.width))
		.collect();
	let shapes: Vec<_> = clusters
		.iter()
		.map(|cluster| (cluster.units.len(), cluster.width))
		.collect();
	assert_eq!(shapes, cloned_shapes);
	assert!(clusters.len() <= input.len());

	let mut consumed = 0;
	let mut measured = 0;
	for cluster in &clusters {
		assert!(!cluster.units.is_empty());
		assert!(cluster.units.len() <= input.len() - consumed);
		assert!(cluster.units == &input[consumed..consumed + cluster.units.len()]);
		consumed += cluster.units.len();
		measured += cluster.width;
	}
	assert_eq!(consumed, input.len());
	assert_eq!(measured, width::<E>(input));

	let truncated = truncate::<E>(input, max_width);
	assert!(truncated.len() <= input.len());
	assert!(truncated == &input[..truncated.len()]);
	let truncated_width = width::<E>(truncated);
	assert!(truncated_width <= max_width);

	let mut boundary = 0;
	let mut found_boundary = truncated.is_empty();
	for cluster in &clusters {
		boundary += cluster.units.len();
		found_boundary |= boundary == truncated.len();
	}
	assert!(found_boundary);
	if truncated.len() < input.len() {
		let next = graphemes::<E>(&input[truncated.len()..]).next().unwrap();
		assert!(truncated_width + next.width > max_width);
	}

	// Every line is a subslice of the input, so offsets recover the exact cut
	// points and let the checks below run against the input's own cluster
	// decomposition instead of re-segmenting a slice (whose trailing bytes
	// decode differently once the following units are gone).
	let offset =
		|line: &[E::Unit]| (line.as_ptr() as usize - input.as_ptr() as usize) / size_of::<E::Unit>();
	let mut boundaries = vec![false; input.len() + 1];
	let mut space_cluster_ends = vec![false; input.len() + 1];
	let space = ascii_unit::<E>(u32::from(b' '));
	let mut at = 0;
	boundaries[0] = true;
	for cluster in &clusters {
		at += cluster.units.len();
		boundaries[at] = true;
		space_cluster_ends[at] = cluster.units == [space];
	}

	let line_limit = input.len().saturating_add(2);
	let iter = wrap::<E>(input, max_width);
	let cloned = iter.clone();
	let lines: Vec<_> = iter.take(line_limit).collect();
	let cloned_lines: Vec<_> = cloned.take(line_limit).collect();
	assert!(lines == cloned_lines);
	assert!(!lines.is_empty());
	assert!(lines.len() <= input.len().saturating_add(1));

	// Only break separators — space runs and line terminators — may be dropped
	// between two lines or after the last one.
	let assert_separators = |range: &[E::Unit]| {
		for unit in range {
			let raw = if E::FOREIGN {
				unit.swap_bytes().to_u32()
			} else {
				unit.to_u32()
			};
			assert!(matches!(raw, 0x20 | 0x0a | 0x0d), "dropped a non-separator unit");
		}
	};

	let mut prev_end = 0;
	for line in &lines {
		let start = offset(line);
		let end = start + line.len();
		assert!(start >= prev_end && end <= input.len());
		assert!(boundaries[start] && boundaries[end], "line cut off a cluster boundary");
		assert_separators(&input[prev_end..start]);
		assert!(line.is_empty() || !space_cluster_ends[end], "line kept a trailing space cluster");
		let spanned = clusters
			.iter()
			.scan(0usize, |seen, cluster| {
				let cluster_start = *seen;
				*seen += cluster.units.len();
				Some(cluster_start)
			})
			.filter(|cluster_start| (start..end).contains(cluster_start))
			.count();
		assert!(
			width::<E>(line) <= max_width || spanned == 1,
			"an over-width line must be one indivisible cluster"
		);
		prev_end = end;
	}
	assert_separators(&input[prev_end..]);
}

fn check_str(input: &str, max_width: usize) {
	let iter = graphemes_str(input);
	let cloned = iter.clone();
	let clusters: Vec<_> = iter.collect();
	assert_eq!(cloned.collect::<Vec<_>>(), clusters);
	assert_eq!(clusters.concat(), input);
	let reference: Vec<_> = UnicodeSegmentation::graphemes(input, true).collect();
	assert_eq!(clusters, reference, "segmentation differed from unicode-segmentation");

	let generic_clusters: Vec<_> = graphemes::<Utf8>(input.as_bytes())
		.map(|cluster| core::str::from_utf8(cluster.units).unwrap())
		.collect();
	assert_eq!(generic_clusters, clusters);
	assert_eq!(width_str(input), width::<Utf8>(input.as_bytes()));
	assert_eq!(
		truncate_str(input, max_width).as_bytes(),
		truncate::<Utf8>(input.as_bytes(), max_width)
	);

	let lines: Vec<_> = wrap_str(input, max_width).collect();
	let generic_lines: Vec<_> = wrap::<Utf8>(input.as_bytes(), max_width)
		.map(|line| core::str::from_utf8(line).unwrap())
		.collect();
	assert_eq!(lines, generic_lines);
}

fn check_valid_encoding<E: Encoding>(input: &[E::Unit], text: &str, max_width: usize) {
	assert_eq!(width::<E>(input), width_str(text));

	let expected_clusters: Vec<_> = graphemes_str(text).map(str::to_owned).collect();
	let actual_clusters: Vec<_> = graphemes::<E>(input)
		.map(|cluster| to_string::<E>(cluster.units).unwrap())
		.collect();
	assert_eq!(actual_clusters, expected_clusters);

	let expected_truncation = truncate_str(text, max_width);
	assert_eq!(to_string::<E>(truncate::<E>(input, max_width)).unwrap(), expected_truncation);

	let expected_lines: Vec<_> = wrap_str(text, max_width).map(str::to_owned).collect();
	let actual_lines: Vec<_> = wrap::<E>(input, max_width)
		.map(|line| to_string::<E>(line).unwrap())
		.collect();
	assert_eq!(actual_lines, expected_lines);
}

fuzz_target!(|data: &[u8]| {
	let width_seed = u16::from_le_bytes([
		data.first().copied().unwrap_or_default(),
		data.get(1).copied().unwrap_or_default(),
	]);
	let max_width = usize::from(width_seed) % 65;
	let raw = data.get(2..).unwrap_or_default();
	let raw16 = u16_units(raw);
	let raw32 = u32_units(raw);

	check_text::<Utf8>(raw, max_width);
	check_text::<Utf16<false>>(&raw16, max_width);
	check_text::<Utf16<true>>(&raw16, max_width);
	check_text::<Utf32<false>>(&raw32, max_width);
	check_text::<Utf32<true>>(&raw32, max_width);

	let text = String::from_utf8_lossy(raw);
	check_str(&text, max_width);

	let utf16: Vec<_> = text.encode_utf16().collect();
	let utf16_foreign: Vec<_> = utf16.iter().map(|unit| unit.swap_bytes()).collect();
	let utf32: Vec<_> = text.chars().map(u32::from).collect();
	let utf32_foreign: Vec<_> = utf32.iter().map(|unit| unit.swap_bytes()).collect();
	check_valid_encoding::<Utf16<false>>(&utf16, &text, max_width);
	check_valid_encoding::<Utf16<true>>(&utf16_foreign, &text, max_width);
	check_valid_encoding::<Utf32<false>>(&utf32, &text, max_width);
	check_valid_encoding::<Utf32<true>>(&utf32_foreign, &text, max_width);
});
