#![no_main]

mod support;

use core::cmp::Ordering;

use libfuzzer_sys::fuzz_target;
use support::{u16_units, u32_units};
use xutf::{
	AsciiCase, Bom, Encoding, Unit, Utf8, Utf16, Utf32, chars, codepoints, compare,
	compare_ignore_ascii_case, detect_bom, equals, equals_ignore_ascii_case, from_bytes, to_string,
	transcode, transcode_into, transcode_with_case, transcoded_len, unit_offset,
};

fn scalar(seed: u32) -> u32 {
	const SCALAR_COUNT: u32 = 0x11_0000 - 0x800;
	let cp = seed % SCALAR_COUNT;
	if cp >= 0xd800 { cp + 0x800 } else { cp }
}

fn fold(cp: u32, case: AsciiCase) -> u32 {
	match case {
		AsciiCase::Preserve => cp,
		AsciiCase::Lower if cp.wrapping_sub(u32::from(b'A')) <= 25 => cp ^ 0x20,
		AsciiCase::Upper if cp.wrapping_sub(u32::from(b'a')) <= 25 => cp ^ 0x20,
		AsciiCase::Lower | AsciiCase::Upper => cp,
	}
}

fn same_encoding<F: Encoding, T: Encoding>() -> bool {
	F::KIND == T::KIND && F::FOREIGN == T::FOREIGN
}

fn check_unit<U: Unit>(seed: u32) {
	let unit = U::from_u32(seed);
	assert!(U::from_u32(unit.to_u32()) == unit);
	assert!(unit.swap_bytes().swap_bytes() == unit);
}

fn check_scalar<E: Encoding>(cp: u32) {
	let expected = E::encoded_length(cp);
	assert!((1..=E::MAX_UNITS).contains(&expected));

	let mut encoded = vec![E::Unit::default(); E::MAX_UNITS];
	let written = E::encode(cp, &mut encoded);
	assert_eq!(written, expected);
	assert_eq!(E::run_length(encoded[0]), written);

	let mut input = &encoded[..written];
	assert_eq!(E::decode(&mut input), cp);
	assert!(input.is_empty());
}

fn check_iterators<E: Encoding>(input: &[E::Unit]) {
	let mut rest = input;
	let mut expected = Vec::new();
	while !rest.is_empty() {
		let before = rest.len();
		expected.push(E::decode(&mut rest));
		assert!(rest.len() < before);
	}

	let iter = codepoints::<E>(input);
	let cloned = iter.clone();
	assert_eq!(iter.collect::<Vec<_>>(), expected);
	assert_eq!(cloned.collect::<Vec<_>>(), expected);

	let expected_chars: Vec<_> = expected
		.iter()
		.map(|&cp| char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER))
		.collect();
	assert_eq!(chars::<E>(input).collect::<Vec<_>>(), expected_chars);
}

fn reference_transcode<F: Encoding, T: Encoding>(
	input: &[F::Unit],
	case: AsciiCase,
) -> Vec<T::Unit> {
	let mut output = Vec::with_capacity(input.len().saturating_mul(T::MAX_UNITS));
	if same_encoding::<F, T>() {
		let mut pos = 0;
		while pos < input.len() {
			let run = if case == AsciiCase::Preserve {
				1
			} else {
				F::run_length(input[pos]).min(input.len() - pos)
			};
			for offset in 0..run {
				let unit = input[pos + offset];
				let value = if offset == 0 {
					let native = if F::FOREIGN { unit.swap_bytes() } else { unit };
					let folded = F::Unit::from_u32(fold(native.to_u32(), case));
					if F::FOREIGN {
						folded.swap_bytes()
					} else {
						folded
					}
				} else {
					unit
				};
				output.push(T::Unit::from_u32(value.to_u32()));
			}
			pos += run;
		}
		return output;
	}

	let mut encoded = vec![T::Unit::default(); T::MAX_UNITS];
	for cp in codepoints::<F>(input) {
		let written = T::encode(fold(cp, case), &mut encoded);
		output.extend_from_slice(&encoded[..written]);
	}
	output
}

fn next_output_len<F: Encoding, T: Encoding>(input: &[F::Unit], case: AsciiCase) -> usize {
	if same_encoding::<F, T>() {
		return if case == AsciiCase::Preserve {
			1
		} else {
			F::run_length(input[0]).min(input.len())
		};
	}
	let mut rest = input;
	T::encoded_length(fold(F::decode(&mut rest), case))
}

fn check_transcode<F: Encoding, T: Encoding>(input: &[F::Unit], capacity: usize) {
	let expected = reference_transcode::<F, T>(input, AsciiCase::Preserve);
	assert_eq!(transcoded_len::<F, T>(input), expected.len());
	assert!(transcode::<F, T>(input) == expected);

	for case in [AsciiCase::Preserve, AsciiCase::Lower, AsciiCase::Upper] {
		let expected = reference_transcode::<F, T>(input, case);
		assert!(transcode_with_case::<F, T>(input, case) == expected);

		let mut exact = vec![T::Unit::default(); expected.len()];
		let (read, written) = transcode_into::<F, T>(input, &mut exact, case);
		assert_eq!(read, input.len());
		assert_eq!(written, expected.len());
		assert!(exact == expected);

		let limit = capacity % expected.len().saturating_add(1);
		let mut bounded = vec![T::Unit::default(); limit];
		let (read, written) = transcode_into::<F, T>(input, &mut bounded, case);
		assert!(read <= input.len());
		assert!(written <= bounded.len());

		let prefix = reference_transcode::<F, T>(&input[..read], case);
		assert_eq!(prefix.len(), written);
		assert!(bounded[..written] == prefix);
		if read < input.len() {
			assert!(next_output_len::<F, T>(&input[read..], case) > limit - written);
		}
	}
}

fn check_offset<F: Encoding, T: Encoding>(input: &[F::Unit], seed: usize) {
	// (codepoint boundary, output units before it)
	let mut stops = vec![(0, 0)];
	let mut rest = input;
	while !rest.is_empty() {
		let from = input.len() - rest.len();
		F::decode(&mut rest);
		let at = input.len() - rest.len();
		let units = reference_transcode::<F, T>(&input[from..at], AsciiCase::Preserve).len();
		stops.push((at, stops.last().unwrap().1 + units));
	}
	let n = seed % (stops.last().unwrap().1 + 2);
	let want = stops[stops.partition_point(|&(_, units)| units <= n) - 1].0;
	assert_eq!(unit_offset::<F, T>(input, n), want);
}

fn reference_compare<A: Encoding, B: Encoding>(
	a: &[A::Unit],
	b: &[B::Unit],
	caseless: bool,
) -> Ordering {
	let normalize = |cp| {
		if caseless {
			fold(cp, AsciiCase::Lower)
		} else {
			cp
		}
	};
	codepoints::<A>(a)
		.map(normalize)
		.cmp(codepoints::<B>(b).map(normalize))
}

fn check_compare<A: Encoding, B: Encoding>(a: &[A::Unit], b: &[B::Unit]) {
	let expected = reference_compare::<A, B>(a, b, false);
	assert_eq!(compare::<A, B>(a, b), expected);
	assert_eq!(equals::<A, B>(a, b), expected == Ordering::Equal);
	assert_eq!(compare::<B, A>(b, a), expected.reverse());

	let expected = reference_compare::<A, B>(a, b, true);
	assert_eq!(compare_ignore_ascii_case::<A, B>(a, b), expected);
	assert_eq!(equals_ignore_ascii_case::<A, B>(a, b), expected == Ordering::Equal);
	assert_eq!(compare_ignore_ascii_case::<B, A>(b, a), expected.reverse());

	assert_eq!(compare::<A, A>(a, a), Ordering::Equal);
	assert!(equals::<A, A>(a, a));
	assert!(equals_ignore_ascii_case::<A, A>(a, a));
}

fn check_to_string<E: Encoding>(input: &[E::Unit]) {
	let expected = String::from_utf8(reference_transcode::<E, Utf8>(input, AsciiCase::Preserve));
	match (to_string::<E>(input), expected) {
		(Ok(actual), Ok(expected)) => assert_eq!(actual, expected),
		(Err(_), Err(_)) => {},
		_ => panic!("to_string disagreed with its UTF-8 output"),
	}
}

fn reference_bom(data: &[u8]) -> (Bom, usize) {
	if data.starts_with(&[0xef, 0xbb, 0xbf]) {
		return (Bom::Utf8, 3);
	}
	if data.starts_with(&0xfeff_u32.to_ne_bytes()) {
		return (Bom::Utf32Native, 4);
	}
	if data.starts_with(&0xfeff_u32.swap_bytes().to_ne_bytes()) {
		return (Bom::Utf32Swapped, 4);
	}
	if data.starts_with(&0xfeff_u16.to_ne_bytes()) {
		return (Bom::Utf16Native, 2);
	}
	if data.starts_with(&0xfeff_u16.swap_bytes().to_ne_bytes()) {
		return (Bom::Utf16Swapped, 2);
	}
	(Bom::None, 0)
}

fn reference_from_bytes<T: Encoding>(data: &[u8]) -> Vec<T::Unit> {
	let (bom, skip) = reference_bom(data);
	let body = &data[skip..];
	match bom {
		Bom::None | Bom::Utf8 => reference_transcode::<Utf8, T>(body, AsciiCase::Preserve),
		Bom::Utf16Native => {
			reference_transcode::<Utf16<false>, T>(&u16_units(body), AsciiCase::Preserve)
		},
		Bom::Utf16Swapped => {
			reference_transcode::<Utf16<true>, T>(&u16_units(body), AsciiCase::Preserve)
		},
		Bom::Utf32Native => {
			reference_transcode::<Utf32<false>, T>(&u32_units(body), AsciiCase::Preserve)
		},
		Bom::Utf32Swapped => {
			reference_transcode::<Utf32<true>, T>(&u32_units(body), AsciiCase::Preserve)
		},
	}
}

fn check_from_bytes<T: Encoding>(data: &[u8]) {
	assert!(from_bytes::<T>(data) == reference_from_bytes::<T>(data));
}

fn check_bytes(data: &[u8]) {
	assert_eq!(detect_bom(data), reference_bom(data));
	check_from_bytes::<Utf8>(data);
	check_from_bytes::<Utf16<false>>(data);
	check_from_bytes::<Utf16<true>>(data);
	check_from_bytes::<Utf32<false>>(data);
	check_from_bytes::<Utf32<true>>(data);
}

fn check_pair<F: Encoding, T: Encoding>(a: &[F::Unit], b: &[T::Unit], capacity: usize, cp: u32) {
	check_scalar::<F>(cp);
	check_scalar::<T>(cp);
	check_iterators::<F>(a);
	check_iterators::<T>(b);
	check_transcode::<F, T>(a, capacity);
	check_offset::<F, T>(a, capacity);
	check_compare::<F, T>(a, b);
	check_to_string::<F>(a);
}

fuzz_target!(|data: &[u8]| {
	let pair = usize::from(data.first().copied().unwrap_or_default()) % 25;
	let capacity = usize::from(u16::from_le_bytes([
		data.get(1).copied().unwrap_or_default(),
		data.get(2).copied().unwrap_or_default(),
	]));
	let split_seed = usize::from(u16::from_le_bytes([
		data.get(3).copied().unwrap_or_default(),
		data.get(4).copied().unwrap_or_default(),
	]));
	let payload = data.get(5..).unwrap_or_default();
	let split = split_seed % payload.len().saturating_add(1);
	let (a8, b8) = payload.split_at(split);
	let a16 = u16_units(a8);
	let b16 = u16_units(b8);
	let a32 = u32_units(a8);
	let b32 = u32_units(b8);
	let seed = u32::from_le_bytes([
		payload.first().copied().unwrap_or_default(),
		payload.get(1).copied().unwrap_or_default(),
		payload.get(2).copied().unwrap_or_default(),
		payload.get(3).copied().unwrap_or_default(),
	]);
	let cp = scalar(seed);

	check_unit::<u8>(seed);
	check_unit::<u16>(seed);
	check_unit::<u32>(seed);
	check_bytes(payload);

	match pair {
		0 => check_pair::<Utf8, Utf8>(a8, b8, capacity, cp),
		1 => check_pair::<Utf8, Utf16<false>>(a8, &b16, capacity, cp),
		2 => check_pair::<Utf8, Utf16<true>>(a8, &b16, capacity, cp),
		3 => check_pair::<Utf8, Utf32<false>>(a8, &b32, capacity, cp),
		4 => check_pair::<Utf8, Utf32<true>>(a8, &b32, capacity, cp),
		5 => check_pair::<Utf16<false>, Utf8>(&a16, b8, capacity, cp),
		6 => check_pair::<Utf16<false>, Utf16<false>>(&a16, &b16, capacity, cp),
		7 => check_pair::<Utf16<false>, Utf16<true>>(&a16, &b16, capacity, cp),
		8 => check_pair::<Utf16<false>, Utf32<false>>(&a16, &b32, capacity, cp),
		9 => check_pair::<Utf16<false>, Utf32<true>>(&a16, &b32, capacity, cp),
		10 => check_pair::<Utf16<true>, Utf8>(&a16, b8, capacity, cp),
		11 => check_pair::<Utf16<true>, Utf16<false>>(&a16, &b16, capacity, cp),
		12 => check_pair::<Utf16<true>, Utf16<true>>(&a16, &b16, capacity, cp),
		13 => check_pair::<Utf16<true>, Utf32<false>>(&a16, &b32, capacity, cp),
		14 => check_pair::<Utf16<true>, Utf32<true>>(&a16, &b32, capacity, cp),
		15 => check_pair::<Utf32<false>, Utf8>(&a32, b8, capacity, cp),
		16 => check_pair::<Utf32<false>, Utf16<false>>(&a32, &b16, capacity, cp),
		17 => check_pair::<Utf32<false>, Utf16<true>>(&a32, &b16, capacity, cp),
		18 => check_pair::<Utf32<false>, Utf32<false>>(&a32, &b32, capacity, cp),
		19 => check_pair::<Utf32<false>, Utf32<true>>(&a32, &b32, capacity, cp),
		20 => check_pair::<Utf32<true>, Utf8>(&a32, b8, capacity, cp),
		21 => check_pair::<Utf32<true>, Utf16<false>>(&a32, &b16, capacity, cp),
		22 => check_pair::<Utf32<true>, Utf16<true>>(&a32, &b16, capacity, cp),
		23 => check_pair::<Utf32<true>, Utf32<false>>(&a32, &b32, capacity, cp),
		24 => check_pair::<Utf32<true>, Utf32<true>>(&a32, &b32, capacity, cp),
		_ => unreachable!(),
	}
});
