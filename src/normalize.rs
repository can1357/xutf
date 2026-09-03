//! Unicode 17 NFC/NFD/NFKC/NFKD normalization with a SIMD quick-check fast
//! path.
//!
//! Quick-check-positive strings return from the in-place API without writes. A
//! 64-lane UTF-8 scan skips ASCII, while compact generated tables drive
//! canonical and compatibility decomposition, ordering, and composition for
//! non-ASCII text. Normalization uses the string's own allocation and no
//! auxiliary allocation.

use alloc::{string::String, vec::Vec};
use core::{
	fmt, ptr,
	simd::{Simd, cmp::SimdPartialOrd},
};

#[path = "normalize_data.rs"]
mod data;

const CCC_MASK: u64 = 0xff;
const NFD_NO_BIT: u64 = 1 << 8;
const NFC_MAYBE_BIT: u64 = 1 << 9;
const NFC_NO_BIT: u64 = 1 << 10;
const DECOMPOSITION_SHIFT: u32 = 11;
const DECOMPOSITION_MASK: u64 = 0x07ff;
const DECOMPOSITION_ORDER_BIT: u64 = 1 << 22;
const COMPOSITION_SHIFT: u32 = 23;
const COMPOSITION_MASK: u64 = 0x01ff;
const NFKD_NO_BIT: u64 = 1 << 32;
const NFKC_MAYBE_BIT: u64 = 1 << 33;
const NFKC_NO_BIT: u64 = 1 << 34;
const NFKD_DECOMPOSITION_SHIFT: u32 = 35;
const NFKD_DECOMPOSITION_MASK: u64 = 0x0fff;
const NFKD_DECOMPOSITION_ORDER_BIT: u64 = 1 << 47;
const ORDER_STACK: usize = 16;
const CODEPOINT_MASK: u64 = (1 << 21) - 1;

const S_BASE: u32 = 0xac00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11a7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

const _: () = assert!(
	data::NORMALIZATION_UNICODE_VERSION.0 == crate::UNICODE_VERSION.0
		&& data::NORMALIZATION_UNICODE_VERSION.1 == crate::UNICODE_VERSION.1
		&& data::NORMALIZATION_UNICODE_VERSION.2 == crate::UNICODE_VERSION.2
);

/// Reports the workspace capacity needed for allocation-free normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizationError {
	required_capacity: usize,
}

impl NormalizationError {
	/// Minimum UTF-8 byte capacity required by the in-place operation.
	#[inline(always)]
	pub const fn required_capacity(self) -> usize {
		self.required_capacity
	}
}

impl fmt::Display for NormalizationError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			formatter,
			"normalization requires capacity for {} UTF-8 bytes",
			self.required_capacity
		)
	}
}

/// Normalizes a string in place without allocating.
///
/// Reserve the capacity reported by [`NormalizationError`] and retry when an
/// expanding decomposition does not fit.
pub trait MakeUnicodeNormalized {
	/// Converts this string to NFC using only its existing allocation.
	fn make_nfc(&mut self) -> Result<(), NormalizationError>;

	/// Converts this string to NFD using only its existing allocation.
	fn make_nfd(&mut self) -> Result<(), NormalizationError>;

	/// Converts this string to NFKC using only its existing allocation.
	fn make_nfkc(&mut self) -> Result<(), NormalizationError>;

	/// Converts this string to NFKD using only its existing allocation.
	fn make_nfkd(&mut self) -> Result<(), NormalizationError>;
}

/// Creates owned NFC, NFD, NFKC or NFKD text from a borrowed string.
pub trait ToUnicodeNormalized {
	/// Copies this string once and returns its NFC form.
	fn to_nfc(&self) -> String;

	/// Copies this string once and returns its NFD form.
	fn to_nfd(&self) -> String;

	/// Copies this string once and returns its NFKC form.
	fn to_nfkc(&self) -> String;

	/// Copies this string once and returns its NFKD form.
	fn to_nfkd(&self) -> String;
}

/// Converts an owned string to NFC, NFD, NFKC or NFKD while reusing its
/// allocation.
pub trait IntoUnicodeNormalized {
	/// Returns this string in NFC, growing its allocation only when required.
	fn into_nfc(self) -> String;

	/// Returns this string in NFD, growing its allocation only when required.
	fn into_nfd(self) -> String;

	/// Returns this string in NFKC, growing its allocation only when required.
	fn into_nfkc(self) -> String;

	/// Returns this string in NFKD, growing its allocation only when required.
	fn into_nfkd(self) -> String;
}

/// Reports whether `text` is definitely in NFC form.
///
/// Quick-check semantics without allocation: `true` means normalizing is a
/// no-op; `false` means normalization *may* change the text (`NFC_QC=Maybe`
/// codepoints and mis-ordered combining marks report `false` without
/// composing). Use before [`ToUnicodeNormalized::to_nfc`] to keep borrowed
/// fast paths.
#[inline]
pub fn is_nfc(text: &str) -> bool {
	scan(text, Form::Nfc).normalized
}

/// [`is_nfc`] over raw codepoints, for callers holding non-UTF-8 text
/// (UTF-16/UTF-32 units) who want the quick-check without transcoding.
///
/// Surrogate and out-of-range values are treated as inert (they normalize
/// to themselves under permissive decoding). Identical verdicts to
/// [`is_nfc`] for any sequence of Unicode scalar values.
pub fn is_nfc_codepoints(codepoints: impl IntoIterator<Item = u32>) -> bool {
	let mut last_ccc = 0u8;
	for cp in codepoints {
		if cp > 0x10ffff {
			last_ccc = 0;
			continue;
		}
		let word = normalization_word(cp);
		let ccc = combining_class(word);
		if (ccc != 0 && last_ccc > ccc) || word & (NFC_NO_BIT | NFC_MAYBE_BIT) != 0 {
			return false;
		}
		last_ccc = ccc;
	}
	true
}

/// Reports whether `text` is definitely in NFKC form.
///
/// Quick-check semantics without allocation: `true` means normalizing is a
/// no-op; `false` means normalization *may* change the text (`NFKC_QC=Maybe`
/// codepoints and mis-ordered combining marks report `false` without
/// composing). Use before [`ToUnicodeNormalized::to_nfkc`] to keep borrowed
/// fast paths.
#[inline]
pub fn is_nfkc(text: &str) -> bool {
	scan(text, Form::Nfkc).normalized
}

/// [`is_nfkc`] over raw codepoints, for callers holding non-UTF-8 text
/// (UTF-16/UTF-32 units) who want the quick-check without transcoding.
///
/// Surrogate and out-of-range values are treated as inert (they normalize
/// to themselves under permissive decoding). Identical verdicts to
/// [`is_nfkc`] for any sequence of Unicode scalar values.
pub fn is_nfkc_codepoints(codepoints: impl IntoIterator<Item = u32>) -> bool {
	let mut last_ccc = 0u8;
	for cp in codepoints {
		if cp > 0x10ffff {
			last_ccc = 0;
			continue;
		}
		let word = normalization_word(cp);
		let ccc = combining_class(word);
		if (ccc != 0 && last_ccc > ccc) || word & (NFKC_NO_BIT | NFKC_MAYBE_BIT) != 0 {
			return false;
		}
		last_ccc = ccc;
	}
	true
}

/// Canonical Combining Class (ccc) of a codepoint; 0 for starters,
/// out-of-range input and unassigned codepoints.
#[inline]
pub fn canonical_combining_class(cp: u32) -> u8 {
	if cp > 0x10ffff {
		return 0;
	}
	combining_class(normalization_word(cp))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
	Nfc,
	Nfd,
	Nfkc,
	Nfkd,
}

impl Form {
	/// Whether this form ends with canonical composition (NFC/NFKC).
	const fn is_composed(self) -> bool {
		matches!(self, Self::Nfc | Self::Nfkc)
	}

	/// Whether this form decomposes compatibility characters (NFKC/NFKD).
	const fn is_compat(self) -> bool {
		matches!(self, Self::Nfkc | Self::Nfkd)
	}
}

#[derive(Clone, Copy)]
struct Scan {
	normalized:  bool,
	nfd_len:     usize,
	decompose:   bool,
	reorder:     bool,
	shrinks:     bool,
	stack_order: bool,
}

#[inline(always)]
fn normalization_word(cp: u32) -> u64 {
	debug_assert!(cp <= 0x10ffff);
	let cp = cp as usize;
	let block = data::NORMALIZATION_STAGE1[cp >> data::NORMALIZATION_SHIFT] as usize;
	data::NORMALIZATION_STAGE2
		[(block << data::NORMALIZATION_SHIFT) | (cp & data::NORMALIZATION_MASK)]
}

#[inline(always)]
const fn combining_class(word: u64) -> u8 {
	(word & CCC_MASK) as u8
}

#[inline(always)]
fn decomposition(word: u64) -> Option<&'static [u32]> {
	let index = (word >> DECOMPOSITION_SHIFT) & DECOMPOSITION_MASK;
	if index == 0 {
		return None;
	}
	let record = data::DECOMPOSITION_RECORDS[index as usize] as usize;
	let start = record >> 3;
	let len = record & 7;
	Some(&data::DECOMPOSITION_CHARS[start..start + len])
}

#[inline(always)]
fn nfkd_decomposition(word: u64) -> Option<&'static [u32]> {
	let index = (word >> NFKD_DECOMPOSITION_SHIFT) & NFKD_DECOMPOSITION_MASK;
	if index == 0 {
		return None;
	}
	let record = data::NFKD_DECOMPOSITION_RECORDS[index as usize] as usize;
	let start = record >> 5;
	let len = record & 31;
	Some(&data::NFKD_DECOMPOSITION_CHARS[start..start + len])
}

#[inline(always)]
const fn is_hangul_syllable(cp: u32) -> bool {
	cp.wrapping_sub(S_BASE) < S_COUNT
}

#[inline(always)]
fn for_each_decomposed(cp: u32, word: u64, compat: bool, mut emit: impl FnMut(u32)) {
	if is_hangul_syllable(cp) {
		let index = cp - S_BASE;
		emit(L_BASE + index / N_COUNT);
		emit(V_BASE + index % N_COUNT / T_COUNT);
		let trailing = index % T_COUNT;
		if trailing != 0 {
			emit(T_BASE + trailing);
		}
	} else if let Some(mapped) = normalization_decomposition(word, compat) {
		for &part in mapped {
			emit(part);
		}
	} else {
		emit(cp);
	}
}

/// Canonical decomposition for NFC/NFD, full compatibility decomposition for
/// NFKC/NFKD.
#[inline(always)]
fn normalization_decomposition(word: u64, compat: bool) -> Option<&'static [u32]> {
	if compat {
		nfkd_decomposition(word)
	} else {
		decomposition(word)
	}
}

#[inline(always)]
const fn utf8_len(cp: u32) -> usize {
	if cp < 0x80 {
		1
	} else if cp < 0x800 {
		2
	} else if cp < 0x10000 {
		3
	} else {
		4
	}
}

#[inline(always)]
fn decomposed_utf8_len(cp: u32, word: u64, compat: bool) -> usize {
	if !is_hangul_syllable(cp) && normalization_decomposition(word, compat).is_none() {
		return utf8_len(cp);
	}
	let mut len = 0;
	for_each_decomposed(cp, word, compat, |part| len += utf8_len(part));
	len
}

#[inline(always)]
fn decode_utf8(input: &[u8], at: usize) -> (u32, usize) {
	let first = input[at];
	if first < 0x80 {
		return (u32::from(first), 1);
	}
	if first < 0xe0 {
		return ((u32::from(first & 0x1f) << 6) | u32::from(input[at + 1] & 0x3f), 2);
	}
	if first < 0xf0 {
		return (
			(u32::from(first & 0x0f) << 12)
				| (u32::from(input[at + 1] & 0x3f) << 6)
				| u32::from(input[at + 2] & 0x3f),
			3,
		);
	}
	(
		(u32::from(first & 7) << 18)
			| (u32::from(input[at + 1] & 0x3f) << 12)
			| (u32::from(input[at + 2] & 0x3f) << 6)
			| u32::from(input[at + 3] & 0x3f),
		4,
	)
}

#[inline(always)]
fn encode_utf8(cp: u32, output: &mut [u8]) -> usize {
	let len = utf8_len(cp);
	match len {
		1 => output[0] = cp as u8,
		2 => {
			output[0] = 0xc0 | (cp >> 6) as u8;
			output[1] = 0x80 | (cp & 0x3f) as u8;
		},
		3 => {
			output[0] = 0xe0 | (cp >> 12) as u8;
			output[1] = 0x80 | ((cp >> 6) & 0x3f) as u8;
			output[2] = 0x80 | (cp & 0x3f) as u8;
		},
		4 => {
			output[0] = 0xf0 | (cp >> 18) as u8;
			output[1] = 0x80 | ((cp >> 12) & 0x3f) as u8;
			output[2] = 0x80 | ((cp >> 6) & 0x3f) as u8;
			output[3] = 0x80 | (cp & 0x3f) as u8;
		},
		_ => unreachable!(),
	}
	len
}

/// Writes one valid scalar without constructing a slice beyond the vector's
/// initialized length.
///
/// # Safety
/// `output` must have room for [`utf8_len`] bytes.
#[inline(always)]
unsafe fn encode_utf8_ptr(cp: u32, output: *mut u8) -> usize {
	let mut encoded = [0; 4];
	let len = encode_utf8(cp, &mut encoded);
	// SAFETY: the caller provides `len` writable bytes and `encoded` contains
	// exactly that many initialized source bytes.
	unsafe { ptr::copy_nonoverlapping(encoded.as_ptr(), output, len) };
	len
}

#[inline(always)]
fn write_decomposition(cp: u32, word: u64, compat: bool, output: &mut [u8]) -> usize {
	let mut written = 0;
	for_each_decomposed(cp, word, compat, |part| {
		written += encode_utf8(part, &mut output[written..]);
	});
	written
}

/// Writes a complete decomposition into spare vector capacity.
///
/// # Safety
/// `output` must have room for [`decomposed_utf8_len`] bytes.
#[inline(always)]
unsafe fn write_decomposition_ptr(cp: u32, word: u64, compat: bool, output: *mut u8) -> usize {
	let mut written = 0;
	for_each_decomposed(cp, word, compat, |part| {
		// SAFETY: the caller reserves the full decomposition and each prior write
		// advances within that region.
		written += unsafe { encode_utf8_ptr(part, output.add(written)) };
	});
	written
}

#[inline(always)]
fn non_ascii_mask(chunk: Simd<u8, 64>) -> u64 {
	chunk.simd_gt(Simd::splat(0x7f)).to_bitmask()
}

#[inline(always)]
fn non_ascii_mask_narrow(chunk: Simd<u8, 16>) -> u64 {
	chunk.simd_gt(Simd::splat(0x7f)).to_bitmask()
}

#[inline]
fn ascii_prefix(input: &[u8]) -> usize {
	if input.is_empty() || input[0] >= 0x80 {
		return 0;
	}
	let mut at = if input.len() >= 16 {
		let mask = non_ascii_mask_narrow(Simd::from_slice(input));
		if mask != 0 {
			return mask.trailing_zeros() as usize;
		}
		16
	} else {
		0
	};
	while input.len() - at >= 256 {
		let first = non_ascii_mask(Simd::from_slice(&input[at..]));
		if first != 0 {
			return at + first.trailing_zeros() as usize;
		}
		let second = non_ascii_mask(Simd::from_slice(&input[at + 64..]));
		if second != 0 {
			return at + 64 + second.trailing_zeros() as usize;
		}
		let third = non_ascii_mask(Simd::from_slice(&input[at + 128..]));
		if third != 0 {
			return at + 128 + third.trailing_zeros() as usize;
		}
		let fourth = non_ascii_mask(Simd::from_slice(&input[at + 192..]));
		if fourth != 0 {
			return at + 192 + fourth.trailing_zeros() as usize;
		}
		at += 256;
	}
	while input.len() - at >= 64 {
		let mask = non_ascii_mask(Simd::from_slice(&input[at..]));
		if mask != 0 {
			return at + mask.trailing_zeros() as usize;
		}
		at += 64;
	}
	while input.len() - at >= 16 {
		let mask = non_ascii_mask_narrow(Simd::from_slice(&input[at..]));
		if mask != 0 {
			return at + mask.trailing_zeros() as usize;
		}
		at += 16;
	}
	while at < input.len() && input[at] < 0x80 {
		at += 1;
	}
	at
}

#[inline(always)]
fn ascii_suffix(input: &[u8]) -> usize {
	if input.is_empty() || input[input.len() - 1] >= 0x80 {
		return 0;
	}
	if input.len() >= 64 {
		let mask = non_ascii_mask(Simd::from_slice(&input[input.len() - 64..]));
		return mask.leading_zeros() as usize;
	}
	input.iter().rev().take_while(|&&byte| byte < 0x80).count()
}

#[inline(always)]
const fn compose_hangul(first: u32, second: u32) -> Option<u32> {
	if first.wrapping_sub(L_BASE) < L_COUNT && second.wrapping_sub(V_BASE) < V_COUNT {
		let leading = first - L_BASE;
		let vowel = second - V_BASE;
		return Some(S_BASE + leading * N_COUNT + vowel * T_COUNT);
	}
	if first.wrapping_sub(S_BASE) < S_COUNT
		&& (first - S_BASE).is_multiple_of(T_COUNT)
		&& second.wrapping_sub(T_BASE + 1) < T_COUNT - 1
	{
		return Some(first + second - T_BASE);
	}
	None
}

#[inline]
fn compose_pair(first: u32, second: u32) -> Option<u32> {
	if let Some(composite) = compose_hangul(first, second) {
		return Some(composite);
	}
	let group = ((normalization_word(first) >> COMPOSITION_SHIFT) & COMPOSITION_MASK) as usize;
	if group == 0 {
		return None;
	}
	let pairs = &data::COMPOSITION_PAIRS
		[data::COMPOSITION_OFFSETS[group] as usize..data::COMPOSITION_OFFSETS[group + 1] as usize];
	if pairs.len() <= 8 {
		for &packed in pairs {
			let candidate = (packed >> 21) as u32;
			if candidate == second {
				return Some((packed & CODEPOINT_MASK) as u32);
			}
			if candidate > second {
				return None;
			}
		}
		return None;
	}
	pairs
		.binary_search_by_key(&second, |packed| (*packed >> 21) as u32)
		.ok()
		.map(|index| (pairs[index] & CODEPOINT_MASK) as u32)
}

#[inline]
fn decomposition_workspace(bytes: &[u8], compat: bool) -> (usize, bool) {
	let no_bit = if compat { NFKD_NO_BIT } else { NFD_NO_BIT };
	let mut at = 0;
	let mut required = 0;
	let mut shrinks = false;
	while at < bytes.len() {
		let ascii = ascii_prefix(&bytes[at..]);
		if ascii != 0 {
			required += ascii;
			at += ascii;
			if at == bytes.len() {
				break;
			}
		}
		let (cp, width) = decode_utf8(bytes, at);
		let word = normalization_word(cp);
		let decomposed_len = if word & no_bit == 0 {
			width
		} else {
			decomposed_utf8_len(cp, word, compat)
		};
		required += decomposed_len;
		shrinks |= decomposed_len < width;
		at += width;
	}
	(required, shrinks)
}

#[inline]
fn scan(input: &str, form: Form) -> Scan {
	let bytes = input.as_bytes();
	let mut at = 0;
	let mut nfd_len = 0;
	let mut normalized = true;
	let mut decompose = false;
	let mut reorder = false;
	let mut shrinks = false;
	let mut segment_len = 0;
	let mut stack_order = true;
	let mut starter_risky = false;
	let mut last_ccc = 0;

	while at < bytes.len() {
		let ascii = ascii_prefix(&bytes[at..]);
		if ascii != 0 {
			if !form.is_composed() {
				nfd_len += ascii;
			}
			at += ascii;
			starter_risky = false;
			last_ccc = 0;
			segment_len = 1;
			if at == bytes.len() {
				break;
			}
		}

		let (cp, width) = decode_utf8(bytes, at);
		let word = normalization_word(cp);
		let ccc = combining_class(word);
		if ccc == 0 {
			segment_len = 1;
		} else {
			segment_len += 1;
			stack_order &= segment_len <= ORDER_STACK;
		}
		if !form.is_composed() {
			let no_bit = if form.is_compat() {
				NFKD_NO_BIT
			} else {
				NFD_NO_BIT
			};
			let decomposed_len = if word & no_bit == 0 {
				width
			} else {
				decomposed_utf8_len(cp, word, form.is_compat())
			};
			nfd_len += decomposed_len;
			shrinks |= decomposed_len < width;
		}
		if ccc != 0 && last_ccc > ccc {
			normalized = false;
			reorder = true;
		}

		match form {
			Form::Nfd | Form::Nfkd => {
				let no_bit = if form.is_compat() {
					NFKD_NO_BIT
				} else {
					NFD_NO_BIT
				};
				let order_bit = if form.is_compat() {
					NFKD_DECOMPOSITION_ORDER_BIT
				} else {
					DECOMPOSITION_ORDER_BIT
				};
				let decomposes = word & no_bit != 0;
				decompose |= decomposes;
				reorder |= decomposes && word & order_bit != 0;
				normalized &= !decomposes;
				last_ccc = ccc;
			},
			Form::Nfc | Form::Nfkc => {
				let compat = form.is_compat();
				let excluded = word & if compat { NFKC_NO_BIT } else { NFC_NO_BIT } != 0;
				let maybe = word
					& if compat {
						NFKC_MAYBE_BIT
					} else {
						NFC_MAYBE_BIT
					} != 0;
				let order_bit = if compat {
					NFKD_DECOMPOSITION_ORDER_BIT
				} else {
					DECOMPOSITION_ORDER_BIT
				};
				normalized &= !excluded && !maybe;
				decompose |= excluded;
				reorder |= excluded && word & order_bit != 0;
				if maybe && starter_risky {
					decompose = true;
					reorder = true;
				}
				if ccc == 0 {
					starter_risky = normalization_decomposition(word, compat).is_some();
				}
				last_ccc = ccc;
			},
		}
		at += width;
	}

	if form.is_composed() {
		if decompose {
			(nfd_len, shrinks) = decomposition_workspace(bytes, form.is_compat());
		} else {
			nfd_len = bytes.len();
		}
	}

	Scan { normalized, nfd_len, decompose, reorder, shrinks, stack_order }
}

#[inline]
fn shrink_decompositions(bytes: &mut Vec<u8>, compat: bool) {
	let original_len = bytes.len();
	let mut read = 0;
	let mut write = 0;
	while read < original_len {
		let ascii = ascii_prefix(&bytes[read..original_len]);
		if ascii != 0 {
			if read != write {
				bytes.copy_within(read..read + ascii, write);
			}
			read += ascii;
			write += ascii;
			continue;
		}

		let (cp, width) = decode_utf8(bytes, read);
		let word = normalization_word(cp);
		let decomposed_len = decomposed_utf8_len(cp, word, compat);
		if decomposed_len < width {
			let written = write_decomposition(cp, word, compat, &mut bytes[write..]);
			debug_assert_eq!(written, decomposed_len);
			write += written;
		} else {
			if read != write {
				bytes.copy_within(read..read + width, write);
			}
			write += width;
		}
		read += width;
	}
	bytes.truncate(write);
}

#[inline(always)]
const fn previous_char_start(bytes: &[u8]) -> usize {
	let mut start = bytes.len() - 1;
	while bytes[start] & 0xc0 == 0x80 {
		start -= 1;
	}
	start
}

#[inline]
fn expand_decompositions(bytes: &mut Vec<u8>, required: usize, compat: bool) {
	let source_len = bytes.len();
	debug_assert!(required >= source_len);
	debug_assert!(required <= bytes.capacity());
	let output = bytes.as_mut_ptr();
	let mut read = source_len;
	let mut write = required;

	while read != 0 {
		let ascii = ascii_suffix(&bytes[..read]);
		if ascii != 0 {
			read -= ascii;
			write -= ascii;
			// SAFETY: the source is initialized, the destination is within
			// reserved capacity, and `copy` permits the possible overlap.
			unsafe { ptr::copy(output.add(read), output.add(write), ascii) };
			continue;
		}

		let start = previous_char_start(&bytes[..read]);
		let (cp, width) = decode_utf8(bytes, start);
		debug_assert_eq!(start + width, read);
		let word = normalization_word(cp);
		let decomposed_len = decomposed_utf8_len(cp, word, compat);
		write -= decomposed_len;
		if is_hangul_syllable(cp) || normalization_decomposition(word, compat).is_some() {
			// SAFETY: `write..write + decomposed_len` lies in reserved capacity
			// and cannot overlap unprocessed input because every remaining
			// mapping is non-shrinking after `shrink_decompositions`.
			let written = unsafe { write_decomposition_ptr(cp, word, compat, output.add(write)) };
			debug_assert_eq!(written, decomposed_len);
		} else {
			// SAFETY: the same directional invariant leaves the source initialized
			// until this overlapping copy completes.
			unsafe { ptr::copy(output.add(start), output.add(write), width) };
		}
		read = start;
	}

	debug_assert_eq!(write, 0);
	// SAFETY: every byte in `0..required` was initialized by the backward pass,
	// and canonical decomposition preserves UTF-8 validity.
	unsafe { bytes.set_len(required) };
}

#[inline]
fn decompose_in_place(bytes: &mut Vec<u8>, required: usize, shrinks: bool, compat: bool) {
	if shrinks {
		shrink_decompositions(bytes, compat);
	}
	expand_decompositions(bytes, required, compat);
}

#[inline]
fn canonical_order(bytes: &mut [u8]) {
	let mut at = 0;
	let mut marks_start = 0;
	let mut previous_ccc = 0;
	while at < bytes.len() {
		let ascii = ascii_prefix(&bytes[at..]);
		if ascii != 0 {
			at += ascii;
			marks_start = at;
			previous_ccc = 0;
			continue;
		}

		let (cp, width) = decode_utf8(bytes, at);
		let ccc = combining_class(normalization_word(cp));
		let next = at + width;
		if ccc == 0 {
			marks_start = next;
			previous_ccc = 0;
		} else if ccc < previous_ccc {
			let mut insert = marks_start;
			while insert < at {
				let (prior, prior_width) = decode_utf8(bytes, insert);
				if combining_class(normalization_word(prior)) > ccc {
					break;
				}
				insert += prior_width;
			}
			bytes[insert..next].rotate_right(width);
		} else {
			previous_ccc = ccc;
		}
		at = next;
	}
}

#[derive(Default)]
struct Composition {
	write:    usize,
	starter:  Option<(usize, usize, u32)>,
	last_ccc: u8,
}

#[inline(always)]
fn emit_ascii(bytes: &mut [u8], source: usize, len: usize, state: &mut Composition) {
	bytes.copy_within(source..source + len, state.write);
	state.write += len;
	state.starter = Some((state.write - 1, 1, u32::from(bytes[state.write - 1])));
	state.last_ccc = 0;
}

#[inline]
fn emit_composed(
	bytes: &mut [u8],
	cp: u32,
	ccc: u8,
	source: Option<(usize, usize)>,
	state: &mut Composition,
) {
	let composite = state.starter.and_then(|(_, _, starter_cp)| {
		(state.last_ccc == 0 || state.last_ccc < ccc)
			.then(|| compose_pair(starter_cp, cp))
			.flatten()
	});
	if let (Some(composite), Some((starter_at, starter_width, _))) = (composite, state.starter) {
		let composite_width = utf8_len(composite);
		let tail_start = starter_at + starter_width;
		if composite_width > starter_width {
			let growth = composite_width - starter_width;
			bytes.copy_within(tail_start..state.write, tail_start + growth);
			state.write += growth;
		} else if composite_width < starter_width {
			let shrink = starter_width - composite_width;
			bytes.copy_within(tail_start..state.write, tail_start - shrink);
			state.write -= shrink;
		}
		let written = encode_utf8(composite, &mut bytes[starter_at..]);
		debug_assert_eq!(written, composite_width);
		state.starter = Some((starter_at, composite_width, composite));
		return;
	}

	let width = if let Some((source, width)) = source {
		bytes.copy_within(source..source + width, state.write);
		width
	} else {
		encode_utf8(cp, &mut bytes[state.write..])
	};
	if ccc == 0 {
		state.starter = Some((state.write, width, cp));
	}
	state.write += width;
	state.last_ccc = ccc;
}

#[inline]
fn compose_in_place(bytes: &mut Vec<u8>) {
	let input_len = bytes.len();
	let mut read = 0;
	let mut state = Composition::default();
	while read < input_len {
		let ascii = ascii_prefix(&bytes[read..input_len]);
		if ascii != 0 {
			emit_ascii(bytes, read, ascii, &mut state);
			read += ascii;
			continue;
		}
		let (cp, width) = decode_utf8(bytes, read);
		let ccc = combining_class(normalization_word(cp));
		emit_composed(bytes, cp, ccc, Some((read, width)), &mut state);
		read += width;
	}
	bytes.truncate(state.write);
}

#[inline]
fn order_and_compose_in_place(bytes: &mut Vec<u8>) {
	let input_len = bytes.len();
	let mut read = 0;
	let mut state = Composition::default();
	let mut codepoints = [0; ORDER_STACK];
	let mut classes = [0; ORDER_STACK];

	while read < input_len {
		let ascii = ascii_prefix(&bytes[read..input_len]);
		if ascii != 0 {
			emit_ascii(bytes, read, ascii, &mut state);
			read += ascii;
			continue;
		}

		let mut count = 0;
		while read < input_len && bytes[read] >= 0x80 {
			let (cp, width) = decode_utf8(bytes, read);
			let ccc = combining_class(normalization_word(cp));
			if count != 0 && ccc == 0 {
				break;
			}
			debug_assert!(count < ORDER_STACK);
			codepoints[count] = cp;
			classes[count] = ccc;
			count += 1;
			read += width;
		}

		let marks = usize::from(classes[0] == 0);
		for current in marks + 1..count {
			let cp = codepoints[current];
			let ccc = classes[current];
			let mut insert = current;
			while insert > marks && classes[insert - 1] > ccc {
				codepoints[insert] = codepoints[insert - 1];
				classes[insert] = classes[insert - 1];
				insert -= 1;
			}
			codepoints[insert] = cp;
			classes[insert] = ccc;
		}
		for index in 0..count {
			emit_composed(bytes, codepoints[index], classes[index], None, &mut state);
		}
	}
	bytes.truncate(state.write);
}

#[inline]
fn normalize_scanned(input: &mut String, form: Form, scan: Scan) {
	debug_assert!(!scan.normalized);
	debug_assert!(input.capacity() >= scan.nfd_len);
	// SAFETY: all transformations below preserve scalar boundaries and encode
	// only valid Unicode scalars before the mutable vector borrow ends.
	let bytes = unsafe { input.as_mut_vec() };
	if form.is_composed() && !scan.decompose && scan.reorder && scan.stack_order {
		order_and_compose_in_place(bytes);
		return;
	}
	if scan.decompose {
		decompose_in_place(bytes, scan.nfd_len, scan.shrinks, form.is_compat());
		if scan.reorder {
			canonical_order(bytes);
		}
	} else if scan.reorder {
		canonical_order(bytes);
	}
	if form.is_composed() {
		compose_in_place(bytes);
	}
}

#[inline]
fn make_normalized(input: &mut String, form: Form) -> Result<(), NormalizationError> {
	let scan = scan(input, form);
	if scan.normalized {
		return Ok(());
	}
	if scan.nfd_len > input.capacity() {
		return Err(NormalizationError { required_capacity: scan.nfd_len });
	}
	normalize_scanned(input, form, scan);
	Ok(())
}

#[inline]
fn to_normalized(input: &str, form: Form) -> String {
	let scan = scan(input, form);
	if scan.normalized {
		return input.to_owned();
	}
	let mut output = String::with_capacity(input.len().max(scan.nfd_len));
	output.push_str(input);
	normalize_scanned(&mut output, form, scan);
	output
}

#[inline]
fn into_normalized(mut input: String, form: Form) -> String {
	let scan = scan(&input, form);
	if scan.normalized {
		return input;
	}
	if scan.nfd_len > input.capacity() {
		input.reserve(scan.nfd_len - input.len());
	}
	normalize_scanned(&mut input, form, scan);
	input
}

impl MakeUnicodeNormalized for String {
	#[inline]
	fn make_nfc(&mut self) -> Result<(), NormalizationError> {
		make_normalized(self, Form::Nfc)
	}

	#[inline]
	fn make_nfd(&mut self) -> Result<(), NormalizationError> {
		make_normalized(self, Form::Nfd)
	}

	#[inline]
	fn make_nfkc(&mut self) -> Result<(), NormalizationError> {
		make_normalized(self, Form::Nfkc)
	}

	#[inline]
	fn make_nfkd(&mut self) -> Result<(), NormalizationError> {
		make_normalized(self, Form::Nfkd)
	}
}

impl ToUnicodeNormalized for str {
	#[inline]
	fn to_nfc(&self) -> String {
		to_normalized(self, Form::Nfc)
	}

	#[inline]
	fn to_nfd(&self) -> String {
		to_normalized(self, Form::Nfd)
	}

	#[inline]
	fn to_nfkc(&self) -> String {
		to_normalized(self, Form::Nfkc)
	}

	#[inline]
	fn to_nfkd(&self) -> String {
		to_normalized(self, Form::Nfkd)
	}
}

impl IntoUnicodeNormalized for String {
	#[inline]
	fn into_nfc(self) -> String {
		into_normalized(self, Form::Nfc)
	}

	#[inline]
	fn into_nfd(self) -> String {
		into_normalized(self, Form::Nfd)
	}

	#[inline]
	fn into_nfkc(self) -> String {
		into_normalized(self, Form::Nfkc)
	}

	#[inline]
	fn into_nfkd(self) -> String {
		into_normalized(self, Form::Nfkd)
	}
}
